#!/usr/bin/env python3
"""Run a trained Bittle policy (.zip) and stream commands to ESP32 over serial.

This script is intended for rack testing first.
Safety behavior:
- Sends reduced-amplitude actions by default.
- Rate limits joint command changes.
- Clips commands away from hard joint limits.
- Uses firmware watchdog timeout (W command).
- Stops immediately if firmware reports a FAULT latch.
"""

import argparse
import collections
import os
import sys
import time
import warnings

import numpy as np

# Silence TensorFlow and related deprecation chatter before importing policy code.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", module=r"tensorflow(\..*)?$")
warnings.filterwarnings("ignore", module=r"tensorboard(\..*)?$")
warnings.filterwarnings("ignore", module=r"stable_baselines(\..*)?$")
warnings.filterwarnings("ignore", message=r".*The TensorFlow contrib module.*")

try:
  import serial
except ImportError as exc:
  raise SystemExit(
      "pyserial is required. Install with: pip install pyserial"
  ) from exc

# Ensure repo root is importable when running as a script.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from motion_imitation import run as mi_run
from motion_imitation.envs import env_builder
from motion_imitation.robots import bittle

try:
  import tensorflow as tf  # type: ignore
  tf.get_logger().setLevel("ERROR")
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
  pass


def _send_line(ser, line):
  ser.write((line.strip() + "\n").encode("utf-8"))


def _parse_float_list(s, n, name):
  parts = [p.strip() for p in s.split(",") if p.strip() != ""]
  if len(parts) != n:
    raise SystemExit(f"{name} must have {n} comma-separated values")
  try:
    vals = [float(x) for x in parts]
  except ValueError as exc:
    raise SystemExit(f"{name} contains non-numeric values") from exc
  return np.asarray(vals, dtype=np.float32)


def _parse_int_list(s, n, name):
  parts = [p.strip() for p in s.split(",") if p.strip() != ""]
  if len(parts) != n:
    raise SystemExit(f"{name} must have {n} comma-separated values")
  try:
    vals = [int(x) for x in parts]
  except ValueError as exc:
    raise SystemExit(f"{name} contains non-integer values") from exc
  if sorted(vals) != list(range(n)):
    raise SystemExit(f"{name} must be a permutation of 0..{n-1}")
  return np.asarray(vals, dtype=np.int32)


def _parse_obs_line(text):
  if not text.startswith("O "):
    return None
  parts = text.split()
  if len(parts) != 13:
    return None
  try:
    vals = [float(x) for x in parts[1:]]
  except ValueError:
    return None
  return np.asarray(vals, dtype=np.float32)


def _drain_serial(ser, verbose=False, max_lines=100):
  """Read available serial lines and return (fault_triggered, lines, latest_obs)."""
  lines = []
  fault = False
  latest_obs = None
  for _ in range(max_lines):
    raw = ser.readline()
    if not raw:
      break
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
      continue
    lines.append(text)
    if verbose:
      print(f"[esp32] {text}")
    if text.startswith("FAULT ") or "ERR fault latched" in text:
      fault = True
    obs_vals = _parse_obs_line(text)
    if obs_vals is not None:
      latest_obs = obs_vals
  return fault, lines, latest_obs


def _build_real_base_obs(latest_obs_packet, imu_hist, action_hist, motor_hist, imu_scales):
  imu_curr = latest_obs_packet[:4] * imu_scales
  motor_curr = latest_obs_packet[4:12]
  imu_hist.appendleft(imu_curr)
  motor_hist.appendleft(motor_curr)
  return np.concatenate(list(imu_hist) + list(action_hist) + list(motor_hist), axis=0)


def _build_policy(model_path, motion_file):
  env = env_builder.build_imitation_env(
      motion_files=[motion_file],
      num_parallel_envs=1,
      mode="test",
      enable_randomizer=False,
      enable_rendering=False,
      robot_class=bittle.Bittle)
  model = mi_run.build_model(
      env=env,
      num_procs=1,
      timesteps_per_actorbatch=mi_run.TIMESTEPS_PER_ACTORBATCH,
      optim_batchsize=mi_run.OPTIM_BATCHSIZE,
      output_dir="")
  model.load_parameters(model_path)
  return model, env


def _make_parser():
  parser = argparse.ArgumentParser(
      description="Run trained Bittle model.zip on physical ESP32 runtime.")
  parser.add_argument("--model", required=True, help="Path to model.zip")
  parser.add_argument(
      "--motion_file",
      default="motion_imitation/data/bittle_motions/trot2_level.txt",
      help="Reference motion used to build the test env observation.")
  parser.add_argument("--port", required=True, help="Serial port (example: COM4)")
  parser.add_argument("--baud", type=int, default=115200)
  parser.add_argument("--hz", type=float, default=25.0, help="Command streaming rate")
  parser.add_argument(
      "--duration_s", type=float, default=30.0,
      help="Run duration in seconds. <=0 means run until Ctrl+C.")
  parser.add_argument(
      "--action_gain", type=float, default=1.0,
      help=("Scales policy action offset (rad) before adding init pose. "
            "For this repo's MotorPoseOffsetGenerator policy, 1.0 is nominal."))
  parser.add_argument(
      "--action_ema", type=float, default=0.6,
      help=("Exponential moving average factor for action smoothing. "
            "0.0=no smoothing, closer to 1.0=more smoothing."))
  parser.add_argument(
      "--max_delta_rad", type=float, default=0.04,
      help="Max joint rad change per command (rate limiter).")
  parser.add_argument(
      "--joint_margin_rad", type=float, default=0.08,
      help="Keep this margin from action bounds for safety.")
  parser.add_argument(
      "--joint_signs",
      default="1,-1,1,-1,1,1,1,1",
      help="Per-joint action sign in policy order (8 comma-separated values).")
  parser.add_argument(
      "--init_pose_rad",
      default="0,0.67,0,0.67,0,0.67,0,0.67",
      help=("Override base pose in policy order, 8 comma-separated rad values. "
            "Default uses bittle.INIT_MOTOR_ANGLES."))
  parser.add_argument(
      "--joint_bias_rad",
      default="0,0,0,0,0,0,0,0",
      help="Per-joint additive bias in rad after init pose (8 comma-separated values).")
  parser.add_argument(
      "--joint_perm",
      default="0,1,2,3,4,5,6,7",
      help="Permutation from model action index to policy index (8 comma-separated ints).")
  parser.add_argument(
      "--imu_scales",
      default="1,1,1,1",
      help=("Scale factors for IMU channels [R, P, dR, dP] from firmware packet "
            "(4 comma-separated values)."))
  parser.add_argument(
      "--watchdog_ms", type=int, default=350,
      help="Firmware watchdog timeout via 'W <ms>' command.")
  parser.add_argument(
      "--obs_timeout_s", type=float, default=0.6,
      help=("For real-observation modes: timeout in seconds before declaring "
            "observation stream stale."))
  parser.add_argument(
      "--obs_mode", choices=["sim", "mixed", "real"], default="mixed",
      help=("Observation source for policy input: "
            "sim=simulated obs only, real=live firmware O packets only, "
            "mixed=prefer live O packets and fall back to sim if stale."))
  parser.add_argument(
      "--serial_warmup_s", type=float, default=1.2,
      help="Delay after opening serial port before sending commands.")
  parser.add_argument("--verbose_serial", action="store_true")
  return parser


def main():
  args = _make_parser().parse_args()

  if not os.path.isfile(args.model):
    raise SystemExit(f"Model file not found: {args.model}")
  if not os.path.isfile(args.motion_file):
    raise SystemExit(f"Motion file not found: {args.motion_file}")
  if args.hz <= 1.0:
    raise SystemExit("--hz must be > 1.0")
  if args.max_delta_rad <= 0:
    raise SystemExit("--max_delta_rad must be > 0")
  if not (0.0 <= args.action_ema < 1.0):
    raise SystemExit("--action_ema must be in [0.0, 1.0)")
  if args.joint_margin_rad < 0:
    raise SystemExit("--joint_margin_rad must be >= 0")
  if args.watchdog_ms < 100:
    raise SystemExit("--watchdog_ms must be >= 100")
  if args.obs_timeout_s <= 0:
    raise SystemExit("--obs_timeout_s must be > 0")

  print("Loading policy...")
  model, env = _build_policy(args.model, args.motion_file)

  lower = np.array([cfg.lower_bound for cfg in bittle.ACTION_CONFIG], dtype=np.float32)
  upper = np.array([cfg.upper_bound for cfg in bittle.ACTION_CONFIG], dtype=np.float32)
  init_pose = np.array(bittle.INIT_MOTOR_ANGLES, dtype=np.float32)
  if args.init_pose_rad.strip():
    init_pose = _parse_float_list(args.init_pose_rad, 8, "--init_pose_rad")
  joint_signs = _parse_float_list(args.joint_signs, 8, "--joint_signs")
  joint_bias = _parse_float_list(args.joint_bias_rad, 8, "--joint_bias_rad")
  joint_perm = _parse_int_list(args.joint_perm, 8, "--joint_perm")
  imu_scales = _parse_float_list(args.imu_scales, 4, "--imu_scales")

  print("Using mapping:")
  print(f"  init_pose_rad={init_pose.tolist()}")
  print(f"  joint_signs={joint_signs.tolist()}")
  print(f"  joint_bias_rad={joint_bias.tolist()}")
  print(f"  joint_perm={joint_perm.tolist()}")
  print(f"  imu_scales={imu_scales.tolist()}")

  obs = env.reset()
  try:
    base_obs_dim = int(env._gym_env.observation_space.shape[0])
  except Exception:
    base_obs_dim = 60
  last_cmd = init_pose.copy()
  filt_action = np.zeros_like(init_pose)
  imu_hist = collections.deque(
      [np.zeros(4, dtype=np.float32) for _ in range(3)], maxlen=3)
  action_hist = collections.deque(
      [np.zeros(8, dtype=np.float32) for _ in range(3)], maxlen=3)
  motor_hist = collections.deque(
      [init_pose.copy() for _ in range(3)], maxlen=3)
  latest_obs_packet = None
  last_obs_time = time.monotonic()
  warned_stale_obs = False
  warned_obs_dim = False
  period = 1.0 / args.hz
  end_time = time.monotonic() + args.duration_s if args.duration_s > 0 else None

  print(f"Opening serial {args.port} @ {args.baud}...")
  with serial.Serial(args.port, args.baud, timeout=0.0, write_timeout=0.2) as ser:
    time.sleep(args.serial_warmup_s)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Safe startup sequence.
    _send_line(ser, "C")
    _send_line(ser, "E")
    _send_line(ser, f"W {args.watchdog_ms}")
    _send_line(ser, "V 0")
    if args.obs_mode == "sim":
      _send_line(ser, "U 0")
    else:
      _send_line(ser, "U 1")
      _send_line(ser, "O")
    _send_line(ser, "S")
    time.sleep(0.35)
    fault, _, obs_pkt = _drain_serial(ser, verbose=args.verbose_serial)
    if fault:
      raise RuntimeError("Firmware fault detected during startup.")
    if obs_pkt is not None:
      latest_obs_packet = obs_pkt
      last_obs_time = time.monotonic()

    print("Streaming policy commands. Press Ctrl+C to stop.")
    next_tick = time.monotonic()
    step_count = 0
    try:
      while True:
        now = time.monotonic()
        if end_time is not None and now >= end_time:
          print("Duration reached, stopping.")
          break

        # Refresh serial input before policy step.
        fault, _, obs_pkt = _drain_serial(ser, verbose=args.verbose_serial)
        if fault:
          print("Firmware fault detected. Stopping stream.")
          break

        if args.obs_mode != "sim":
          if obs_pkt is not None:
            latest_obs_packet = obs_pkt
            last_obs_time = now
            warned_stale_obs = False

            real_base_obs = _build_real_base_obs(
                latest_obs_packet, imu_hist, action_hist, motor_hist, imu_scales)
            if real_base_obs.shape[0] == base_obs_dim:
              obs = np.concatenate([real_base_obs, obs[base_obs_dim:]], axis=0)
            elif not warned_obs_dim:
              warned_obs_dim = True
              print(
                  f"WARN real_obs_dim={real_base_obs.shape[0]} expected={base_obs_dim}; "
                  "ignoring real obs and using sim obs")
          else:
            stale_s = now - last_obs_time
            if args.obs_mode == "real" and stale_s > args.obs_timeout_s:
              print("No observation packet received in time. Stopping stream for safety.")
              break
            if args.obs_mode == "mixed" and stale_s > args.obs_timeout_s and not warned_stale_obs:
              warned_stale_obs = True
              print("WARN real observation stream stale; temporarily using sim observation")

        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != init_pose.shape[0]:
          raise RuntimeError(
              f"Unexpected action size {action.shape[0]}, expected {init_pose.shape[0]}")

        # Rack-safe action shaping.
        action = np.clip(action, -1.0, 1.0) * float(args.action_gain)
        filt_action = args.action_ema * filt_action + (1.0 - args.action_ema) * action
        # Apply optional remap/sign correction to align model convention with hardware.
        action_hw = filt_action[joint_perm] * joint_signs
        cmd = init_pose + action_hw + joint_bias
        cmd = np.clip(cmd, lower + args.joint_margin_rad, upper - args.joint_margin_rad)
        cmd = np.clip(cmd, last_cmd - args.max_delta_rad, last_cmd + args.max_delta_rad)

        cmd_line = "N " + " ".join(f"{v:.5f}" for v in cmd.tolist())
        _send_line(ser, cmd_line)
        last_cmd = cmd
        action_hist.appendleft(filt_action.copy())

        # Keep env rollout coherent with the command stream.
        obs, _, done, _ = env.step(filt_action)
        if done:
          obs = env.reset()

        step_count += 1
        next_tick += period
        sleep_s = next_tick - time.monotonic()
        if sleep_s > 0:
          time.sleep(sleep_s)
        else:
          next_tick = time.monotonic()
    except KeyboardInterrupt:
      print("Interrupted by user.")
    finally:
      # Safe shutdown.
      try:
        _send_line(ser, "U 0")
        _send_line(ser, "S")
        time.sleep(0.2)
        _send_line(ser, "P")
      except Exception:
        pass
      _drain_serial(ser, verbose=args.verbose_serial)
      print(f"Done. Steps sent: {step_count}")


if __name__ == "__main__":
  main()
