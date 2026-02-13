#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np
from pybullet_utils import transformations


def _roll_from_quat_xyzw(quat):
  x, y, z, w = quat
  t0 = 2.0 * (w * x + y * z)
  t1 = 1.0 - 2.0 * (x * x + y * y)
  return math.atan2(t0, t1)


def _yaw_from_quat_xyzw(quat):
  x, y, z, w = quat
  t0 = 2.0 * (w * z + x * y)
  t1 = 1.0 - 2.0 * (y * y + z * z)
  return math.atan2(t0, t1)


def _normalize_angle(angle):
  return math.atan2(math.sin(angle), math.cos(angle))


def _summarize(frames):
  root_y = frames[:, 1]
  root_quat = frames[:, 3:7]
  roll = np.array([_roll_from_quat_xyzw(q) for q in root_quat], dtype=np.float64)
  yaw = np.array([_yaw_from_quat_xyzw(q) for q in root_quat], dtype=np.float64)
  cycle_delta_y = float(frames[-1, 1] - frames[0, 1])
  cycle_delta_yaw = float(_normalize_angle(yaw[-1] - yaw[0]))
  return {
      "mean_roll_rad": float(roll.mean()),
      "mean_roll_deg": float(roll.mean() * 180.0 / math.pi),
      "mean_root_y": float(root_y.mean()),
      "cycle_delta_y": cycle_delta_y,
      "cycle_delta_yaw_rad": cycle_delta_yaw,
      "cycle_delta_yaw_deg": float(cycle_delta_yaw * 180.0 / math.pi),
  }


def _standardize_quaternion(quat):
  quat = np.array(quat, dtype=np.float64)
  norm = np.linalg.norm(quat)
  if norm > 0:
    quat = quat / norm
  if quat[3] < 0:
    quat = -quat
  return quat


def level_motion_clip(
    input_path,
    output_path,
    center_lateral=True,
    center_roll=True,
    remove_cycle_lateral_drift=False,
    remove_cycle_yaw_drift=False):
  with open(input_path, "r") as file_obj:
    motion_json = json.load(file_obj)

  frames = np.array(motion_json["Frames"], dtype=np.float64)
  if frames.shape[1] < 7:
    raise ValueError("Unexpected frame size: need at least root pos + root quat.")

  before = _summarize(frames)

  mean_root_y = before["mean_root_y"]
  mean_roll = before["mean_roll_rad"]
  num_frames = frames.shape[0]
  phase = np.linspace(0.0, 1.0, num_frames) if num_frames > 1 else np.zeros(num_frames)

  if center_lateral:
    frames[:, 1] -= mean_root_y

  if center_roll:
    roll_correction = transformations.quaternion_about_axis(-mean_roll, [1, 0, 0])
    for index in range(frames.shape[0]):
      q = frames[index, 3:7]
      q_new = transformations.quaternion_multiply(q, roll_correction)
      frames[index, 3:7] = _standardize_quaternion(q_new)

  if remove_cycle_yaw_drift:
    start_yaw = _yaw_from_quat_xyzw(frames[0, 3:7])
    end_yaw = _yaw_from_quat_xyzw(frames[-1, 3:7])
    cycle_delta_yaw = _normalize_angle(end_yaw - start_yaw)
    origin_xy = frames[0, 0:2].copy()

    for index in range(num_frames):
      correction_angle = -phase[index] * cycle_delta_yaw
      yaw_correction = transformations.quaternion_about_axis(correction_angle, [0, 0, 1])
      q = frames[index, 3:7]
      q_new = transformations.quaternion_multiply(q, yaw_correction)
      frames[index, 3:7] = _standardize_quaternion(q_new)

      rel_x = frames[index, 0] - origin_xy[0]
      rel_y = frames[index, 1] - origin_xy[1]
      cos_a = math.cos(correction_angle)
      sin_a = math.sin(correction_angle)
      frames[index, 0] = origin_xy[0] + cos_a * rel_x - sin_a * rel_y
      frames[index, 1] = origin_xy[1] + sin_a * rel_x + cos_a * rel_y

  if remove_cycle_lateral_drift:
    cycle_delta_y = frames[-1, 1] - frames[0, 1]
    frames[:, 1] -= phase * cycle_delta_y

  if center_lateral:
    frames[:, 1] -= float(frames[:, 1].mean())

  if center_roll:
    mean_roll_after = float(np.array(
        [_roll_from_quat_xyzw(q) for q in frames[:, 3:7]], dtype=np.float64).mean())
    final_roll_correction = transformations.quaternion_about_axis(-mean_roll_after, [1, 0, 0])
    for index in range(num_frames):
      q = frames[index, 3:7]
      q_new = transformations.quaternion_multiply(q, final_roll_correction)
      frames[index, 3:7] = _standardize_quaternion(q_new)

  motion_json["Frames"] = frames.tolist()
  Path(output_path).parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w") as file_obj:
    json.dump(motion_json, file_obj, indent=2)
    file_obj.write("\n")

  after = _summarize(frames)
  print("Input:", input_path)
  print("Output:", output_path)
  print("Before:", before)
  print("After:", after)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, type=str)
  parser.add_argument("--output", required=True, type=str)
  parser.add_argument("--no_center_lateral", action="store_true")
  parser.add_argument("--no_center_roll", action="store_true")
  parser.add_argument("--remove_cycle_lateral_drift", action="store_true")
  parser.add_argument("--remove_cycle_yaw_drift", action="store_true")
  args = parser.parse_args()

  level_motion_clip(
      input_path=args.input,
      output_path=args.output,
      center_lateral=not args.no_center_lateral,
      center_roll=not args.no_center_roll,
      remove_cycle_lateral_drift=args.remove_cycle_lateral_drift,
      remove_cycle_yaw_drift=args.remove_cycle_yaw_drift)


if __name__ == "__main__":
  main()
