"""Visual standing check for Bittle (no retarget motion, no policy)."""

import argparse
import inspect
import os
import time

import numpy as np

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import env_builder
from motion_imitation.robots import bittle
from motion_imitation.robots import robot_config


def _set_status_text(client, text, color):
  client.removeAllUserDebugItems()
  client.addUserDebugText(
      text=text,
      textPosition=[0.0, 0.0, 0.5],
      textColorRGB=color,
      textSize=1.8)


def run_check(seconds, roll_pitch_limit_deg, min_height, hold_after_seconds):
  env = env_builder.build_regular_env(
      robot_class=bittle.Bittle,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      enable_rendering=True,
      wrap_trajectory_generator=False)

  hold_pose = getattr(bittle, "STAND_MOTOR_ANGLES", bittle.INIT_MOTOR_ANGLES)
  hold_action = np.asarray(hold_pose, dtype=np.float32)
  step_dt = env._env_time_step  # pylint: disable=protected-access

  fail_reason = None

  try:
    env.reset(initial_motor_angles=hold_action, reset_duration=0.0)
    _set_status_text(env.pybullet_client, "Bittle Standing Check: RUNNING", [1, 1, 0])
    total_steps = int(seconds / step_dt)

    for i in range(total_steps):
      _, _, done, _ = env.step(hold_action)
      rpy = np.asarray(env.robot.GetTrueBaseRollPitchYaw())
      pos = np.asarray(env.robot.GetBasePosition())

      roll_deg = float(np.degrees(rpy[0]))
      pitch_deg = float(np.degrees(rpy[1]))
      z = float(pos[2])

      if abs(roll_deg) > roll_pitch_limit_deg or abs(pitch_deg) > roll_pitch_limit_deg:
        fail_reason = "tilt exceeded limit (roll={:.1f}, pitch={:.1f})".format(
            roll_deg, pitch_deg)
        break
      if z < min_height:
        fail_reason = "height below threshold (z={:.3f}m)".format(z)
        break
      if done:
        fail_reason = "env terminated early"
        break

    if fail_reason:
      _set_status_text(
          env.pybullet_client, "Bittle Standing Check: FAIL", [1, 0, 0])
    else:
      _set_status_text(
          env.pybullet_client, "Bittle Standing Check: PASS", [0, 1, 0])

    # Keep the result visible for a short period.
    t_end = time.time() + max(0.0, hold_after_seconds)
    while time.time() < t_end:
      env.step(hold_action)

  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seconds", type=float, default=10.0)
  parser.add_argument("--roll_pitch_limit_deg", type=float, default=35.0)
  parser.add_argument("--min_height", type=float, default=0.10)
  parser.add_argument("--hold_after_seconds", type=float, default=5.0)
  args = parser.parse_args()

  run_check(
      seconds=args.seconds,
      roll_pitch_limit_deg=args.roll_pitch_limit_deg,
      min_height=args.min_height,
      hold_after_seconds=args.hold_after_seconds)


if __name__ == "__main__":
  main()
