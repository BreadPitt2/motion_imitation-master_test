"""Interactive visual tuner for a stable Bittle standing pose.

Use sliders in the PyBullet GUI to find a pose that can stand without falling.
Press Ctrl+C in terminal to exit; the final pose is printed for copy/paste.
"""

import argparse
import inspect
import os
import time

import numpy as np
import pybullet as p  # pytype: disable=import-error

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from motion_imitation.envs import env_builder
from motion_imitation.robots import bittle
from motion_imitation.robots import robot_config


def _add_joint_sliders(env, start_pose):
  slider_ids = []
  action_cfg = bittle.ACTION_CONFIG
  for i, cfg in enumerate(action_cfg):
    slider_ids.append(
        p.addUserDebugParameter(
            paramName=cfg.name,
            rangeMin=float(cfg.lower_bound),
            rangeMax=float(cfg.upper_bound),
            startValue=float(start_pose[i])))
  return slider_ids


def _read_sliders(slider_ids):
  return np.array([p.readUserDebugParameter(i) for i in slider_ids],
                  dtype=np.float32)


def _update_status(env, pose):
  rpy = np.asarray(env.robot.GetTrueBaseRollPitchYaw())
  pos = np.asarray(env.robot.GetBasePosition())
  contacts = env.robot.GetFootContacts()
  roll = float(np.degrees(rpy[0]))
  pitch = float(np.degrees(rpy[1]))
  z = float(pos[2])
  contact_count = int(np.sum(np.asarray(contacts, dtype=np.int32)))
  status = "z={:.3f}  roll={:+.1f}  pitch={:+.1f}  contacts={}/4".format(
      z, roll, pitch, contact_count)
  p.removeAllUserDebugItems()
  p.addUserDebugText(
      text=status,
      textPosition=[0.0, 0.0, 0.45],
      textColorRGB=[1, 1, 0],
      textSize=1.6)
  p.addUserDebugText(
      text="Current pose: [{}]".format(
          ", ".join(["{:+.3f}".format(v) for v in pose.tolist()])),
      textPosition=[0.0, 0.0, 0.38],
      textColorRGB=[0.7, 1, 0.7],
      textSize=1.2)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seconds", type=float, default=0.0)
  args = parser.parse_args()

  env = env_builder.build_regular_env(
      robot_class=bittle.Bittle,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      enable_rendering=True,
      wrap_trajectory_generator=False)

  start_pose = getattr(bittle, "STAND_MOTOR_ANGLES", bittle.INIT_MOTOR_ANGLES)
  pose = np.asarray(start_pose, dtype=np.float32)
  slider_ids = _add_joint_sliders(env, pose)
  env.reset(initial_motor_angles=pose, reset_duration=0.0)

  start_t = time.time()
  step_count = 0

  try:
    while True:
      pose = _read_sliders(slider_ids)
      env.step(pose)
      if step_count % 8 == 0:
        _update_status(env, pose)
      step_count += 1

      if args.seconds > 0 and (time.time() - start_t) >= args.seconds:
        break
  except KeyboardInterrupt:
    pass
  finally:
    print("Final tuned pose:")
    print(np.array2string(pose, precision=5, separator=", "))
    env.close()


if __name__ == "__main__":
  main()
