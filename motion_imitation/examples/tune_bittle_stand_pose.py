"""Interactive visual tuner for a stable Bittle standing pose.

Use sliders in the PyBullet GUI to find a pose that can stand without falling.
Press Ctrl+C in terminal to exit; the final pose is printed for copy/paste.
"""

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

_STATUS_TEXT_IDS = {"summary": -1, "pose": -1, "joint_lines": []}


def _add_joint_sliders(client, start_pose):
  slider_ids = []
  action_cfg = bittle.ACTION_CONFIG
  for i, cfg in enumerate(action_cfg):
    slider_ids.append(
        client.addUserDebugParameter(
            paramName=cfg.name,
            rangeMin=float(cfg.lower_bound),
            rangeMax=float(cfg.upper_bound),
            startValue=float(start_pose[i])))
  return slider_ids


def _read_sliders(client, slider_ids):
  return np.array([client.readUserDebugParameter(i) for i in slider_ids],
                  dtype=np.float32)


def _update_status(client, env, pose):
  rpy = np.asarray(env.robot.GetTrueBaseRollPitchYaw())
  q = np.asarray(env.robot.GetTrueMotorAngles())
  pos = np.asarray(env.robot.GetBasePosition())
  contacts = env.robot.GetFootContacts()
  roll = float(np.degrees(rpy[0]))
  pitch = float(np.degrees(rpy[1]))
  z = float(pos[2])
  contact_count = int(np.sum(np.asarray(contacts, dtype=np.int32)))
  status = "z={:.3f}  roll={:+.1f}  pitch={:+.1f}  contacts={}/4".format(
      z, roll, pitch, contact_count)
  _STATUS_TEXT_IDS["summary"] = client.addUserDebugText(
      text=status,
      textPosition=[0.0, 0.0, 0.45],
      textColorRGB=[1, 1, 0],
      textSize=1.6,
      replaceItemUniqueId=_STATUS_TEXT_IDS["summary"])
  _STATUS_TEXT_IDS["pose"] = client.addUserDebugText(
      text="Current pose: [{}]".format(
          ", ".join(["{:+.3f}".format(v) for v in pose.tolist()])),
      textPosition=[0.0, 0.0, 0.38],
      textColorRGB=[0.7, 1, 0.7],
      textSize=1.2,
      replaceItemUniqueId=_STATUS_TEXT_IDS["pose"])

  joint_names = [cfg.name for cfg in bittle.ACTION_CONFIG]
  if len(_STATUS_TEXT_IDS["joint_lines"]) != len(joint_names):
    _STATUS_TEXT_IDS["joint_lines"] = [-1] * len(joint_names)

  for i, name in enumerate(joint_names):
    err = float(pose[i] - q[i])
    color = [0.6, 1.0, 0.6] if abs(err) < 0.15 else [1.0, 0.5, 0.4]
    line = "{} cmd={:+.2f} act={:+.2f} err={:+.2f}".format(
        name, float(pose[i]), float(q[i]), err)
    _STATUS_TEXT_IDS["joint_lines"][i] = client.addUserDebugText(
        text=line,
        textPosition=[0.0, 0.0, 0.32 - 0.03 * i],
        textColorRGB=color,
        textSize=1.0,
        replaceItemUniqueId=_STATUS_TEXT_IDS["joint_lines"][i])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seconds", type=float, default=0.0)
  parser.add_argument("--on_rack", action="store_true", default=False)
  args = parser.parse_args()

  env = env_builder.build_regular_env(
      robot_class=bittle.Bittle,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      enable_rendering=True,
      on_rack=args.on_rack,
      wrap_trajectory_generator=False)
  client = env.pybullet_client

  start_pose = getattr(bittle, "STAND_MOTOR_ANGLES", bittle.INIT_MOTOR_ANGLES)
  pose = np.asarray(start_pose, dtype=np.float32)
  slider_ids = _add_joint_sliders(client, pose)
  env.reset(initial_motor_angles=pose, reset_duration=0.0)

  start_t = time.time()
  step_count = 0

  try:
    while True:
      pose = _read_sliders(client, slider_ids)
      env.step(pose)
      if step_count % 8 == 0:
        _update_status(client, env, pose)
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
