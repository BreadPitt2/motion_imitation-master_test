#!/usr/bin/env python3
import argparse
import inspect
import os
import time

import pybullet as pyb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from motion_imitation.utilities.motion_data import MotionData


def set_pose(robot, pose):
  pyb.resetBasePositionAndOrientation(robot, pose[0:3], pose[3:7])

  for joint_id in range(pyb.getNumJoints(robot)):
    joint_info = pyb.getJointInfo(robot, joint_id)
    joint_state = pyb.getJointStateMultiDof(robot, joint_id)

    pose_index = joint_info[3]
    pose_size = len(joint_state[0])
    velocity_size = len(joint_state[1])
    if pose_size > 0:
      pyb.resetJointStateMultiDof(
          robot,
          joint_id,
          pose[pose_index:pose_index + pose_size],
          [0.0] * velocity_size)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--motion_file",
      type=str,
      default="motion_imitation/data/bittle_motions/trot2_level.txt")
  parser.add_argument(
      "--urdf_file",
      type=str,
      default="bittle/urdf/bittle_toes.urdf")
  parser.add_argument(
      "--camera_distance",
      type=float,
      default=1.2)
  parser.add_argument(
      "--camera_yaw",
      type=float,
      default=35.0)
  parser.add_argument(
      "--camera_pitch",
      type=float,
      default=-20.0)
  parser.add_argument(
      "--camera_target_x",
      type=float,
      default=0.0)
  parser.add_argument(
      "--camera_target_y",
      type=float,
      default=0.0)
  parser.add_argument(
      "--camera_target_z",
      type=float,
      default=0.25)
  parser.add_argument(
      "--loops",
      type=int,
      default=-1,
      help="-1 means infinite loops")
  args = parser.parse_args()

  motion = MotionData(args.motion_file)
  frames = motion.get_frames()
  frame_duration = motion.get_frame_duration()

  pyb.connect(pyb.GUI)
  robot = pyb.loadURDF(args.urdf_file)
  pyb.resetDebugVisualizerCamera(
      cameraDistance=args.camera_distance,
      cameraYaw=args.camera_yaw,
      cameraPitch=args.camera_pitch,
      cameraTargetPosition=[
          args.camera_target_x,
          args.camera_target_y,
          args.camera_target_z
      ])

  loop_count = 0
  while args.loops < 0 or loop_count < args.loops:
    for frame in frames:
      set_pose(robot, frame)
      time.sleep(frame_duration)
    loop_count += 1

  pyb.disconnect()


if __name__ == "__main__":
  main()
