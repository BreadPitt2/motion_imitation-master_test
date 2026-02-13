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


def _summarize(frames):
  root_y = frames[:, 1]
  root_quat = frames[:, 3:7]
  roll = np.array([_roll_from_quat_xyzw(q) for q in root_quat], dtype=np.float64)
  return {
      "mean_roll_rad": float(roll.mean()),
      "mean_roll_deg": float(roll.mean() * 180.0 / math.pi),
      "mean_root_y": float(root_y.mean()),
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
    center_roll=True):
  with open(input_path, "r") as file_obj:
    motion_json = json.load(file_obj)

  frames = np.array(motion_json["Frames"], dtype=np.float64)
  if frames.shape[1] < 7:
    raise ValueError("Unexpected frame size: need at least root pos + root quat.")

  before = _summarize(frames)

  mean_root_y = before["mean_root_y"]
  mean_roll = before["mean_roll_rad"]

  if center_lateral:
    frames[:, 1] -= mean_root_y

  if center_roll:
    roll_correction = transformations.quaternion_about_axis(-mean_roll, [1, 0, 0])
    for index in range(frames.shape[0]):
      q = frames[index, 3:7]
      q_new = transformations.quaternion_multiply(q, roll_correction)
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
  args = parser.parse_args()

  level_motion_clip(
      input_path=args.input,
      output_path=args.output,
      center_lateral=not args.no_center_lateral,
      center_roll=not args.no_center_roll)


if __name__ == "__main__":
  main()
