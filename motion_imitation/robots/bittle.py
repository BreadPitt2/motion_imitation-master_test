# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyBullet simulation wrapper for the Bittle robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import re
import numpy as np
import pybullet as pyb  # pytype: disable=import-error

from motion_imitation.robots import minitaur
from motion_imitation.robots import minitaur_motor
from motion_imitation.robots import minitaur_constants
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config

NUM_MOTORS = 8
NUM_LEGS = 4
DOFS_PER_LEG = 2

MOTOR_NAMES = [
    "left-back-shoulder-joint",
    "left-back-knee-joint",
    "left-front-shoulder-joint",
    "left-front-knee-joint",
    "right-back-shoulder-joint",
    "right-back-knee-joint",
    "right-front-shoulder-joint",
    "right-front-knee-joint",
]

INIT_RACK_POSITION = [0, 0, 1.0]
INIT_POSITION = [0, 0, 0.88]
JOINT_DIRECTIONS = np.ones(NUM_MOTORS)
JOINT_OFFSETS = np.zeros(NUM_MOTORS)

# Matches retarget config ordering in retarget_motion/retarget_config_bittle.py.
INIT_MOTOR_ANGLES = np.array([0.0, -0.67,
                              0.0, -0.67,
                              0.0, 0.67,
                              0.0, 0.67])

# Dedicated standing pose used during reset/settle. Keep this separate from
# INIT_MOTOR_ANGLES so imitation action offsets remain unchanged.
STAND_MOTOR_ANGLES = np.array([1.02883, -1.05,
                               0.0, -1.05,
                               -1.02883, 1.05,
                               0.0, 1.05])

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.3

# Rough hip locations in base frame, used by generic tooling.
_DEFAULT_HIP_POSITIONS = (
    (-0.446, 0.523, -0.021),   # left front
    (-0.446, -0.519, -0.021),  # left back
    (0.452, 0.523, -0.021),    # right front
    (0.452, -0.519, -0.021),   # right back
)

SHOULDER_P_GAIN = 80.0
SHOULDER_D_GAIN = 1.5
KNEE_P_GAIN = 80.0
KNEE_D_GAIN = 1.5

_SHOULDER_NAME_PATTERN = re.compile(r"\w+-\w+-shoulder-joint")
_KNEE_NAME_PATTERN = re.compile(r"\w+-\w+-knee-joint")
_TOE_NAME_PATTERN = re.compile(r"toe\w*_joint")

URDF_FILENAME = "bittle/urdf/bittle_toes.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

SHOULDER_LOWER_BOUND = -1.57079632679
SHOULDER_UPPER_BOUND = 1.2217304764
KNEE_LOWER_BOUND = -1.2217304764
KNEE_UPPER_BOUND = 1.4835298642

ACTION_CONFIG = [
    locomotion_gym_config.ScalarField(
        name="left_back_shoulder",
        upper_bound=SHOULDER_UPPER_BOUND,
        lower_bound=SHOULDER_LOWER_BOUND),
    locomotion_gym_config.ScalarField(
        name="left_back_knee",
        upper_bound=KNEE_UPPER_BOUND,
        lower_bound=KNEE_LOWER_BOUND),
    locomotion_gym_config.ScalarField(
        name="left_front_shoulder",
        upper_bound=SHOULDER_UPPER_BOUND,
        lower_bound=SHOULDER_LOWER_BOUND),
    locomotion_gym_config.ScalarField(
        name="left_front_knee",
        upper_bound=KNEE_UPPER_BOUND,
        lower_bound=KNEE_LOWER_BOUND),
    locomotion_gym_config.ScalarField(
        name="right_back_shoulder",
        upper_bound=SHOULDER_UPPER_BOUND,
        lower_bound=SHOULDER_LOWER_BOUND),
    locomotion_gym_config.ScalarField(
        name="right_back_knee",
        upper_bound=KNEE_UPPER_BOUND,
        lower_bound=KNEE_LOWER_BOUND),
    locomotion_gym_config.ScalarField(
        name="right_front_shoulder",
        upper_bound=SHOULDER_UPPER_BOUND,
        lower_bound=SHOULDER_LOWER_BOUND),
    locomotion_gym_config.ScalarField(
        name="right_front_knee",
        upper_bound=KNEE_UPPER_BOUND,
        lower_bound=KNEE_LOWER_BOUND),
]


class Bittle(minitaur.Minitaur):
  """A simulation wrapper for the Bittle quadruped."""

  MPC_BODY_MASS = 2.0
  MPC_BODY_INERTIA = (0.02, 0, 0, 0, 0.02, 0, 0, 0, 0.02)
  MPC_BODY_HEIGHT = 0.2
  ACTION_CONFIG = ACTION_CONFIG

  def __init__(
      self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      sensors=None,
      control_latency=0.002,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=False,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      reset_time=1,
      allow_knee_contact=False):
    self._urdf_filename = urdf_filename
    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands

    motor_kp = [
        SHOULDER_P_GAIN, KNEE_P_GAIN,
        SHOULDER_P_GAIN, KNEE_P_GAIN,
        SHOULDER_P_GAIN, KNEE_P_GAIN,
        SHOULDER_P_GAIN, KNEE_P_GAIN,
    ]
    motor_kd = [
        SHOULDER_D_GAIN, KNEE_D_GAIN,
        SHOULDER_D_GAIN, KNEE_D_GAIN,
        SHOULDER_D_GAIN, KNEE_D_GAIN,
        SHOULDER_D_GAIN, KNEE_D_GAIN,
    ]

    super(Bittle, self).__init__(
        pybullet_client=pybullet_client,
        time_step=time_step,
        action_repeat=action_repeat,
        num_motors=NUM_MOTORS,
        dofs_per_leg=DOFS_PER_LEG,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=JOINT_OFFSETS,
        motor_overheat_protection=False,
        motor_control_mode=motor_control_mode,
        motor_model_class=minitaur_motor.MotorModel,
        sensors=sensors,
        motor_kp=motor_kp,
        motor_kd=motor_kd,
        control_latency=control_latency,
        on_rack=on_rack,
        enable_action_interpolation=enable_action_interpolation,
        enable_action_filter=enable_action_filter,
        reset_time=reset_time)

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    self.ReceiveObservation()
    if reset_time <= 0:
      return

    for _ in range(300):
      self._StepInternal(
          STAND_MOTOR_ANGLES,
          motor_control_mode=robot_config.MotorControlMode.POSITION)

    if default_motor_angles is not None:
      num_steps_to_reset = int(reset_time / self.time_step)
      for _ in range(num_steps_to_reset):
        self._StepInternal(
            default_motor_angles,
            motor_control_mode=robot_config.MotorControlMode.POSITION)

  def GetHipPositionsInBaseFrame(self):
    return _DEFAULT_HIP_POSITIONS

  def GetFootContacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

    contacts = [False, False, False, False]
    for contact in all_contacts:
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue
      try:
        toe_link_index = self._foot_link_ids.index(contact[_LINK_A_FIELD_NUMBER])
        contacts[toe_link_index] = True
      except ValueError:
        continue

    return contacts

  def ResetPose(self, add_constraint):
    del add_constraint
    for joint_name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[joint_name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=joint_id,
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)

    for i, motor_name in enumerate(MOTOR_NAMES):
      self._pybullet_client.resetJointState(
          self.quadruped,
          self._joint_name_to_id[motor_name],
          STAND_MOTOR_ANGLES[i],
          targetVelocity=0)

  def GetURDFFile(self):
    return self._urdf_filename

  def _BuildUrdfIds(self):
    num_joints = self.pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._lower_link_ids = []
    self._foot_link_ids = []

    for i in range(num_joints):
      joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      joint_type = joint_info[2]

      if _SHOULDER_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      elif _KNEE_NAME_PATTERN.match(joint_name):
        self._lower_link_ids.append(joint_id)
      elif _TOE_NAME_PATTERN.match(joint_name):
        self._foot_link_ids.append(joint_id)
      elif joint_type == self.pybullet_client.JOINT_FIXED:
        self._chassis_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)

    self._leg_link_ids.extend(self._lower_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)
    if self._allow_knee_contact:
      self._foot_link_ids.extend(self._lower_link_ids)

    self._chassis_link_ids.sort()
    self._motor_link_ids.sort()
    self._lower_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    return pyb.getQuaternionFromEuler([0, 0, -math.pi / 2.0])

  def GetDefaultInitPosition(self):
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    return (STAND_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS

  def ApplyAction(self, motor_commands, motor_control_mode=None):
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)
    super(Bittle, self).ApplyAction(motor_commands, motor_control_mode)

  def _ClipMotorCommands(self, motor_commands):
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles = self.GetMotorAngles()
    return np.clip(motor_commands,
                   current_motor_angles - max_angle_change,
                   current_motor_angles + max_angle_change)

  @classmethod
  def GetConstants(cls):
    del cls
    return minitaur_constants
