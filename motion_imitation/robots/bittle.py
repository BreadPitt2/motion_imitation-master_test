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

"""PyBullet simulation model for a Bittle robot."""

import math
import numpy as np
import pybullet as pyb  # pytype: disable=import-error
import re

from motion_imitation.envs import locomotion_gym_config
from motion_imitation.robots import minitaur
from motion_imitation.robots import minitaur_motor
from motion_imitation.robots import robot_config


NUM_MOTORS = 8
NUM_LEGS = 4
DOFS_PER_LEG = 2

# Joint order follows the existing Bittle retarget outputs and default joint
# pose in retarget_config_bittle.py:
# [LB_shoulder, LB_knee, LF_shoulder, LF_knee, RB_shoulder, RB_knee, RF_shoulder, RF_knee]
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

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.25]

JOINT_DIRECTIONS = np.ones(NUM_MOTORS)
JOINT_OFFSETS = np.zeros(NUM_MOTORS)

# Matches current retarget default used to keep knee bend direction stable.
INIT_MOTOR_ANGLES = np.array([0.0, -0.67,
                              0.0, -0.67,
                              0.0, 0.67,
                              0.0, 0.67])

SHOULDER_P_GAIN = 60.0
SHOULDER_D_GAIN = 1.0
KNEE_P_GAIN = 60.0
KNEE_D_GAIN = 1.0

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2

# Approximate shoulder locations in base frame.
_DEFAULT_HIP_POSITIONS = (
    (-0.446, -0.519, 0.0),  # LB
    (-0.446, 0.523, 0.0),   # LF
    (0.451, -0.519, 0.0),   # RB
    (0.451, 0.523, 0.0),    # RF
)

URDF_FILENAME = "bittle/urdf/bittle_toes.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

_SHOULDER_NAME_PATTERN = re.compile(r".*-shoulder-joint$")
_KNEE_NAME_PATTERN = re.compile(r".*-knee-joint$")
_TOE_NAME_PATTERN = re.compile(r"toe[A-Z]{2}_joint$")

UPPER_BOUND = 6.28318548203
LOWER_BOUND = -6.28318548203


class Bittle(minitaur.Minitaur):
  """A simulation for the Bittle robot."""

  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name="motor_angle_0",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_1",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_2",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_3",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_4",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_5",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_6",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
      locomotion_gym_config.ScalarField(name="motor_angle_7",
                                        upper_bound=UPPER_BOUND,
                                        lower_bound=LOWER_BOUND),
  ]

  def __init__(self,
               pybullet_client,
               motor_control_mode,
               urdf_filename=URDF_FILENAME,
               enable_clip_motor_commands=False,
               time_step=0.001,
               action_repeat=33,
               sensors=None,
               control_latency=0.002,
               on_rack=False,
               enable_action_interpolation=True,
               enable_action_filter=False,
               reset_time=-1,
               allow_knee_contact=False):
    self._urdf_filename = urdf_filename
    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands
    self._knee_link_ids = []
    self._toe_link_ids = []

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

  def _LoadRobotURDF(self):
    bittle_urdf_path = self.GetURDFFile()
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          bittle_urdf_path,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          bittle_urdf_path,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    self.ReceiveObservation()
    if reset_time <= 0:
      return

    for _ in range(500):
      self._StepInternal(
          INIT_MOTOR_ANGLES,
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
    contacts = [False] * NUM_LEGS

    toe_by_leg = self._toe_link_ids[:]
    knee_by_leg = self._knee_link_ids[:]
    if len(toe_by_leg) != NUM_LEGS or len(knee_by_leg) != NUM_LEGS:
      return contacts

    for contact in all_contacts:
      # Ignore self contacts.
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue

      link_id = contact[_LINK_A_FIELD_NUMBER]
      for leg_id in range(NUM_LEGS):
        if link_id == toe_by_leg[leg_id]:
          contacts[leg_id] = True
        elif self._allow_knee_contact and link_id == knee_by_leg[leg_id]:
          contacts[leg_id] = True

    return contacts

  def ResetPose(self, add_constraint):
    del add_constraint

    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=joint_id,
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)

    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      if "shoulder" in name:
        angle = INIT_MOTOR_ANGLES[i]
      elif "knee" in name:
        angle = INIT_MOTOR_ANGLES[i]
      else:
        raise ValueError("The name %s is not recognized as a motor joint." %
                         name)

      self._pybullet_client.resetJointState(self.quadruped,
                                            self._joint_name_to_id[name],
                                            angle,
                                            targetVelocity=0)

  def GetURDFFile(self):
    return self._urdf_filename

  def _BuildUrdfIds(self):
    """Build link IDs from the Bittle URDF."""
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._knee_link_ids = []
    self._toe_link_ids = []
    self._foot_link_ids = []

    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      joint_type = joint_info[2]

      if _SHOULDER_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      elif _KNEE_NAME_PATTERN.match(joint_name):
        self._knee_link_ids.append(joint_id)
      elif _TOE_NAME_PATTERN.match(joint_name):
        self._toe_link_ids.append(joint_id)
      elif joint_type == self._pybullet_client.JOINT_FIXED:
        continue
      else:
        # Ignore any non-actuated auxiliary joints.
        continue

    self._leg_link_ids.extend(self._knee_link_ids)
    self._leg_link_ids.extend(self._toe_link_ids)
    self._foot_link_ids.extend(self._toe_link_ids)
    if self._allow_knee_contact:
      self._foot_link_ids.extend(self._knee_link_ids)

    self._chassis_link_ids.sort()
    self._motor_link_ids.sort()
    self._knee_link_ids.sort()
    self._toe_link_ids.sort()
    self._leg_link_ids.sort()
    self._foot_link_ids.sort()

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    # Keep this aligned with the retarget config's initial yaw.
    return pyb.getQuaternionFromEuler([0.0, 0.0, -math.pi / 2.0])

  def GetDefaultInitPosition(self):
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    return (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS

  def ApplyAction(self, motor_commands, motor_control_mode):
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)
    super(Bittle, self).ApplyAction(motor_commands, motor_control_mode)

  def _ClipMotorCommands(self, motor_commands):
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles = self.GetMotorAngles()
    return np.clip(motor_commands,
                   current_motor_angles - max_angle_change,
                   current_motor_angles + max_angle_change)
