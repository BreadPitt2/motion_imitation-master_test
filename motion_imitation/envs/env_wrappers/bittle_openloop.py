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

"""Bittle-specific simple openloop trajectory generators."""

import numpy as np
from gym import spaces


class BittlePoseOffsetGenerator(object):
  """A trajectory generator that returns constant 8-motor angles for Bittle."""

  def __init__(self,
               init_pose=(0.0, -0.67, 0.0, -0.67, 0.0, 0.67, 0.0, 0.67),
               action_limit=0.5):
    self._pose = np.array(init_pose)
    action_high = np.array([action_limit] * len(self._pose))
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

  def reset(self):
    pass

  def get_action(self, current_time=None, input_action=None):
    del current_time
    return self._pose + input_action

  def get_observation(self, input_observation):
    return input_observation
