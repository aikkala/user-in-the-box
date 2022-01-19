import gym
from gym import spaces
import mujoco_py
import numpy as np
import os

from UIB.envs.mobl_arms.base import BaseModel


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class OnePolicyNoCamera(BaseModel, gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def get_observation(self):
    # Ignore eye qpos and qvel for now
    jnt_range = self.sim.model.jnt_range[self.independent_joints]

    qpos = self.sim.data.qpos[self.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5)*2
    qvel = self.sim.data.qvel[self.independent_joints].copy()
    qacc = self.sim.data.qacc[self.independent_joints].copy()

    act = (self.sim.data.act.copy() - 0.5)*2
    finger_position = self.sim.data.get_geom_xpos(self.fingertip).copy() - self.target_origin.copy()

    return np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position-self.target_origin, self.target_position.copy(),
                           np.array([self.target_radius]), act])
