import gym
from gym import spaces
import numpy as np

from UIB.envs.mobl_arms.base import BaseModel


class OnePolicyNoCamera(BaseModel):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Reset
    observation = self.reset()

    # Set observation space
    low = np.ones_like(observation)*-float('inf')
    high = np.ones_like(observation)*float('inf')
    self.observation_space = spaces.Box(low=np.float32(low), high=np.float32(high))


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

    time_left = -1.0 + 2*np.min([1.0, self.steps_since_last_hit/self.max_steps_without_hit,
                                self.steps/self.max_episode_length])
    dwell_time = -1.0 + 2*np.min([1.0, self.steps_inside_target/self.dwelling_threshold])

    return np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position, self.target_position.copy(),
                           np.array([self.target_radius]), act, np.array([dwell_time]), np.array([time_left])])
