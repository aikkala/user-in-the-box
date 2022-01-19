import gym
from gym import spaces
import mujoco_py
import numpy as np
import os
import pathlib
import torch
from stable_baselines3 import PPO


from UIB.archs.regressor import SimpleSequentialCNN, SimpleCNN

from UIB.envs.mobl_arms.base import BaseModel

class TwoLevelSL(BaseModel, gym.Env):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.height = 80
    self.width = 120

    # Load network for estimating target position
    target_extractor_file = os.path.join(self.project_path, '../../archs/regressor')
    #self.target_extractor = SimpleSequentialCNN(height=80, width=120,
    #                                            proprioception_size=self.grab_proprioception().size,
    #                                            seq_max_len=self.max_steps_without_hit+1)
    self.target_extractor = SimpleCNN(height=self.height, width=self.width,
                                      proprioception_size=self.grab_proprioception().size)
    self.target_extractor.load_state_dict(torch.load(target_extractor_file))
    self.target_extractor.eval()

    # Load policy
    policy_file = os.path.join(os.path.join(self.project_path,
                                            '../../../output/current-best-v0/checkpoint/model_100000000_steps.zip'))
    self.policy = PPO.load(policy_file)

  def get_action(self, observation):

    # Estimate target position and radius
    estimate = self.target_extractor.estimate(self.grab_image(height=self.height, width=self.width),
                                              self.grab_proprioception()).squeeze()

    # Update position of estimate for rendering
    self.model.body_pos[self.model._body_name2id["target-estimate"]] = estimate[:3]
    self.model.geom_size[self.model._geom_name2id["target-sphere-estimate"]][0] = estimate[3]

    # Needed because the way we've trained v0
    estimate[:3] -= self.target_origin

    # Replace target in the observation
    target_idxs = np.arange(18, 22, dtype=np.int)
    observation[target_idxs] = estimate

    # Use policy to get action
    action, _ = self.policy.predict(observation, deterministic=False)

    return action

  def get_observation(self):
    # Ignore eye qpos and qvel for now
    jnt_range = self.sim.model.jnt_range[self.independent_joints]

    qpos = self.sim.data.qpos[self.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5) * 2
    qvel = self.sim.data.qvel[self.independent_joints].copy()
    qacc = self.sim.data.qacc[self.independent_joints].copy()

    act = (self.sim.data.act.copy() - 0.5) * 2
    finger_position = self.sim.data.get_geom_xpos(self.fingertip) - self.target_origin

    return np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position - self.target_origin, self.target_position,
                           np.array([self.target_radius]), act])

  def step(self, action):

    observation, reward, done, info = BaseModel.step(self, action)

    # Zero out lstm hidden states if target has been hit
    if info["target_hit"]:
      self.target_extractor.initialise()

    return observation, reward, done, info