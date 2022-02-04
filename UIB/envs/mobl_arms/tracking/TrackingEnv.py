import numpy as np
import mujoco_py
from abc import ABC
from gym import spaces

from UIB.envs.mobl_arms.models.FixedEye.FixedEye import FixedEye

class TrackingEnv(ABC, FixedEye):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Define episode length
    self.max_episode_steps = kwargs.get('max_episode_steps', self.action_sample_freq*10)
    self.steps = 0

    # Target radius
    self.target_radius = kwargs.get('target_radius', 0.05)

    self.sin_y, self.sin_z = self.generate_trajectory()

    # Do a forward step so stuff like geom and body positions are calculated
    self.sim.forward()

    # Define plane where targets will move: 0.5m in front of shoulder, or the "humphant" body. Note that this
    # body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    self.target_origin = self.sim.data.get_body_xpos("humphant") + np.array([0.5, 0, 0])
    self.target_position = self.target_origin.copy()
    self.target_limits_y = np.array([-0.3, 0.3])
    self.target_limits_z = np.array([-0.3, 0.3])

    # Update plane location
    self.target_plane_geom_idx = self.model._geom_name2id["target-plane"]
    self.target_plane_body_idx = self.model._body_name2id["target-plane"]
    self.model.geom_size[self.target_plane_geom_idx] = np.array([0.005,
                                                            (self.target_limits_y[1] - self.target_limits_y[0])/2,
                                                            (self.target_limits_z[1] - self.target_limits_z[0])/2])
    self.model.body_pos[self.target_plane_body_idx] = self.target_origin


  def step(self, action):

    info = {}

    # Set muscle control
    self.set_ctrl(action)

    finished = False
    info["termination"] = False
    try:
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      finished = True
      info["termination"] = "MujocoException"

    # Get finger position
    finger_position = self.sim.data.get_geom_xpos(self.fingertip)

    # Distance to target
    dist = np.linalg.norm(self.target_position - (finger_position - self.target_origin))

    # Estimate reward
    reward = np.exp(-dist * 10)

    # Check if time limit has been reached
    self.steps += 1
    if self.steps >= self.max_episode_steps:
      finished = True
      info["termination"] = "time_limit_reached"

    # Add an effort cost to reward
    if self.cost_function == "neural_effort":
      reward -= 1e-4 * np.sum(self.sim.data.ctrl ** 2)
    elif self.cost_function == "composite":
      angle_acceleration = np.sum(self.sim.data.qacc[self.independent_joints] ** 2)
      energy = np.sum(
        self.sim.data.qacc[self.independent_joints] ** 2 * self.sim.data.qfrc_unc[self.independent_joints] ** 2)
      reward -= 1e-7 * (energy + 0.05 * angle_acceleration)

    return self.get_observation(), reward, finished, info

  def reset(self):

    # Reset counters
    self.steps = 0

    # Generate a new trajectory
    self.sin_y, self.sin_z = self.generate_trajectory()

    # Update target location
    self.update_target_location()

    super().reset()

  def generate_trajectory(self):
    sin_y = self.generate_sine_wave(self.max_episode_steps, num_components=5)
    sin_z = self.generate_sine_wave(self.max_episode_steps, num_components=5)
    return sin_y, sin_z

  def generate_sine_wave(self, length, num_components=5):
    t = np.linspace(-0.3, 0.3, length)
    sine = np.zeros((t.size,))
    for _ in range(num_components):
      sine += np.sin(np.random.uniform(5, 50) * t + np.random.uniform(-10, 10))
    sine = 0.3 * (sine / num_components)
    return sine

  def update_target_location(self):
    self.target_position[1] = self.sin_y[self.steps]
    self.target_position[2] = self.sin_z[self.steps]
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position
    self.sim.forward()


class ProprioceptionAndVisual(TrackingEnv):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Size of ocular image
    self.height = 80
    self.width = 120

    # Reset
    observation = self.reset()

    # Set observation space
    self.observation_space = spaces.Dict({
      'proprioception': spaces.Box(low=-float('inf'), high=float('inf'), shape=observation['proprioception'].shape,
                                   dtype=np.float32),
      'visual': spaces.Box(low=-1, high=1, shape=observation['visual'].shape, dtype=np.float32)})

  def get_observation(self):

    # Get proprioception + visual observation
    observation = super().get_observation()

    return observation