import numpy as np
import mujoco_py
from abc import ABC
from gym import spaces
from collections import deque

from UIB.envs.mobl_arms.models.FixedEye.FixedEye import FixedEye

class TrackingEnv(FixedEye):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Define episode length
    self.max_episode_steps = kwargs.get('max_episode_steps', self.action_sample_freq*4)
    self.steps = 0

    # Define a visual buffer
    self.visual_buffer = deque(maxlen=3)

    # Target radius
    self.target_radius = kwargs.get('target_radius', 0.05)

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

    # Generate trajectory
    self.sin_y, self.sin_z = self.generate_trajectory()


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
    reward += self.effort_term.get(self)

    # Update target location
    self.update_target_location()

    return self.get_observation(), reward, finished, info

  def reset(self):

    # Reset counters
    self.steps = 0

    # Reset visual buffer
    self.visual_buffer.clear()

    # Generate a new trajectory
    self.sin_y, self.sin_z = self.generate_trajectory()

    # Update target location
    self.update_target_location()

    return super().reset()

  def generate_trajectory(self):
    sin_y = self.generate_sine_wave(self.target_limits_y, self.max_episode_steps, num_components=5)
    sin_z = self.generate_sine_wave(self.target_limits_z, self.max_episode_steps, num_components=5)
    return sin_y, sin_z

  def generate_sine_wave(self, limits, length, num_components=5, min_amplitude=5, max_amplitude=50,
                         min_phase_diff=-10, max_phase_diff=10):

    # Generate a sine wave with multiple components
    t = np.linspace(limits[0], limits[1], length+1)
    sine = np.zeros((t.size,))
    for _ in range(num_components):
      sine += np.sin(self.rng.uniform(min_amplitude, max_amplitude) * t +
                     self.rng.uniform(min_phase_diff, max_phase_diff))

    # Normalise to [0, 1]
    sine = 0.5*((sine/num_components)+1)

    # Shift to fit limits
    sine = limits[0] + (limits[1] - limits[0])*sine

    return sine

  def update_target_location(self):
    self.target_position[0] = 0
    self.target_position[1] = self.sin_y[self.steps]
    self.target_position[2] = self.sin_z[self.steps]
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position
    self.sim.forward()


class ProprioceptionAndVisual(TrackingEnv):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

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

    depth = observation["visual"][:, :, 3, None]

    if len(self.visual_buffer) > 0:
      self.visual_buffer.pop()

    while len(self.visual_buffer) < self.visual_buffer.maxlen:
      self.visual_buffer.appendleft(depth)

    # Use only depth image
    observation["visual"] = np.concatenate([img for img in self.visual_buffer], axis=2)

    return observation