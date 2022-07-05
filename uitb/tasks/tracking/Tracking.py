import numpy as np
import mujoco

from ..base import BaseTask
from .reward_functions import NegativeDistance


class Tracking(BaseTask):

  def __init__(self, model, data, end_effector, shoulder, freq_curriculum, **kwargs):
    super().__init__(model, data, **kwargs)

    # This task requires an end-effector to be defined TODO could be either body or geom, or why not a site
    self.end_effector = end_effector

    # Also a shoulder that is used to define the location of target plane
    self.shoulder = shoulder

    # Define episode length
    episode_length_seconds = kwargs.get('episode_length_seconds', 4)
    self.max_episode_steps = kwargs.get('max_episode_steps', self.action_sample_freq*episode_length_seconds)
    self.steps = 0

    # Define some limits for target movement speed
    self.min_frequency = 0.0
    self.max_frequency = 0.5
    self.freq_curriculum = freq_curriculum

    # Define a default reward function
    self.reward_function = NegativeDistance()

    # Target radius
    self.target_radius = kwargs.get('target_radius', 0.05)
    model.geom("target").size[0] = self.target_radius

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Define plane where targets will move: 0.55m in front of and 0.1m to the right of shoulder.
    # Note that the body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    self.target_origin = data.body(self.shoulder).xpos + np.array([0.55, -0.1, 0])
    self.target_position = self.target_origin.copy()
    self.target_limits_y = np.array([-0.3, 0.3])
    self.target_limits_z = np.array([-0.3, 0.3])

    # Update plane location
    model.geom("target-plane").size = np.array([0.005,
                                                (self.target_limits_y[1] - self.target_limits_y[0])/2,
                                                (self.target_limits_z[1] - self.target_limits_z[0])/2])
    model.body("target-plane").pos = self.target_origin

    # Generate trajectory
    self.sin_y, self.sin_z = self.generate_trajectory()

    model.cam("for_testing").pos = np.array([-0.8, -0.6, 1.5])
    model.cam("for_testing").quat = np.array([0.718027, 0.4371043, -0.31987, -0.4371043])

  def update(self, model, data):

    finished = False
    info = {"termination": False}

    # Get end-effector position
    ee_position = data.geom(self.end_effector).xpos.copy()

    # Distance to target origin
    dist = np.linalg.norm(self.target_position - (ee_position - self.target_origin))

    # Is fingertip inside target?
    if dist <= self.target_radius:
      info["inside_target"] = True
    else:
      info["inside_target"] = False

    # Check if time limit has been reached
    self.steps += 1
    if self.steps >= self.max_episode_steps:
      finished = True
      info["termination"] = "time_limit_reached"

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # fingertip is inside target
    reward = self.reward_function.get(self, dist-self.target_radius, info)

    # Update target location
    self.update_target_location(model, data)

    return reward, finished, info

  def reset(self, model, data):

    # Reset counters
    self.steps = 0

    # Generate a new trajectory
    self.sin_y, self.sin_z = self.generate_trajectory()

    # Update target location
    self.update_target_location(model, data)

  def generate_trajectory(self):
    sin_y = self.generate_sine_wave(self.target_limits_y, num_components=5)
    sin_z = self.generate_sine_wave(self.target_limits_z, num_components=5)
    return sin_y, sin_z

  def generate_sine_wave(self, limits, num_components=5, min_amplitude=1, max_amplitude=5):

    max_frequency = self.min_frequency + (self.max_frequency-self.min_frequency) * self.freq_curriculum.value()

    # Generate a sine wave with multiple components
    t = np.arange(self.max_episode_steps+1) * self.dt
    sine = np.zeros((t.size,))
    sum_amplitude = 0
    for _ in range(num_components):
      amplitude = self.rng.uniform(min_amplitude, max_amplitude)
      sine +=  amplitude *\
              np.sin(self.rng.uniform(self.min_frequency, max_frequency)*2*np.pi*t + self.rng.uniform(0, 2*np.pi))
      sum_amplitude += amplitude

    # Normalise to fit limits
    sine = (sine + sum_amplitude) / (2*sum_amplitude)
    sine = limits[0] + (limits[1] - limits[0])*sine

    return sine

  def update_target_location(self, model, data):
    self.target_position[0] = 0
    self.target_position[1] = self.sin_y[self.steps]
    self.target_position[2] = self.sin_z[self.steps]
    model.body("target").pos = self.target_origin + self.target_position
    mujoco.mj_forward(model, data)
