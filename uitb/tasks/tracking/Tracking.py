import numpy as np
import mujoco

from ..base import BaseTask
from .reward_functions import NegativeDistance, NegativeExpDistanceWithHitBonus


class Tracking(BaseTask):

  def __init__(self, model, data, end_effector, shoulder, freq_curriculum, **kwargs):
    super().__init__(model, data, **kwargs)

    # This task requires an end-effector to be defined
    if not isinstance(end_effector, list) and len(end_effector) != 2:
      raise RuntimeError("'end_effector' must be a list with two elements: first defines the mujoco element type, and "
                         "second defines the name.")
    self._end_effector = end_effector

    # Also a shoulder that is used to define the location of target plane
    if not isinstance(shoulder, list) and len(shoulder) != 2:
      raise RuntimeError("'shoulder' must be a list with two elements: first defines the mujoco element type, and "
                         "second defines the name.")
    self._shoulder = shoulder

    # Define episode length
    episode_length_seconds = kwargs.get('episode_length_seconds', 10)
    self._max_episode_steps = kwargs.get('max_episode_steps', self._action_sample_freq*episode_length_seconds)

    # Define some limits for target movement speed
    self._min_frequency = 0.0
    self._max_frequency = 0.5
    self._freq_curriculum = freq_curriculum.value

    # For logging
    self._info = {"termination": False, "inside_target": False}

    # Define a default reward function
    self._reward_function = NegativeDistance()
    #self._reward_function = NegativeExpDistanceWithHitBonus(k=3, scale=1.0, bonus=0)

    # Target radius
    self._target_radius = kwargs.get('target_radius', 0.05)
    model.geom("target").size[0] = self._target_radius

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Define plane where targets will move: 0.55m in front of and 0.1m to the right of shoulder.
    # Note that the body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    self._target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + np.array([0.55, -0.1, 0])
    self._target_position = self._target_origin.copy()
    self._target_limits_y = np.array([-0.3, 0.3])
    self._target_limits_z = np.array([-0.3, 0.3])

    # Update plane location
    model.geom("target-plane").size = np.array([0.005,
                                                (self._target_limits_y[1] - self._target_limits_y[0])/2,
                                                (self._target_limits_z[1] - self._target_limits_z[0])/2])
    model.body("target-plane").pos = self._target_origin

    # Generate trajectory
    self._sin_y, self._sin_z = self._generate_trajectory()

    model.cam("for_testing").pos = np.array([-0.8, -0.6, 1.5])
    model.cam("for_testing").quat = np.array([0.718027, 0.4371043, -0.31987, -0.4371043])

  def _update(self, model, data):

    finished = False
    self._info = {"termination": False}

    # Get end-effector position
    ee_position = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos.copy()

    # Distance to target origin
    dist = np.linalg.norm(self._target_position - (ee_position - self._target_origin))

    # Is fingertip inside target?
    if dist <= self._target_radius:
      self._info["inside_target"] = True
    else:
      self._info["inside_target"] = False

    # Check if time limit has been reached
    if self._steps >= self._max_episode_steps:
      finished = True
      self._info["termination"] = "time_limit_reached"

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # fingertip is inside target
    reward = self._reward_function.get(self, dist-self._target_radius, self._info.copy())

    # Update target location
    self._update_target_location(model, data)

    return reward, finished, self._info.copy()

  def _get_state(self, model, data):
    state = dict()
    state.update(self._info)
    return state

  def _reset(self, model, data):

    self._info = {"termination": False, "inside_target": False}

    # Generate a new trajectory
    self._sin_y, self._sin_z = self._generate_trajectory()

    # Update target location
    self._update_target_location(model, data)

  def _generate_trajectory(self):
    sin_y = self._generate_sine_wave(self._target_limits_y, num_components=5)
    sin_z = self._generate_sine_wave(self._target_limits_z, num_components=5)
    return sin_y, sin_z

  def _generate_sine_wave(self, limits, num_components=5, min_amplitude=1, max_amplitude=5):

    max_frequency = self._min_frequency + (self._max_frequency-self._min_frequency) * self._freq_curriculum()

    # Generate a sine wave with multiple components
    t = np.arange(self._max_episode_steps+1) * self._dt
    sine = np.zeros((t.size,))
    sum_amplitude = 0
    for _ in range(num_components):
      amplitude = self._rng.uniform(min_amplitude, max_amplitude)
      sine +=  amplitude *\
              np.sin(self._rng.uniform(self._min_frequency, max_frequency)*2*np.pi*t + self._rng.uniform(0, 2*np.pi))
      sum_amplitude += amplitude

    # Normalise to fit limits
    sine = (sine + sum_amplitude) / (2*sum_amplitude)
    sine = limits[0] + (limits[1] - limits[0])*sine

    return sine

  def _update_target_location(self, model, data):
    self._target_position[0] = 0
    self._target_position[1] = self._sin_y[self._steps]
    self._target_position[2] = self._sin_z[self._steps]
    model.body("target").pos = self._target_origin + self._target_position
    mujoco.mj_forward(model, data)
