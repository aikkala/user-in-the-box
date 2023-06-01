import numpy as np
import mujoco

from .reward_functions import NegativeExpDistanceWithHitBonus
from ..base import BaseTask

class Pointing(BaseTask):

  def __init__(self, model, data, end_effector, shoulder, **kwargs):
    super().__init__(model, data, **kwargs)

    # This task requires an end-effector to be defined
    if not isinstance(end_effector, list) and len(end_effector) != 2:
      raise RuntimeError("'end_effector' must be a list with two elements: first defining what type of mujoco element "
                         "it is, and second defining the name")
    self._end_effector = end_effector

    # Also a shoulder that is used to define the location of target plane
    if not isinstance(shoulder, list) and len(shoulder) != 2:
      raise RuntimeError("'shoulder' must be a list with two elements: first defining what type of mujoco element "
                         "it is, and second defining the name")
    self._shoulder = shoulder

    # Use early termination if target is not hit in time
    self._steps_since_last_hit = 0
    self._max_steps_without_hit = self._action_sample_freq*4

    # Used for logging states
    self._info = {"target_hit": False, "inside_target": False, "target_spawned": False, "finished": False,
                 "termination": False}

    # Define a maximum number of trials (if needed for e.g. evaluation / visualisation)
    self._trial_idx = 0
    self._max_trials = kwargs.get('max_trials', 10)
    self._targets_hit = 0

    # Dwelling based selection -- fingertip needs to be inside target for some time
    self._steps_inside_target = 0
    self._dwell_threshold = int(0.5*self._action_sample_freq)

    # Radius limits for target
    self._target_radius_limit = kwargs.get('target_radius_limit', np.array([0.05, 0.15]))
    self._target_radius = self._target_radius_limit[0]

    # Minimum distance to new spawned targets is twice the max target radius limit
    self._new_target_distance_threshold = 2*self._target_radius_limit[1]

    # Define a default reward function
    #if self.reward_function is None:
    self._reward_function = NegativeExpDistanceWithHitBonus(k=10)

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Define plane where targets will move: 0.55m in front of and 0.1m to the right of shoulder, or the "humphant" body.
    # Note that this body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    self._target_origin = getattr(data, self._shoulder[0])(self._shoulder[1]).xpos + np.array([0.55, -0.1, 0])
    self._target_position = self._target_origin.copy()
    self._target_limits_y = np.array([-0.3, 0.3])
    self._target_limits_z = np.array([-0.3, 0.3])

    # Update plane location
    model.geom("target-plane").size = np.array([0.005,
                                                (self._target_limits_y[1] - self._target_limits_y[0])/2,
                                                (self._target_limits_z[1] - self._target_limits_z[0])/2])
    model.body("target-plane").pos = self._target_origin

    # Set camera angle TODO need to rethink how cameras are implemented
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([1.1, -0.9, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])
    #model.cam_pos[model.camera_name2id('for_testing')] = np.array([-0.8, -0.6, 1.5])
    #model.cam_quat[model.camera_name2id('for_testing')] = np.array(
    #  [0.718027, 0.4371043, -0.31987, -0.4371043])

  def _update(self, model, data):

    # Set some defaults
    finished = False
    self._info["target_spawned"] = False

    # Get end-effector position
    ee_position = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos

    # Distance to target
    dist = np.linalg.norm(self._target_position - (ee_position - self._target_origin))

    # Check if fingertip is inside target
    if dist < self._target_radius:
      self._steps_inside_target += 1
      self._info["inside_target"] = True
    else:
      self._steps_inside_target = 0
      self._info["inside_target"] = False

    if self._info["inside_target"] and self._steps_inside_target >= self._dwell_threshold:

      # Update counters
      self._info["target_hit"] = True
      self._trial_idx += 1
      self._targets_hit += 1
      self._steps_since_last_hit = 0
      self._steps_inside_target = 0
      self._spawn_target(model, data)
      self._info["target_spawned"] = True

    else:

      self._info["target_hit"] = False

      # Check if time limit has been reached
      self._steps_since_last_hit += 1
      if self._steps_since_last_hit >= self._max_steps_without_hit:
        # Spawn a new target
        self._steps_since_last_hit = 0
        self._trial_idx += 1
        self._spawn_target(model, data)
        self._info["target_spawned"] = True

    # Check if max number trials reached
    if self._trial_idx >= self._max_trials:
      finished = True
      self._info["termination"] = "max_trials_reached"

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # fingertip is inside target
    reward = self._reward_function.get(self, dist-self._target_radius, self._info.copy())

    return reward, finished, self._info.copy()

  def _get_state(self, model, data):
    state = dict()
    state["target_position"] = self._target_origin.copy()+self._target_position.copy()
    state["target_radius"] = self._target_radius
    state["trial_idx"] = self._trial_idx
    state["targets_hit"] = self._targets_hit
    state.update(self._info)
    return state

  def _reset(self, model, data):

    # Reset counters
    self._steps_since_last_hit = 0
    self._steps_inside_target = 0
    self._trial_idx = 0
    self._targets_hit = 0

    self._info = {"target_hit": False, "inside_target": False, "target_spawned": False, "finished": False,
                 "termination": False}

    # Spawn a new location
    self._spawn_target(model, data)

  def _spawn_target(self, model, data):

    # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
    for _ in range(10):
      target_y = self._rng.uniform(*self._target_limits_y)
      target_z = self._rng.uniform(*self._target_limits_z)
      new_position = np.array([0, target_y, target_z])
      distance = np.linalg.norm(self._target_position - new_position)
      if distance > self._new_target_distance_threshold:
        break
    self._target_position = new_position

    # Set location
    model.body("target").pos[:] = self._target_origin + self._target_position

    # Sample target radius
    self._target_radius = self._rng.uniform(*self._target_radius_limit)

    # Set target radius
    model.geom("target").size[0] = self._target_radius

    mujoco.mj_forward(model, data)

  def get_stateful_information(self, model, data):
    # Time features (time left to reach target, time spent inside target)
    targets_hit = -1.0 + 2*(self._trial_idx/self._max_trials)
    dwell_time = -1.0 + 2 * np.min([1.0, self._steps_inside_target / self._dwell_threshold])
    return np.array([dwell_time, targets_hit])
