import numpy as np
import mujoco_py
from gym import spaces

from UIB.envs.mobl_arms.models.FixedEye import FixedEye
from UIB.envs.mobl_arms.pointing.reward_functions import ExpDistanceWithHitBonus

class PointingEnv(FixedEye):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Use early termination if target is not hit in time
    self.steps_since_last_hit = 0
    self.max_steps_without_hit = self.action_sample_freq*4
    self.steps = 0

    # Define a maximum number of trials (if needed for e.g. evaluation / visualisation)
    self.trial_idx = 0
    self.max_trials = kwargs.get('max_trials', 10)
    self.targets_hit = 0

    # Dwelling based selection -- fingertip needs to be inside target for some time
    self.steps_inside_target = 0
    self.dwell_threshold = int(0.3*self.action_sample_freq)

    # Radius limits for target
    self.target_radius_limit = kwargs.get('target_radius_limit', np.array([0.05, 0.15]))

    # Minimum distance to new spawned targets is twice the max target radius limit
    self.new_target_distance_threshold = 2*self.target_radius_limit[1]

    # Define a default reward function
    if self.reward_function is None:
      self.reward_function = ExpDistanceWithHitBonus()

    # Do a forward step so stuff like geom and body positions are calculated
    self.sim.forward()

    # Define plane where targets will be spawned: 0.5m in front of shoulder, or the "humphant" body. Note that this
    # body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    #self.target_origin = np.array([0.5, 0.0, 0.8])
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

    # Set muscle control
    self.set_ctrl(action)

    finished = False
    info = {"termination": False, "target_spawned": False}
    try:
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      finished = True
      info["termination"] = "MujocoException"

    # Get finger position
    finger_position = self.sim.data.get_geom_xpos(self.fingertip)

    # Distance to target
    dist = np.linalg.norm(self.target_position - (finger_position - self.target_origin))

    # Check if fingertip is inside target
    if dist < self.target_radius:
      self.steps_inside_target += 1
      info["inside_target"] = True
    else:
      self.steps_inside_target = 0
      info["inside_target"] = False

    if info["inside_target"] and self.steps_inside_target >= self.dwell_threshold:

      # Update counters
      info["target_hit"] = True
      self.trial_idx += 1
      self.targets_hit += 1
      self.steps_since_last_hit = 0
      self.steps_inside_target = 0
      self.spawn_target()
      info["target_spawned"] = True

    else:

      info["target_hit"] = False

      # Check if time limit has been reached
      self.steps_since_last_hit += 1
      if self.steps_since_last_hit >= self.max_steps_without_hit:
        #finished = True
        #info["termination"] = "time_limit_reached"
        # Spawn a new target
        self.steps_since_last_hit = 0
        self.trial_idx += 1
        self.spawn_target()
        info["target_spawned"] = True

    # Check if max number trials reached
    if self.trial_idx >= self.max_trials:
      finished = True
      info["termination"] = "max_trials_reached"

    # Increase counter
    self.steps += 1

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # fingertip is inside target
    reward = self.reward_function.get(self, dist-self.target_radius, info)

    # Add an effort cost to reward
    reward -= self.effort_term.get(self)

    return self.get_observation(), reward, finished, info

  def get_state(self):
    state = super().get_state()
    state["target_position"] = self.target_origin.copy()+self.target_position.copy()
    state["target_radius"] = self.target_radius
    state["target_hit"] = False
    state["inside_target"] = False
    state["target_spawned"] = False
    state["trial_idx"] = self.trial_idx
    state["targets_hit"] = self.targets_hit
    return state

  def reset(self):

    # Reset counters
    self.steps_since_last_hit = 0
    self.steps = 0
    self.steps_inside_target = 0
    self.trial_idx = 0
    self.targets_hit = 0

    # Spawn a new location
    self.spawn_target()

    return super().reset()

  def spawn_target(self):

    # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
    for _ in range(10):
      target_y = self.rng.uniform(*self.target_limits_y)
      target_z = self.rng.uniform(*self.target_limits_z)
      new_position = np.array([0, target_y, target_z])
      distance = np.linalg.norm(self.target_position - new_position)
      if distance > self.new_target_distance_threshold:
        break
    self.target_position = new_position

    # Set location
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position

    # Sample target radius
    self.target_radius = self.rng.uniform(*self.target_radius_limit)

    # Set target radius
    self.model.geom_size[self.model._geom_name2id["target-sphere"]][0] = self.target_radius

    self.sim.forward()

  def set_target_position(self, position):
    self.target_position = position.copy()
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position
    self.sim.forward()

  def set_target_radius(self, radius):
    self.target_radius = radius
    self.model.geom_size[self.model._geom_name2id["target-sphere"]][0] = self.target_radius
    self.sim.forward()


class ProprioceptionAndVisual(PointingEnv):
  metadata = {'render.modes': ['human']}

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

    # Use only depth image
    observation["visual"] = observation["visual"][:, :, 3, None]

    # Time features (time left to reach target, time spent inside target)
    #time_left = -1.0 + 2*np.min([1.0, self.steps_since_last_hit/self.max_steps_without_hit])
    targets_hit = -1.0 + 2*(self.trial_idx/self.max_trials)
    dwell_time = -1.0 + 2*np.min([1.0, self.steps_inside_target/self.dwell_threshold])

    # Append to proprioception since those will be handled with a fully-connected layer
    observation["proprioception"] = np.concatenate([observation["proprioception"], np.array([dwell_time, targets_hit])])

    return observation


class Proprioception(PointingEnv):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):

    # Make sure images aren't rendered while get_observation is called, it would slow down computations a lot
    super().__init__(render_observations=False, **kwargs)

    # Reset
    observation = self.reset()

    # Set observation space
    low = np.ones_like(observation) * -float('inf')
    high = np.ones_like(observation) * float('inf')
    self.observation_space = spaces.Box(low=np.float32(low), high=np.float32(high))

  def get_observation(self):

    # Get proprioception + visual observation
    observation = super().get_observation()

    # Time features (time left to reach target, time spent inside target)
    #time_left = -1.0 + 2 * np.min([1.0, self.steps_since_last_hit / self.max_steps_without_hit])
    targets_hit = -1.0 + 2*(self.trial_idx/self.max_trials)
    dwell_time = -1.0 + 2 * np.min([1.0, self.steps_inside_target / self.dwell_threshold])

    # Append target position + radius and time features to proprioception
    features = np.concatenate([observation["proprioception"], self.target_position.copy(),
                               np.array([self.target_radius, dwell_time, targets_hit])])

    return features
