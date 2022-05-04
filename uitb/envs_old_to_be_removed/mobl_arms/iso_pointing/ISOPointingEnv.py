import numpy as np
import mujoco_py
from gym import spaces
import xml.etree.ElementTree as ET
import os

from UIB.envs_old_to_be_removed.mobl_arms.models.FixedEye import FixedEye
from UIB.envs_old_to_be_removed.mobl_arms.pointing.reward_functions import ExpDistanceWithHitBonus
from UIB.utils.functions import project_path


def add_target(worldbody):
  target = ET.Element('body', name='target', pos="0.5 0 0.8")
  target.append(ET.Element('geom', name='target', type="sphere", size="0.025", rgba="0.1 0.8 0.1 1.0"))
  worldbody.append(target)

def add_target_plane(worldbody):
  target_plane = ET.Element('body', name='target-plane', pos='0.5 0 0.8')
  target_plane.append(ET.Element('geom', name='target-plane', type='box', size='0.005 0.3 0.3', rgba='0.1 0.8 0.1 0'))
  worldbody.append(target_plane)


class ISOPointingEnv(FixedEye):

  def __init__(self, user, **kwargs):

    if user == "general":
      xml_file = os.path.join(project_path(), f"envs/mobl_arms/models/variants/mobl_arms_muscles_v2_modified.xml")
    else:
      xml_file = os.path.join(project_path(),
                                 f"envs/mobl_arms/models/variants/mobl_arms_muscles_v2_scaled{user}_modified.xml")

    # Modify the xml file first
    tree = ET.parse(xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    # Add target and target plane -- exact coordinates and size doesn't matter, they are set later
    add_target(worldbody)
    add_target_plane(worldbody)

    # Save the modified XML file and replace old one
    self.xml_file = os.path.join(project_path(), f'envs/mobl_arms/models/variants/iso_pointing_env_{user}.xml')
    with open(self.xml_file, 'w') as file:
      file.write(ET.tostring(tree.getroot(), encoding='unicode'))

    # Now initialise variants with the modified XML file
    super().__init__(**kwargs)

    # Use early termination if target is not hit in time
    self.steps_since_last_hit = 0
    self.max_steps_without_hit = self.action_sample_freq*4
    self.steps = 0

    # If set to evaluate mode, then we'll only use ISO positions
    self.evaluate = kwargs.get('evaluate', False)

    # Define a maximum number of trials (if needed for e.g. evaluation / visualisation)
    self.trial_idx = 0
    self.max_trials = 14 if self.evaluate else kwargs.get('max_trials', 14)
    self.targets_hit = 0

    # Dwelling based selection -- fingertip needs to be inside target for some time
    self.steps_inside_target = 0
    self.dwell_threshold = int(0.5*self.action_sample_freq)

    # Add some metrics to episode statistics
    self._episode_statistics = {**self._episode_statistics, **{"targets hit": 0}}

    # Set limits for target plane
    self.target_limits_y = np.array([-0.3, 0.3])
    self.target_limits_z = np.array([-0.3, 0.3])

    # Target radius
    self.target_radius_limit = kwargs.get('target_radius_limit', np.array([0.05, 0.15]))
    self.target_radius = self.target_radius_limit[0]

    # Minimum distance to new spawned targets is twice the max target radius limit
    self.new_target_distance_threshold = 2*self.target_radius_limit[1]

    # Add targets according to ISO 9241-9
    self.ISO_positions = np.array([[0., -0., -0.15],
                                   [0., -0.03589735, 0.14564127],
                                   [0., 0.06970848, -0.1328184],
                                   [0., -0.0994684, 0.11227661],
                                   [0., 0.12344758, -0.08520971],
                                   [0., -0.14025244, 0.05319073],
                                   [0., 0.14890633, -0.0180805],
                                   [0., -0.14890633, -0.0180805],
                                   [0., 0.14025244, 0.05319073],
                                   [0., -0.12344758, -0.08520971],
                                   [0., 0.0994684, 0.11227661],
                                   [0., -0.06970848, -0.1328184],
                                   [0., 0.03589735, 0.14564127]])
    self.target_idx = -1

    # Do a forward step so stuff like geom and body positions are calculated
    self.sim.forward()

    # Set target plane origin in front of shoulder
    shoulder_pos = self.sim.data.get_body_xpos("humphant")
    self.target_origin = shoulder_pos + np.array([0.55, -0.1, 0])
    self.target_position = self.target_origin.copy()

    # Update plane location
    self.target_plane_geom_idx = self.model._geom_name2id["target-plane"]
    self.target_plane_body_idx = self.model._body_name2id["target-plane"]
    self.model.geom_size[self.target_plane_geom_idx] = \
      np.array([0.005, (self.target_limits_y[1] - self.target_limits_y[0]) / 2,
                (self.target_limits_z[1] - self.target_limits_z[0]) / 2])
    self.model.body_pos[self.target_plane_body_idx] = self.target_origin

    # Define a default reward function
    if self.reward_function is None:
      self.reward_function = ExpDistanceWithHitBonus()

    #self.sim.model.cam_pos[self.sim.model._camera_name2id['for_testing']] = np.array([-0.8, -0.6, 1.5])
    #self.sim.model.cam_quat[self.sim.model._camera_name2id['for_testing']] = np.array(
    #  [0.718027, 0.4371043, -0.31987, -0.4371043])

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

    # Get finger velocity
    #finger_velocity = np.linalg.norm(self.sim.data.get_geom_xvelp(self.fingertip))

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
        # Go to next target
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

    # Update statistics
    self._episode_statistics["reward"] += reward

    return self.get_observation(), reward, finished, info

  def get_episode_statistics(self):
    self._episode_statistics["length (steps)"] = self.steps
    self._episode_statistics["length (seconds)"] = self.steps * self.dt
    self._episode_statistics["targets hit"] = self.targets_hit
    return self._episode_statistics.copy()

  def get_state(self):
    state = super().get_state()
    state["target_position"] = self.target_origin.copy()+self.target_position.copy()
    state["target_position_in_plane"] = self.target_position.copy()
    state["target_radius"] = self.target_radius
    state["target_hit"] = False
    state["inside_target"] = False
    state["target_spawned"] = False
    state["trial_idx"] = self.trial_idx
    state["targets_hit"] = self.targets_hit
    state["target_idx"] = self.target_idx

    return state

  def reset(self):

    # Reset counters
    self.steps_since_last_hit = 0
    self.steps = 0
    self.steps_inside_target = 0
    self.trial_idx = 0
    self.targets_hit = 0
    self.target_idx = -1

    # Spawn a new location
    self.spawn_target()

    return super().reset()

  def spawn_target(self):

    if self.evaluate:

      # Choose next position in the list of ISO positions
      self.target_idx = (self.target_idx + 1) % len(self.ISO_positions)
      new_position = self.ISO_positions[self.target_idx]

      # Use a fixed target radius
      self.target_radius = 0.05

    else:

      # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
      for _ in range(10):
        target_y = self.rng.uniform(*self.target_limits_y)
        target_z = self.rng.uniform(*self.target_limits_z)
        new_position = np.array([0, target_y, target_z])
        distance = np.linalg.norm(self.target_position - new_position)
        if distance > self.new_target_distance_threshold:
          break

      # Sample target radius
      self.target_radius = self.rng.uniform(*self.target_radius_limit)

    # Set new position
    self.target_position = new_position

    # Set location
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position

    # Set target radius
    self.model.geom_size[self.model._geom_name2id["target"]][0] = self.target_radius

    self.sim.forward()

  def set_target_position(self, position):
    self.target_position = position.copy()
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position
    self.sim.forward()

  def set_target_radius(self, radius):
    self.target_radius = radius
    self.model.geom_size[self.model._geom_name2id["target"]][0] = self.target_radius
    self.sim.forward()


class ProprioceptionAndVisual(ISOPointingEnv):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Reset
    observation = self.reset()

    # Set observation space
    self.observation_space = spaces.Dict({
      'proprioception': spaces.Box(low=-float('inf'), high=float('inf'), shape=observation['proprioception'].shape,
                                   dtype=np.float32),
      'vision': spaces.Box(low=-1, high=1, shape=observation['vision'].shape, dtype=np.float32)})

  def get_observation(self):

    # Get proprioception + vision observation
    observation = super().get_observation()

    # Use only depth image
    observation["vision"] = observation["vision"][:, :, 3, None]

    # Time features (how many targets left, time spent inside target)
    targets_hit = -1.0 + 2*(self.trial_idx/self.max_trials)
    dwell_time = -1.0 + 2 * np.min([1.0, self.steps_inside_target / self.dwell_threshold])

    # Append to proprioception since those will be handled with a fully-connected layer
    observation["proprioception"] = np.concatenate([observation["proprioception"], np.array([dwell_time, targets_hit])])

    return observation
