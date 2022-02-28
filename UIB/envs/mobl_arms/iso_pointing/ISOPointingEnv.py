import numpy as np
import mujoco_py
from gym import spaces
import xml.etree.ElementTree as ET
import os

from UIB.envs.mobl_arms.models.FixedEye import FixedEye
from UIB.envs.mobl_arms.pointing.reward_functions import ExpDistanceWithHitBonus
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

    # Modify the xml file first
    tree = ET.parse(os.path.join(project_path(), f"envs/mobl_arms/models/variants/mobl_arms_muscles_scaled{user}.xml"))
    root = tree.getroot()
    worldbody = root.find('worldbody')

    # Add target and target plane -- exact coordinates and size doesn't matter, they are set later
    add_target(worldbody)
    add_target_plane(worldbody)

    # Save the modified XML file and replace old one
    xml_file = os.path.join(project_path(), f'envs/mobl_arms/models/variants/iso_pointing_env_{user}.xml')
    with open(xml_file, 'w') as file:
      file.write(ET.tostring(tree.getroot(), encoding='unicode'))
    self.xml_file = xml_file

    # Now initialise variants with the modified XML file
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
    #self.steps_inside_target = 0
    #self.dwell_threshold = int(0.3*self.action_sample_freq)

    # Set limits for target plane
    self.target_limits_y = np.array([-0.3, 0.3])
    self.target_limits_z = np.array([-0.3, 0.3])

    # Target radius
    self.target_radius_limit = kwargs.get('target_radius_limit', np.array([0.025, 0.15]))
    self.target_radius = self.target_radius_limit[0]

    # Minimum distance to new spawned targets is twice the max target radius limit
    self.new_target_distance_threshold = 2*self.target_radius_limit[1]

    # Add targets according to ISO 9241-9
    self.ISO_positions = np.array([[0, 0.10, -0.15],
                                   [0, 0.13589734964313366, 0.1456412726139078],
                                   [0, 0.030291524193434755, -0.1328184038479815],
                                   [0, 0.19946839873611927, 0.11227661222566519],
                                   [0, -0.023447579884048414, -0.08520971200967342],
                                   [0, 0.24025243640281224, 0.05319073305638029],
                                   [0, -0.048906331114708074, -0.018080502038298547],
                                   [0, 0.24890633111470814, -0.01808050203829822],
                                   [0, -0.04025243640281234, 0.05319073305637998],
                                   [0, 0.22344757988404879, -0.08520971200967294],
                                   [0, 0.0005316012638802159, 0.11227661222566471],
                                   [0, 0.169708475806566, -0.1328184038479811],
                                   [0, 0.06410265035686591, 0.1456412726139077]])

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
    finger_velocity = np.linalg.norm(self.sim.data.get_geom_xvelp(self.fingertip))

    # Distance to target
    dist = np.linalg.norm(self.target_position - (finger_position - self.target_origin))

    # Check if fingertip is inside target
    if dist < self.target_radius:
      #self.steps_inside_target += 1
      info["inside_target"] = True
    else:
      #self.steps_inside_target = 0
      info["inside_target"] = False

    if info["inside_target"] and finger_velocity < 0.5:

      # Update counters
      info["target_hit"] = True
      self.trial_idx += 1
      self.targets_hit += 1
      self.steps_since_last_hit = 0
      #self.steps_inside_target = 0
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
    #self.steps_inside_target = 0
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
      'visual': spaces.Box(low=-1, high=1, shape=observation['visual'].shape, dtype=np.float32)})

  def get_observation(self):

    # Get proprioception + visual observation
    observation = super().get_observation()

    # Use only depth image
    observation["visual"] = observation["visual"][:, :, 3, None]

    # Time features (how many targets left, time spent inside target)
    #targets_hit = -1.0 + 2*(self.trial_idx/self.max_trials)

    # Append to proprioception since those will be handled with a fully-connected layer
    #observation["proprioception"] = np.concatenate([observation["proprioception"], np.array([dwell_time, targets_hit])])

    return observation
