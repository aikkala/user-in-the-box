import numpy as np
import mujoco_py
from gym import spaces
import xml.etree.ElementTree as ET
import os

from UIB.envs.mobl_arms.models.FixedEye import FixedEye
from UIB.envs.mobl_arms.pointing.reward_functions import ExpDistanceWithHitBonus
from UIB.utils.functions import project_path


def add_target(worldbody, target_radius, idx, position):
  # Add target
  target = ET.Element('body', name=f'target{idx}-body', pos=np.array2string(position)[1:-1])
  target.append(ET.Element('geom', name=f'target{idx}', type="sphere", size=str(target_radius),
                           rgba="1.0 1.0 1.0 0.08"))
  worldbody.append(target)

  # Add to collection of targets
  return {"name": f'target{idx}', "position": position, "idx": idx}


class ISOPointingEnv(FixedEye):

  def __init__(self, user, **kwargs):

    # Modify the xml file first
    tree = ET.parse(os.path.join(project_path(), f"envs/mobl_arms/models/variants/mobl_arms_muscles_scaled{user}.xml"))
    root = tree.getroot()
    worldbody = root.find('worldbody')

    # Try to place buttons in front of shoulder
    shoulder_pos = np.array([-0.01878, -0.1819,  0.9925])  #TODO currently needs to be checked run-time for exact coords
    self.target_origin = shoulder_pos + np.array([0.55, -0.1, 0])

    # Target radius
    self.target_radius = 0.025

    # Add targets according to ISO 9241-9
    self.targets = []
    positions = np.array([[0, 0.10, -0.15],
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

    # Add targets to above positions
    for idx, pos in enumerate(positions):
      self.targets.append(add_target(worldbody, target_radius=self.target_radius, idx=idx,
                                     position=self.target_origin + pos))
    self.target_idx = 0

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
    self.max_trials = kwargs.get('max_trials', 13)
    self.targets_hit = 0

    # Dwelling based selection -- fingertip needs to be inside target for some time
    #self.steps_inside_target = 0
    #self.dwell_threshold = int(0.3*self.action_sample_freq)

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
    dist = np.linalg.norm(self.targets[self.target_idx]["position"] - finger_position)

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
      self.next_target()
      info["target_spawned"] = True

    else:

      info["target_hit"] = False

      # Check if time limit has been reached
      self.steps_since_last_hit += 1
      if self.steps_since_last_hit >= self.max_steps_without_hit:
        # Go to next target
        self.steps_since_last_hit = 0
        self.trial_idx += 1
        self.next_target()
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
    state["target_position"] = self.targets[self.target_idx]["position"]
    state["target_radius"] = self.target_radius
    state["target_idx"] = self.targets[self.target_idx]["idx"]
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
    self.next_target()

    return super().reset()

  def next_target(self):

    # Disable highlight of previous target
    self.model.geom_rgba[self.model._geom_name2id[self.targets[self.target_idx]["name"]]] = \
      np.array([1.0, 1.0, 1.0, 0.08])

    # Choose next target in the list
    self.target_idx = (self.target_idx+1) % len(self.targets)

    # Highlight new target
    self.model.geom_rgba[self.model._geom_name2id[self.targets[self.target_idx]["name"]]] = \
      np.array([1.0, 1.0, 0.0, 1.0])

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
    #observation["visual"] = observation["visual"][:, :, 3, None]

    # Time features (how many targets left, time spent inside target)
    #targets_hit = -1.0 + 2*(self.trial_idx/self.max_trials)

    # Append to proprioception since those will be handled with a fully-connected layer
    #observation["proprioception"] = np.concatenate([observation["proprioception"], np.array([dwell_time, targets_hit])])

    return observation
