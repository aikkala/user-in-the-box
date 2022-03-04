import numpy as np
import mujoco_py
from gym import spaces
from collections import deque
import xml.etree.ElementTree as ET
import os

from UIB.envs.mobl_arms.models.FixedEye import FixedEye
from UIB.envs.mobl_arms.choosing.reward_functions import NegativeExpDistanceWithHitBonus
from UIB.utils.functions import project_path


class ChoosingEnv(FixedEye):

  def __init__(self, **kwargs):

    # Modify the xml file first
    tree = ET.parse(self.xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    worldbody.remove(worldbody.find("body[@name='target']"))
    worldbody.remove(worldbody.find("body[@name='target-estimate']"))
    worldbody.remove(worldbody.find("body[@name='target-plane']"))

    # Add a screen
    screen = ET.Element('body', name='screen-body', pos="0.8 0 1.2", euler="0 1.57 0")
    screen.append(ET.Element('geom', name='screen', type="box", size="0.2 0.2 0.005"))
    worldbody.append(screen)

    # Try to place buttons in front of shoulder
    shoulder_pos = np.array([-0.017555, -0.17, 0.993])

    # Add buttons of different colors
    self.buttons = []
    self.add_button(root, worldbody, idx=0, position=shoulder_pos + np.array([0.41, -0.07, -0.15]),
                    euler="0 -0.79 0", color=np.array([0.8, 0.1, 0.1, 1.0]))
    self.add_button(root, worldbody, idx=1, position=shoulder_pos + np.array([0.41, 0.07, -0.15]),
                    euler="0 -0.79 0", color=np.array([0.1, 0.8, 0.1, 1.0]))
    self.add_button(root, worldbody, idx=2, position=shoulder_pos + np.array([0.5, -0.07, -0.05]),
                    euler="0 -0.79 0", color=np.array([0.1, 0.1, 0.8, 1.0]))
    self.add_button(root, worldbody, idx=3, position=shoulder_pos + np.array([0.5, 0.07, -0.05]),
                    euler="0 -0.79 0", color=np.array([0.8, 0.8, 0.1, 1.0]))
    self.current_button = self.buttons[0]

    # Save the modified XML file and replace old one
    xml_file = os.path.join(project_path(), 'envs/mobl_arms/models/variants/choosing_env.xml')
    with open(xml_file, 'w') as file:
      file.write(ET.tostring(tree.getroot(), encoding='unicode'))
    self.xml_file = xml_file

    # Now initialise variants with the modified XML file
    super().__init__(**kwargs)

    # Use early termination if target is not hit in time
    self.steps_since_last_hit = 0
    self.max_steps_without_hit = self.action_sample_freq*4
    self.steps = 0

    # Define a maximum number of button presses
    self.trial_idx = 0
    self.max_trials = kwargs.get('max_trials', 10)
    self.targets_hit = 0

    # Define a default reward function
    if self.reward_function is None:
      self.reward_function = NegativeExpDistanceWithHitBonus()

  def add_button(self, root, worldbody, idx, position, euler, color):

    # Add button
    button = ET.Element('body', name=f'button{idx}-body', pos=np.array2string(position)[1:-1], euler=euler)
    button.append(ET.Element('geom', name=f'button{idx}', type="box", size="0.05 0.05 0.01",
                             rgba=np.array2string(color)[1:-1]))
    button.append(ET.Element('site', name=f'button{idx}-site', pos="0 0 0.01", type="box", size="0.045 0.045 0.01",
                             rgba="0.5 0.5 0.5 0.0"))
    worldbody.append(button)

    # Add to collection of buttons
    self.buttons.append({"name": f'button{idx}', "position": position, "color": color, "idx": idx})

    # Add contact between fingertip and button
    if root.find('contact') is None:
      root.append(ET.Element('contact'))
    root.find('contact').append(ET.Element('pair', name=f"fingertip-button{idx}", geom1=self.fingertip,
                                           geom2=f"button{idx}"))

    # Add a force sensor to button
    if root.find('sensor') is None:
      root.append(ET.Element('sensor'))
    root.find('sensor').append(ET.Element('touch', name=f"button{idx}-sensor", site=f"button{idx}-site"))

  def step(self, action):

    # Set muscle control
    self.set_ctrl(action)

    finished = False
    info = {"termination": False, "new_button_generated": False}
    try:
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      finished = True
      info["termination"] = "MujocoException"

    # Check if the correct button has been pressed with suitable force
    force = self.sim.data.sensordata[self.current_button["idx"]]

    if 100 > force > 50:
      info["target_hit"] = True
      self.trial_idx += 1
      self.targets_hit += 1
      self.steps_since_last_hit = 0
      self.choose_button()
      info["new_button_generated"] = True

    else:

      info["target_hit"] = False

      # Check if time limit has been reached
      self.steps_since_last_hit += 1
      if self.steps_since_last_hit >= self.max_steps_without_hit:
        # Choose a new button
        self.steps_since_last_hit = 0
        self.trial_idx += 1
        self.choose_button()
        info["new_button_generated"] = True

    # Check if max number trials reached
    if self.trial_idx >= self.max_trials:
      finished = True
      info["termination"] = "max_trials_reached"

    # Increase counter
    self.steps += 1

    # Get finger position and target position
    finger_position = self.sim.data.get_geom_xpos(self.fingertip)
    target_position = self.current_button["position"]

    # Distance to target
    dist = np.linalg.norm(target_position - finger_position)

    # Calculate reward
    reward = self.reward_function.get(self, dist, info)

    # Add an effort cost to reward
    reward -= self.effort_term.get(self)

    return self.get_observation(), reward, finished, info

  def choose_button(self):

    # Choose a new button randomly, but don't choose the same button as previous one
    while True:
      new_button = np.random.choice(self.buttons)
      if new_button["idx"] != self.current_button["idx"]:
        self.current_button = new_button
        break

    # Set color of screen
    self.model.geom_rgba[self.model._geom_name2id['screen']] = self.current_button["color"]

    self.sim.forward()

  def reset(self):

    # Reset counters
    self.steps_since_last_hit = 0
    self.steps = 0
    self.trial_idx = 0
    self.targets_hit = 0

    # Choose a new button
    self.choose_button()

    return super().reset()


class ProprioceptionAndVisual(ChoosingEnv):

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

    # Time features (trials remaining)
    targets_hit = -1.0 + 2*(self.trial_idx/self.max_trials)

    # Append to proprioception since those will be handled with a fully-connected layer
    observation["proprioception"] = np.concatenate([observation["proprioception"], np.array([targets_hit])])

    return observation
