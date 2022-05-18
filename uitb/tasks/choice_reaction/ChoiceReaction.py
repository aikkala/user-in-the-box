import numpy as np
import mujoco
import xml.etree.ElementTree as ET

from .reward_functions import NegativeExpDistanceWithHitBonus
from ..base import BaseTask


class ChoiceReaction(BaseTask):

  def __init__(self, model, data, end_effector, shoulder, **kwargs):
    super().__init__(model, data, **kwargs)

    # This task requires an end-effector and shoulder to be defined
    assert end_effector[0] == "geom", "end-effector must be a geom because contacts are between geoms"
    self.end_effector = end_effector[1]
    self.shoulder = shoulder

    # Get buttons
    self.buttons = [f"button-{idx}" for idx in range(4)]
    self.current_button = self.buttons[0]

    # Use early termination if target is not hit in time
    self.steps_since_last_hit = 0
    self.max_steps_without_hit = self.action_sample_freq*4
    self.steps = 0

    # Define a maximum number of button presses
    self.trial_idx = 0
    self.max_trials = kwargs.get('max_trials', 10)
    self.targets_hit = 0

    # Define a default reward function
    self.reward_function = NegativeExpDistanceWithHitBonus()

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Get shoulder position
    shoulder_pos = getattr(data, self.shoulder[0])(self.shoulder[1]).xpos.copy()

    # Update button positions
    model.body("button-0").pos = shoulder_pos + [0.41, -0.07, -0.15]
    model.body("button-1").pos = shoulder_pos + [0.41, 0.07, -0.15]
    model.body("button-2").pos = shoulder_pos + [0.5, -0.07, -0.05]
    model.body("button-3").pos = shoulder_pos + [0.5, 0.07, -0.05]

    # Set camera angle
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([1.1, -0.9, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])
    #model.cam_pos[model.camera_name2id('for_testing')] = np.array([-0.8, -0.6, 1.5])
    #model.cam_quat[model.camera_name2id('for_testing')] = np.array(
    #  [0.718027, 0.4371043, -0.31987, -0.4371043])

  @classmethod
  def initialise(cls, task_kwargs):

    assert "end_effector" in task_kwargs, "End-effector must be defined for this environment"
    end_effector = task_kwargs["end_effector"][1]

    # Parse xml file
    tree = ET.parse(cls.get_xml_file())
    root = tree.getroot()

    # Add contact between end-effector and buttons
    if root.find('contact') is None:
      root.append(ET.Element('contact'))
    for idx in range(4):
      root.find('contact').append(ET.Element('pair', name=f"ee-button-{idx}", geom1=end_effector,
                                             geom2=f"button-{idx}"))

    return tree

  def update(self, model, data):

    finished = False
    info = {"termination": False, "new_button_generated": False}

    # Check if the correct button has been pressed with suitable force
    force = data.sensor(self.current_button).data

    if 50 > force > 25:
      info["target_hit"] = True
      self.trial_idx += 1
      self.targets_hit += 1
      self.steps_since_last_hit = 0
      self.choose_button(model, data)
      info["new_button_generated"] = True

    else:

      info["target_hit"] = False

      # Check if time limit has been reached
      self.steps_since_last_hit += 1
      if self.steps_since_last_hit >= self.max_steps_without_hit:
        # Choose a new button
        self.steps_since_last_hit = 0
        self.trial_idx += 1
        self.choose_button(model, data)
        info["new_button_generated"] = True

    # Check if max number trials reached
    if self.trial_idx >= self.max_trials:
      finished = True
      info["termination"] = "max_trials_reached"

    # Increase counter
    self.steps += 1

    # Get end-effector position and target position
    ee_position = data.geom(self.end_effector).xpos
    target_position = data.geom(self.current_button).xpos

    # Distance to target
    dist = np.linalg.norm(target_position - ee_position)

    # Calculate reward
    reward = self.reward_function.get(self, dist, info)

    return reward, finished, info

  def choose_button(self, model, data):

    # Choose a new button randomly, but don't choose the same button as previous one
    while True:
      new_button = self.rng.choice(self.buttons)
      if new_button != self.current_button:
        self.current_button = new_button
        break

    # Set color of screen
    model.geom("screen").rgba = model.geom(self.current_button).rgba

    mujoco.mj_forward(model, data)

  def reset(self, model, data):

    # Reset counters
    self.steps_since_last_hit = 0
    self.steps = 0
    self.trial_idx = 0
    self.targets_hit = 0

    # Choose a new button
    self.choose_button(model, data)

  def get_stateful_information(self, model, data):
    # Time features
    targets_hit = -1.0 + 2*(self.trial_idx/self.max_trials)
    return np.array([targets_hit])

  def get_stateful_information_space_params(self):
    return {"low": -1, "high": 1, "shape": (1,)}