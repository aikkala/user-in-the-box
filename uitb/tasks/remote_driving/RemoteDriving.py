import numpy as np
import xml.etree.ElementTree as ET
import os
import mujoco

from uitb.tasks.base import BaseTask
from uitb.utils.functions import parent_path
from .reward_functions import NegativeExpDistance, RewardBonus


class RemoteDriving(BaseTask):

  xml_file = os.path.join(parent_path(__file__), "task.xml")

  def __init__(self, model, data, **kwargs):
    super().__init__(model, data)

    assert "end_effector" in kwargs, "End-effector must be defined for this env"
    self.end_effector = kwargs["end_effector"]

    # Define names
    self.gamepad_body = "gamepad"
    self.joystick_geom = "thumb-stick-1"
    self.joystick_joint = "thumb-stick-1:rot-x"
    self.car_body = "car"
    self.car_joint = "car"
    self.engine_back_joint = "axis-back:rot-x"
    self.engine_front_joint = "axis-front:rot-x"
    self.wheels = [f"wheel{i}" for i in range(1, 5)]
    self.target = "target"

    # Define joystick parameters
    self.throttle_factor = 50

    # Define episode length
    action_sample_freq = kwargs["action_sample_freq"]
    episode_length_seconds = kwargs.get('episode_length_seconds', 4)
    self.max_episode_steps = kwargs.get('max_episode_steps', action_sample_freq * episode_length_seconds)
    self.steps = 0

    # Episodes can be extended if joystick is touched
    episode_length_seconds_extratime = kwargs.get('episode_length_seconds_extratime', 6)
    self.max_episode_steps_extratime = kwargs.get('max_episode_steps_extratime',
                                                  action_sample_freq * episode_length_seconds_extratime)
    self.max_episode_steps_with_extratime = self.max_episode_steps  # extra time only applies if joystick is touched

    # Measure whether contact between fingertip and joystick exists
    self.fingertip_at_joystick = False

    # Measure distances between fingertip and joystick, and between car and target area
    self.dist_fingertip_to_joystick = 0
    self.dist_car_to_bound = 0

    # Terminal velocity -- car needs to stay inside target area (i.e., velocity in x-direction needs to fall below some threshold)
    self.car_velocity_threshold = kwargs.get('car_velocity_threshold', 0.1)

    # Halfsize limits for target
    self.target_halfsize_limit = kwargs.get('target_halfsize_limit', np.array([0.15, 0.5]))
    self.target_halfsize = self.target_halfsize_limit[0]

    # Minimum distance betwen car and target is twice the max target halfsize limit
    self.new_target_distance_threshold = 2 * np.max(self.target_halfsize_limit)

    # Define the reward function used to keep the hand at the joystick
    # TODO need to redo the way reward functions are implemented since they currently aren't cloned
    self.reward_function_joystick = kwargs.get('reward_function_joystick', NegativeExpDistance(shift=-1, scale=1))

    # Define the reward function used for the task controlled via joystick
    self.reward_function_target = kwargs.get('reward_function_target', NegativeExpDistance(shift=-1, scale=0.1))

    # Define bonus reward terms
    min_total_reward = self.max_episode_steps_extratime * (
          self.reward_function_joystick.get_min() + self.reward_function_target.get_min())
    self.reward_function_joystick_bonus = kwargs.get('reward_function_joystick_bonus',
                                                     RewardBonus(bonus=-min_total_reward, onetime=True))
    self.reward_function_target_bonus = kwargs.get('reward_function_target_bonus',
                                                   RewardBonus(bonus=8, onetime=False))

    # Define target origin, position, and limits
    self.target_origin = np.array([2, 0, 0])
    self.target_position = 1000 * self.target_origin.copy()
    self.target_limits = np.array([-2, 2])  # -y-direction

    # Define indices for target plane
    self.target_plane_geom_idx = None
    self.target_plane_body_idx = None

    # Define car position
    self.car_position = None

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Update plane location
    self.target_plane_geom_idx = model.geom_name2id("target-plane")
    self.target_plane_body_idx = model.body_name2id("target-plane")
    model.geom_size[self.target_plane_geom_idx] = \
      np.array([0.15, (self.target_limits[1] - self.target_limits[0]) / 2, 0.005])
    model.body_pos[self.target_plane_body_idx] = self.target_origin

    # Reset camera position and angle TODO need to rethink camera implementations
    model.cam_pos[model.camera_name2id('for_testing')] = np.array([-2, 0, 2.5])
    model.cam_quat[model.camera_name2id('for_testing')] = \
      model.body_quat[model.body_name2id("eye")].copy()

  @classmethod
  def initialise_task(cls, config):

    assert "end_effector" in config["simulator"]["task_kwargs"], \
      "End-effector must be defined for this environment"
    end_effector = config["simulator"]["task_kwargs"]["end_effector"]

    # Parse xml file
    tree = ET.parse(cls.xml_file)
    root = tree.getroot()

    # Add contact
    root.find("contact").append(ET.Element('pair', geom1="thumb-stick-1", geom2=end_effector, margin="100",
                                                 gap="100"))

    return tree

  def update_car_dynamics(self, model, data):
    """
    Sets the speed of the car based on the joystick state.
    Info: Negative throttle_control values result in forward motion, positive values in backward motion.
    """
    throttle_control = data.qpos[model._joint_name2id[self.joystick_joint]]
    throttle = throttle_control * self.throttle_factor
    data.qfrc_applied[model._joint_name2id[self.engine_back_joint]] = throttle
    data.qfrc_applied[model._joint_name2id[self.engine_front_joint]] = throttle

  def update(self, model, data):

    finished = False
    info = {"termination": False, "extra_time_given": False}

    # Update car dynamics
    self.update_car_dynamics(model, data)

    # Distance between car and target
    car_target_contact = [contact for contact in data.contact if
                          "target" in {model.geom_id2name(contact.geom1), model.geom_id2name(contact.geom2)}
                          and any([f"wheel{i}" in {model.geom_id2name(contact.geom1),
                                                   model.geom_id2name(contact.geom2)} for i in range(1, 5)])]
    contacts = [{model.geom_id2name(contact.geom1), model.geom_id2name(contact.geom2), contact.dist}
                for contact in car_target_contact]
    assert len(car_target_contact) >= 4, f"Cannot compute distance between car and target, since only {contacts} " \
                                         f"contacts are detected! Try increasing attribute 'margin' of geom 'target'."
    self.dist_car_to_bound = max(map(lambda x: x.dist, car_target_contact))  # WARNING: not differentiable!

    # Contact between body and joystick
    fingertip_joystick_contact = [contact for contact in data.contact if
                                  self.joystick_geom in {model.geom_id2name(contact.geom1),
                                                         model.geom_id2name(contact.geom2)}
                                  and self.end_effector in {
                                    model.geom_id2name(contact.geom1),
                                    model.geom_id2name(contact.geom2)}]
    assert len(fingertip_joystick_contact) >= 1
    self.dist_fingertip_to_joystick = fingertip_joystick_contact[0].dist

    # Increase counter
    self.steps += 1

    # Check if hand is kept at joystick
    if self.dist_fingertip_to_joystick <= 1e-3:
      if not self.fingertip_at_joystick:
        # # Provide extra time that is reset whenever joystick is touched:
        # self.max_episode_steps_with_extratime = self.steps + self.max_episode_steps_extratime

        # # Provide (constant) extra time if joystick is touched at least once within regular time:
        self.max_episode_steps_with_extratime = self.max_episode_steps + self.max_episode_steps_extratime
        info["extra_time_given"] = True

      self.fingertip_at_joystick = True
    else:
      self.fingertip_at_joystick = False
    info["fingertip_at_joystick"] = self.fingertip_at_joystick

    # Check if car has reached target
    if self.dist_car_to_bound <= 1e-3:  # dist < self.target_radius:
      # self.steps_inside_target += 1
      info["inside_target"] = True
    else:
      # self.steps_inside_target = 0
      info["inside_target"] = False

    # Check if target is inside target area with (close to) zero velocity
    if info["inside_target"] and np.abs(data.get_body_xvelp("car")[1]) <= self.car_velocity_threshold:  #TODO (see below)
      finished = True
      info["target_hit"] = True
    else:
      info["target_hit"] = False

    # Check if time limit has been reached
    if self.steps >= self.max_episode_steps_with_extratime:
      finished = True
      info["termination"] = "time_limit_reached"

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # car has reached the target
    # Reward to incentivize keeping the hand at the joystick
    reward = self.reward_function_joystick.get(self.dist_fingertip_to_joystick)
    reward += self.reward_function_joystick_bonus.get(info["fingertip_at_joystick"])  # should only be given once!

    # Reward to incentivize moving the car inside the target area and stopping there
    reward += self.reward_function_target.get(self.dist_car_to_bound)
    reward += self.reward_function_target_bonus.get(info["target_hit"])


    return reward, finished, info

  def spawn_car(self, model, data, rng):
    # Choose qpos value of slide joint in x-direction uniformly from joint angle range
    new_car_qpos = rng.uniform(*model.jnt_range[model._joint_name2id[self.car_joint]])

    data.qpos[model._joint_name2id[self.car_joint]] = new_car_qpos
    mujoco.mj_forward(model, data)

    # Get car position
    self.car_position = data.body(self.car_body).xpos.copy()

  def spawn_target(self, model, data, rng):

    # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
    for _ in range(10):
      target_rel = rng.uniform(*self.target_limits)
      # negative sign required, since y goes to left but car looks to the right
      new_position =  np.array([0, -target_rel, 0])
      distance = np.linalg.norm(self.car_position - (new_position + self.target_origin))
      if distance > self.new_target_distance_threshold:
        break
    self.target_position = new_position

    # Set location
    model.body_pos[model._body_name2id["target"]] = self.target_origin + self.target_position

    # Sample target half-size
    self.target_halfsize = rng.uniform(*self.target_halfsize_limit)

    # Set target half-size
    model.geom_size[model._geom_name2id["target"]][0] = self.target_halfsize

    mujoco.mj_forward(model, data)

  def get_state(self):
    state = super().get_state()
    state["joystick_xpos"] = self.data.geom(self.joystick_geom).xpos.copy()
    state["joystick_xmat"] = self.data.geom(self.joystick_geom).xmat.reshape((3, 3)).copy()
    state["joystick_xvelp"] = self.data.get_geom_xvelp(self.joystick_geom).copy()  #TODO (https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2646)
    state["joystick_xvelr"] = self.data.get_geom_xvelr(self.joystick_geom).copy()  #TODO
    state["fingertip_at_joystick"] = False
    state["dist_fingertip_to_joystick"] = self.dist_fingertip_to_joystick

    state["car_xpos"] = self.data.body(self.car_body).xpos.copy()
    state["car_xmat"] = self.data.body(self.car_body).xmat.reshape((3, 3)).copy()
    state["car_xvelp"] = self.data.get_body_xvelp(self.car_body).copy()  #TODO
    state["car_xvelr"] = self.data.get_body_xvelr(self.car_body).copy()  #TODO
    state["target_position"] = self.target_origin.copy() + self.target_position.copy()
    state["target_radius"] = self.target_halfsize
    state["target_hit"] = False
    state["inside_target"] = False
    state["dist_car_to_bound"] = self.dist_car_to_bound
    return state

  def reset(self, model, data, rng):

    # Reset counters
    self.steps = 0
    self.max_episode_steps_with_extratime = self.max_episode_steps

    # Reset bonus reward functions
    self.reward_function_joystick_bonus.reset()
    self.reward_function_target_bonus.reset()

    # Spawn a new car location
    # WARNING: needs to be executed AFTER setting qpos above!
    self.spawn_car(model, data, rng)

    # Spawn a new target location (depending on current car location)
    self.spawn_target(model, data, rng)