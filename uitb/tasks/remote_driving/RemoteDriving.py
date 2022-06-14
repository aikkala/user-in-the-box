import numpy as np
import xml.etree.ElementTree as ET
import mujoco

from ..base import BaseTask


class RemoteDriving(BaseTask):

  def __init__(self, model, data, end_effector, **kwargs):
    super().__init__(model, data, **kwargs)

    self.end_effector = end_effector

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
    episode_length_seconds = kwargs.get('episode_length_seconds', 4)
    self.max_episode_steps = kwargs.get('max_episode_steps', self.action_sample_freq * episode_length_seconds)
    self.steps = 0

    # Episodes can be extended if joystick is touched
    episode_length_seconds_extratime = kwargs.get('episode_length_seconds_extratime', 6)
    self.max_episode_steps_extratime = kwargs.get('max_episode_steps_extratime',
                                                  self.action_sample_freq * episode_length_seconds_extratime)
    self.max_episode_steps_with_extratime = self.max_episode_steps  # extra time only applies if joystick is touched

    # Measure whether contact between end-effector and joystick exists
    self.ee_at_joystick = False

    # Measure distances between end-effector and joystick, and between car and target area
    self.dist_ee_to_joystick = 0
    self.dist_car_to_target = 0

    # Terminal velocity -- car needs to stay inside target area (i.e., velocity in x-direction needs to fall below some threshold)
    self.car_velocity_threshold = kwargs.get('car_velocity_threshold', 0.1)

    # Halfsize limits for target
    self.target_halfsize_limit = kwargs.get('target_halfsize_limit', np.array([0.15, 0.5]))
    self.target_halfsize = self.target_halfsize_limit[0]

    # Minimum distance between car and target is twice the max target halfsize limit
    self.new_target_distance_threshold = 2 * np.max(self.target_halfsize_limit)

    # Get the reward function
    self.reward_function = self.get_reward_function(kwargs["reward_function"])

    # Define target origin, position, and limits
    self.target_origin = np.array([2, 0, 0])
    self.target_position = 1000 * self.target_origin.copy()
    self.target_limits = np.array([-2, 2])  # -y-direction

    # Define car position
    self.car_position = None

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Update plane location
    model.geom("target-plane").size = np.array([0.15, (self.target_limits[1] - self.target_limits[0]) / 2, 0.005])
    model.body("target-plane").pos = self.target_origin

    # Reset camera position and angle TODO need to rethink camera implementations
    model.camera("for_testing").pos = np.array([-2, 0, 2.5])
    model.camera("for_testing").quat = np.array([0.583833, 0.399104, -0.399421, -0.583368])

  @classmethod
  def initialise(cls, task_kwargs):

    if "end_effector" not in task_kwargs:
      raise KeyError("Key 'end_effector' is missing from task kwargs. The end-effector must be defined for this "
                     "environment")

    # Parse xml file
    tree = ET.parse(cls.get_xml_file())
    root = tree.getroot()

    # Add contact
    root.find("contact").append(ET.Element('pair', geom1="thumb-stick-1", geom2=task_kwargs["end_effector"],
                                           margin="100", gap="100"))

    # Add touch sensor
    thumb_stick_1 = root.find(".//body[@name='thumb-stick-1']")
    thumb_stick_1.append(ET.Element('site', name=f'thumb-stick-1', size="0.025", rgba="0.5 0.5 0.5 0.0"))
    if root.find('sensor') is None:
      root.append(ET.Element('sensor'))
    root.find('sensor').append(ET.Element('touch', name=f"thumb-stick-1-sensor", site=f"thumb-stick-1"))

    return tree

  def update_car_dynamics(self, model, data):
    """
    Sets the speed of the car based on the joystick state.
    Info: Negative throttle_control values result in forward motion, positive values in backward motion.
    """
    throttle_control = data.joint(self.joystick_joint).qpos
    throttle = throttle_control * self.throttle_factor
    data.joint(self.engine_back_joint).qfrc_applied = throttle
    data.joint(self.engine_front_joint).qfrc_applied = throttle

  def update(self, model, data):

    finished = False
    info = {"termination": False, "extra_time_given": False}

    # Update car dynamics
    self.update_car_dynamics(model, data)

    # Max distance between front/back wheel and center of target
    self.dist_car_to_target = np.abs(max([data.body("wheel1").xpos[1], data.body("wheel3").xpos[1]] - data.body("target").xpos[1]))

    # Contact between end-effector and joystick
    end_effector_joystick_contact = [contact for contact in data.contact if {contact.geom1, contact.geom2} ==
                                     {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.end_effector),
                                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.joystick_geom)}]
    assert len(end_effector_joystick_contact) >= 1
    self.dist_ee_to_joystick = end_effector_joystick_contact[0].dist

    # Increase counter
    self.steps += 1

    # Check if hand is kept at joystick
    if self.dist_ee_to_joystick <= 1e-3:
      if not self.end_effector_at_joystick:
        # # Provide extra time that is reset whenever joystick is touched:
        # self.max_episode_steps_with_extratime = self.steps + self.max_episode_steps_extratime

        # # Provide (constant) extra time if joystick is touched at least once within regular time:
        self.max_episode_steps_with_extratime = self.max_episode_steps + self.max_episode_steps_extratime
        info["extra_time_given"] = True

      self.end_effector_at_joystick = True
    else:
      self.end_effector_at_joystick = False
    info["end_effector_at_joystick"] = self.end_effector_at_joystick

    # Check if car has reached target
    if self.dist_car_to_target <= self.target_halfsize:  # dist < self.target_radius:
      # self.steps_inside_target += 1
      info["inside_target"] = True
    else:
      # self.steps_inside_target = 0
      info["inside_target"] = False

    # Check if target is inside target area with (close to) zero velocity
    if info["inside_target"] and np.abs(data.body(self.car_body).cvel[4]) <= self.car_velocity_threshold:
      finished = True
      info["target_hit"] = True
    else:
      info["target_hit"] = False

    # Check if time limit has been reached
    if self.steps >= self.max_episode_steps_with_extratime:
      finished = True
      info["termination"] = "time_limit_reached"

    # Calculate reward
    reward = self.reward_function.get(self.dist_ee_to_joystick, self.dist_car_to_target, info, model, data)


    return reward, finished, info

  def spawn_car(self, model, data):
    # Choose qpos value of slide joint in x-direction uniformly from joint angle range
    new_car_qpos = self.rng.uniform(*model.joint(self.car_joint).range)

    data.joint(self.car_joint).qpos = new_car_qpos
    mujoco.mj_forward(model, data)

    # Get car position
    self.car_position = data.body(self.car_body).xpos.copy()

  def spawn_target(self, model, data):

    # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
    for _ in range(10):
      target_rel = self.rng.uniform(*self.target_limits)
      # negative sign required, since y goes to left but car looks to the right
      new_position =  np.array([0, -target_rel, 0])
      distance = np.linalg.norm(self.car_position - (new_position + self.target_origin))
      if distance > self.new_target_distance_threshold:
        break
    self.target_position = new_position

    # Set location
    model.body("target").pos = self.target_origin + self.target_position

    # Sample target half-size
    self.target_halfsize = self.rng.uniform(*self.target_halfsize_limit)

    # Set target half-size
    model.geom("target").size[0] = self.target_halfsize

    mujoco.mj_forward(model, data)

  def get_state(self):
    state = super().get_state()
    state["joystick_xpos"] = self.data.geom(self.joystick_geom).xpos.copy()
    state["joystick_xmat"] = self.data.geom(self.joystick_geom).xmat.reshape((3, 3)).copy()
    joystick_xvelp, joystick_xvelr = self._get_geom_xvelp_xvelr(self.model, self.data, self.joystick_geom)
    state["joystick_xvelp"] = joystick_xvelp
    state["joystick_xvelr"] = joystick_xvelr
    state["end_effector_at_joystick"] = False
    state["dist_ee_to_joystick"] = self.dist_ee_to_joystick

    state["car_xpos"] = self.data.body(self.car_body).xpos.copy()
    state["car_xmat"] = self.data.body(self.car_body).xmat.reshape((3, 3)).copy()
    car_xvelp, car_xvelr = self._get_body_xvelp_xvelr(self.model, self.data, self.car_body)
    state["car_xvelp"] = car_xvelp
    state["car_xvelr"] = car_xvelr
    state["target_position"] = self.target_origin.copy() + self.target_position.copy()
    state["target_radius"] = self.target_halfsize
    state["target_hit"] = False
    state["inside_target"] = False
    state["dist_car_to_target"] = self.dist_car_to_target
    return state

  def reset(self, model, data):

    # Reset counters
    self.steps = 0
    self.max_episode_steps_with_extratime = self.max_episode_steps

    # Reset reward function
    self.reward_function.reset()

    # Spawn a new car location
    self.spawn_car(model, data)

    # Spawn a new target location (depending on current car location)
    self.spawn_target(model, data)