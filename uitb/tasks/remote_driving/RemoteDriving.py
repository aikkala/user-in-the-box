import numpy as np
import xml.etree.ElementTree as ET
import mujoco

from ..base import BaseTask


class RemoteDriving(BaseTask):

  def __init__(self, model, data, end_effector, **kwargs):
    super().__init__(model, data, **kwargs)

    # TODO currently works only with 'geom' type of end_effector
    self._end_effector = end_effector

    # Define names
    self._gamepad_body = "gamepad"
    self._joystick_geom = "thumb-stick-1"
    self._joystick_joint = "thumb-stick-1:rot-x"
    self._car_body = "car"
    self._car_joint = "car"
    self._engine_back_joint = "axis-back:rot-x"
    self._engine_front_joint = "axis-front:rot-x"
    self._wheels = [f"wheel{i}" for i in range(1, 5)]
    self._target = "target"

    # Define joystick parameters
    self._throttle_factor = 25

    # Define episode length
    episode_length_seconds = kwargs.get('episode_length_seconds', 4)
    self._max_episode_steps = kwargs.get('max_episode_steps', self._action_sample_freq * episode_length_seconds)

    # Episodes can be extended if joystick is touched
    episode_length_seconds_extratime = kwargs.get('episode_length_seconds_extratime', 6)
    self._max_episode_steps_extratime = kwargs.get('max_episode_steps_extratime',
                                                   self._action_sample_freq * episode_length_seconds_extratime)
    self._max_episode_steps_with_extratime = self._max_episode_steps  # extra time only applies if joystick is touched

    # Measure whether contact between end-effector and joystick exists
    self._ee_at_joystick = False

    # Measure distances between end-effector and joystick, and between car and target area
    self._dist_ee_to_joystick = 0
    self._dist_car_to_target = 0

    # Terminal velocity -- car needs to stay inside target area (i.e., velocity in x-direction needs to fall below some threshold)
    self._car_velocity_threshold = kwargs.get('car_velocity_threshold', 0.1)

    # Halfsize limits for target
    self._target_halfsize_limit = kwargs.get('target_halfsize_limit', np.array([0.15, 0.5]))
    self._target_halfsize = self._target_halfsize_limit[0]

    # Minimum distance between car and target is twice the max target halfsize limit
    self._new_target_distance_threshold = 2 * np.max(self._target_halfsize_limit)

    # For logging
    self._info = {"target_hit": False, "inside_target": False, "end_effector_at_joystick": False}

    # Get the reward function
    self._reward_function = self._get_reward_function(kwargs["reward_function"])

    # Define target origin, position, and limits
    self._target_origin = np.array([2, 0, 0])
    self._target_position = 1000 * self._target_origin.copy()
    self._target_limits = np.array([-2, 2])  # -y-direction

    # Define car position
    self._car_position = None

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Update plane location
    model.geom("target-plane").size = np.array([0.15, (self._target_limits[1] - self._target_limits[0]) / 2, 0.005])
    model.body("target-plane").pos = self._target_origin

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

  def _update_car_dynamics(self, model, data):
    """
    Sets the speed of the car based on the joystick state.
    Info: Negative throttle_control values result in forward motion, positive values in backward motion.
    """
    throttle_control = data.joint(self._joystick_joint).qpos
    if np.abs(throttle_control) < 1e-3:
      throttle_control = 0
    throttle = throttle_control * self._throttle_factor
    data.joint(self._engine_back_joint).qfrc_applied = throttle
    data.joint(self._engine_front_joint).qfrc_applied = throttle

  def _update(self, model, data):

    # Set defaults
    finished = False
    self._info = {"extra_time_given": False}

    # Update car dynamics
    self._update_car_dynamics(model, data)

    # Max distance between front/back wheel and center of target
    self._dist_car_to_target = np.abs(max([data.body("wheel1").xpos[1], data.body("wheel3").xpos[1]] - data.body("target").xpos[1]))

    # Contact between end-effector and joystick
    end_effector_joystick_contact = [contact for contact in data.contact if {contact.geom1, contact.geom2} ==
                                     {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._end_effector),
                                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self._joystick_geom)}]
    assert len(end_effector_joystick_contact) >= 1
    self._dist_ee_to_joystick = end_effector_joystick_contact[0].dist

    # Check if hand is kept at joystick
    if self._dist_ee_to_joystick <= 1e-3:
      if not self._end_effector_at_joystick:
        # # Provide extra time that is reset whenever joystick is touched:
        # self.max_episode_steps_with_extratime = self.steps + self.max_episode_steps_extratime

        # # Provide (constant) extra time if joystick is touched at least once within regular time:
        self._max_episode_steps_with_extratime = self._max_episode_steps + self._max_episode_steps_extratime
        self._info["extra_time_given"] = True

      self._end_effector_at_joystick = True
    else:
      self._end_effector_at_joystick = False
    self._info["end_effector_at_joystick"] = self._end_effector_at_joystick

    # Check if car has reached target
    if self._dist_car_to_target <= self.target_halfsize:
      self._info["inside_target"] = True
    else:
      self._info["inside_target"] = False

    # Check if target is inside target area with (close to) zero velocity
    if self._info["inside_target"] and np.abs(data.body(self._car_body).cvel[4]) <= self._car_velocity_threshold:
      finished = True
      self._info["target_hit"] = True
    else:
      self._info["target_hit"] = False

    # Check if time limit has been reached
    if self._steps >= self._max_episode_steps_with_extratime:
      finished = True
      self._info["termination"] = "time_limit_reached"

    # Calculate reward
    reward = self._reward_function.get(self._dist_ee_to_joystick, self._dist_car_to_target, self._info.copy(),
                                       model, data)

    return reward, finished, self._info.copy()

  def _spawn_car(self, model, data):
    # Choose qpos value of slide joint in x-direction uniformly from joint angle range
    new_car_qpos = self._rng.uniform(*model.joint(self._car_joint).range)

    data.joint(self._car_joint).qpos = new_car_qpos
    mujoco.mj_forward(model, data)

    # Get car position
    self.car_position = data.body(self._car_body).xpos.copy()

  def _spawn_target(self, model, data):

    # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
    for _ in range(10):
      target_rel = self._rng.uniform(*self._target_limits)
      # negative sign required, since y goes to left but car looks to the right
      new_position =  np.array([0, -target_rel, 0])
      distance = np.linalg.norm(self.car_position - (new_position + self._target_origin))
      if distance > self._new_target_distance_threshold:
        break
    self.target_position = new_position

    # Set location
    model.body("target").pos = self._target_origin + self._target_position

    # Sample target half-size
    self.target_halfsize = self._rng.uniform(*self._target_halfsize_limit)

    # Set target half-size
    model.geom("target").size[0] = self._target_halfsize

    mujoco.mj_forward(model, data)

  def _get_state(self, model, data):
    state = dict()
    state["joystick_xpos"] = data.geom(self._joystick_geom).xpos.copy()
    state["joystick_xmat"] = data.geom(self._joystick_geom).xmat.copy()
    state["dist_ee_to_joystick"] = self._dist_ee_to_joystick
    state["car_xpos"] = data.body(self._car_body).xpos.copy()
    state["car_xmat"] = data.body(self._car_body).xmat.copy()
    state["car_cvel"] = data.body("car").cvel.copy()
    state["target_position"] = self._target_origin.copy() + self._target_position.copy()
    state["target_radius"] = self._target_halfsize
    state["dist_car_to_target"] = self._dist_car_to_target
    state.update(self._info)
    return state

  def _reset(self, model, data):

    # Reset counters
    self._max_episode_steps_with_extratime = self._max_episode_steps
    self._info = {"target_hit": False, "inside_target": False, "end_effector_at_joystick": False}

    # Reset reward function
    self._reward_function.reset()

    # Spawn a new car location
    self._spawn_car(model, data)

    # Spawn a new target location (depending on current car location)
    self._spawn_target(model, data)
