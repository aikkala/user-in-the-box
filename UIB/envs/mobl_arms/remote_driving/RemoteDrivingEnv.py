import numpy as np
import mujoco_py
from gym import spaces
from collections import deque
import xml.etree.ElementTree as ET
import os

from UIB.envs.mobl_arms.models.FixedEye import FixedEye
from UIB.envs.mobl_arms.remote_driving.reward_functions import NegativeExpDistance, RewardBonus
from UIB.utils.functions import project_path


class RemoteDrivingEnv(FixedEye):
  metadata = {'render.modes': ['human']}

  # Model file
  xml_file = os.path.join(project_path(), "envs/mobl_arms/models/variants/mobl_arms_muscles.xml")

  # Joystick
  gamepad_body = "gamepad"
  joystick_geom = "thumb-stick-1"
  joystick_joint = "thumb-stick-1:rot-x"

  # Car
  car_body = "car"
  car_joint = "car"
  engine_back_joint = "axis-back:rot-x"
  engine_front_joint = "axis-front:rot-x"
  wheels = [f"wheel{i}" for i in range(1,5)]

  # Target
  target = "target"

  def __init__(self, direction="vertical", **kwargs):

    # Modify the xml file first
    tree = ET.parse(self.xml_file)
    root = tree.getroot()

    worldbody = root.find('worldbody')

    worldbody.remove(worldbody.find("geom[@name='floor']"))
    thoraxbody = worldbody.find("body[@name='thorax']")
    thoraxbody.remove(thoraxbody.find("light"))
    eye = worldbody.find("body[@name='eye']")
    eye.attrib["euler"] = "0 -1.0 -1.57"

    # Include xml file with simulation options
    #include_options = ET.Element('include', file='options.xml')
    #root.append(include_options)

    #include_car_scene = ET.Element('include',  file=os.path.join(project_path(), 'envs/mobl_arms/remote_driving/models/scene.xml'))
    #root.append(include_car_scene)

    # Include additional xml files
    xml_file_car_scene = os.path.join(project_path(), 'envs/mobl_arms/remote_driving/models/scene_complete.xml')
    # tree_car_scene = ET.parse(xml_file_car_scene)
    # root_car_scene = tree_car_scene.getroot()

    root = XMLCombiner((root, xml_file_car_scene)).combine()

    # Include contact pair to measure distance between hand and joystick
    contact = root.find("contact")
    include_hand_joystick_contact = ET.Element('pair', geom1="thumb-stick-1", geom2="hand_2distph", margin="100", gap="100")
    contact.append(include_hand_joystick_contact)

    # Add a touch sensor
    thumb_stick_1 = worldbody.find("body[@name='gamepad']/body[@name='thumb-stick-1']")
    thumb_stick_1.append(ET.Element('site', name=f'thumb-stick-1', size="0.025", rgba="0.5 0.5 0.5 0.0"))
    if root.find('sensor') is None:
      root.append(ET.Element('sensor'))
    root.find('sensor').append(ET.Element('touch', name=f"thumb-stick-1-sensor", site=f"thumb-stick-1"))


    # If direction=="horizontal", change position and orientation of car and target scene
    if direction == "horizontal":
      worldbody = root.find('worldbody')
      car = worldbody.find(f"body[@name='{self.car_body}']")
      car.attrib["pos"] = "2 3 0.065"  #2 meters in front, 3 meters to the left (default position, corresponding to qpos=0 for car joint)
      car.attrib["euler"] = "0 0 -3.14"
    else:
      assert direction == "vertical", f"ERROR: Direction '{direction}' is not valid!"
    self.direction = direction

    # Save the modified XML file and replace old one
    xml_file = os.path.join(project_path(), 'envs/mobl_arms/models/variants/remote_driving_env.xml')
    with open(xml_file, 'w') as file:
      file.write(ET.tostring(root, encoding='unicode'))
    self.xml_file = xml_file

    # Now initialise variants with the modified XML file
    super().__init__(**kwargs)

    # Remove car and gamepad joints from list of "independent" joints, as these are actuated by the learned policy
    self.independent_joints = [i for i in self.independent_joints if self.sim.model.jnt_bodyid[i] < min((self.model._body_name2id[self.car_body], self.model._body_name2id[self.gamepad_body]))]

    # Define joystick parameters
    self.throttle_factor = 50

    # Define episode length
    episode_length_seconds = kwargs.get('episode_length_seconds', 4)
    self.max_episode_steps = kwargs.get('max_episode_steps', self.action_sample_freq*episode_length_seconds)
    self.steps = 0

    # Define a visual buffer; use a size equivalent to 0.1 seconds
    maxlen = 1 + int(0.1/self.dt)
    self.visual_buffer = deque(maxlen=maxlen)

    # Episodes can be extended if joystick is touched
    episode_length_seconds_extratime = kwargs.get('episode_length_seconds_extratime', 6)
    self.max_episode_steps_extratime = kwargs.get('max_episode_steps_extratime', self.action_sample_freq*episode_length_seconds_extratime)
    self.max_episode_steps_with_extratime = self.max_episode_steps  #extra time only applies if joystick is touched

    # Measure whether contact between fingertip and joystick exists
    self.fingertip_at_joystick = False

    # Measure distances between fingertip and joystick, and between car and target area
    self.dist_fingertip_to_joystick = 0
    self.dist_car_to_bound = 0

    # # Dwelling based selection -- car needs to be inside target for some time
    # self.steps_inside_target = 0
    # self.dwell_threshold = 0 * int(0.3*self.action_sample_freq)

    # Terminal velocity -- car needs to stay inside target area (i.e., velocity in x-direction needs to fall below some threshold)
    self.car_velocity_threshold = 0.1

    # Halfsize limits for target
    self.target_halfsize_limit = kwargs.get('target_halfsize_limit', np.array([0.15, 0.5]))

    # Minimum distance betwen car and target is twice the max target halfsize limit
    #self.new_target_distance_threshold = 2*np.max(np.hstack((self.target_halfsize_limit, self.model.geom_size[self.model._geom_name2id[self.car]])))
    self.new_target_distance_threshold = 2*np.max(self.target_halfsize_limit)

    # Define the reward function used to keep the hand at the joystick
    self.reward_function_joystick = kwargs.get('reward_function_joystick', NegativeExpDistance(shift=-1, scale=1))

    # Define the reward function used for the task controlled via joystick
    self.reward_function_target = kwargs.get('reward_function_target', NegativeExpDistance(shift=-1, scale=0.1))

    # Define bonus reward terms
    min_total_reward = self.max_episode_steps_extratime * (self.reward_function_joystick.get_min() + self.reward_function_target.get_min())
    self.reward_function_joystick_bonus = kwargs.get('reward_function_joystick_bonus', RewardBonus(bonus=-min_total_reward, onetime=True))
    self.reward_function_target_bonus = kwargs.get('reward_function_target_bonus', RewardBonus(bonus=8, onetime=False))

    # Do a forward step so stuff like geom and body positions are calculated
    self.sim.forward()

    # Define plane where targets will be spawned: 0.5m in front of shoulder, or the "humphant" body. Note that this
    # body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    #self.target_origin = np.array([0.5, 0.0, 0.8])
    if self.direction == "vertical":
      self.target_origin = self.sim.data.get_body_xpos("eye") * np.array([1, 1, 0]) + np.array([4, 0, 0])
      self.target_position = 1000 * self.target_origin.copy()
      self.target_limits = np.array([-2, 2])  #x-direction
    elif self.direction == "horizontal":
      self.target_origin = self.sim.data.get_body_xpos("eye") * np.array([1, 1, 0]) + np.array([2, 0, 0])
      self.target_position = 1000 * self.target_origin.copy()
      self.target_limits = np.array([-2, 2])  #-y-direction
    else:
      raise NotImplementedError

    # Update plane location
    self.target_plane_geom_idx = self.model._geom_name2id["target-plane"]
    self.target_plane_body_idx = self.model._body_name2id["target-plane"]
    self.model.geom_size[self.target_plane_geom_idx] = np.array([(self.target_limits[1] - self.target_limits[0])/2 if self.direction == "vertical" else 0.15,
                                                                (self.target_limits[1] - self.target_limits[0])/2 if self.direction == "horizontal" else 0.15,
                                                                0.005,
                                                                ])
    self.model.body_pos[self.target_plane_body_idx] = self.target_origin

    # Initialize car position (might be overwritten by reset())
    self.car_position = self.sim.data.get_body_xpos(self.car_body).copy()

    # Reset camera position and angle
    if direction == "vertical":
      self.sim.model.cam_pos[self.sim.model._camera_name2id['for_testing']] = np.array([-1, -3, 0.9])
      self.sim.model.cam_quat[self.sim.model._camera_name2id['for_testing']] = np.array([ -0.6593137, -0.6591959, 0.2552777, 0.256124 ])
    else:
      self.sim.model.cam_pos[self.sim.model._camera_name2id['for_testing']] = np.array([-2, 0, 2.5])
      self.sim.model.cam_quat[self.sim.model._camera_name2id['for_testing']] = self.sim.model.body_quat[self.sim.model._body_name2id["eye"]].copy()

  def update_car_dynamics(self):
    """
    Sets the speed of the car based on the joystick state.
    Info: Negative throttle_control values result in forward motion, positive values in backward motion.
    """
    throttle_control = self.sim.data.qpos[self.model._joint_name2id[self.joystick_joint]]
    throttle = throttle_control * self.throttle_factor
    self.sim.data.qfrc_applied[self.model._joint_name2id[self.engine_back_joint]] = throttle
    self.sim.data.qfrc_applied[self.model._joint_name2id[self.engine_front_joint]] = throttle

  def step(self, action, apply_car_dynamics=True):

    # Set muscle control
    self.set_ctrl(action)

    finished = False
    info = {"termination": False, "extra_time_given": False}
    try:
      # Update car dynamics
      if apply_car_dynamics:
        self.update_car_dynamics()

      # Run forward step for both body/joystick and car
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      finished = True
      info["termination"] = "MujocoException"

    # Distance between car and target
    #dist = np.linalg.norm(self.target_position - (car_position - self.target_origin))
    car_target_contact = [contact for contact in self.sim.data.contact if "target" in {self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)} and any([f"wheel{i}" in {self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)} for i in range(1, 5)])]
    assert len(car_target_contact) >= 4, f"Cannot compute distance between car and target, since only {[{self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2), contact.dist} for contact in car_target_contact]} contacts are detected! Try increasing attribute 'margin' of geom 'target'."
    self.dist_car_to_bound = max(map(lambda x: x.dist, car_target_contact))  #WARNING: not differentiable!
    #ALTERNATIVE:
    #self.dist_car_to_bound = np.mean(list(map(lambda x: x.dist, car_target_contact)))  #linear distance costs that keep sign (negative if inside target)

    # Contact between body and joystick
    fingertip_joystick_contact = [contact for contact in self.sim.data.contact if self.joystick_geom in {self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)} and self.fingertip in {self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)}]
    assert len(fingertip_joystick_contact) == 1
    self.dist_fingertip_to_joystick = fingertip_joystick_contact[0].dist

    # Increase counter
    self.steps += 1

    # Check if hand is kept at joystick
    if self.dist_fingertip_to_joystick <= 0:
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
    if self.dist_car_to_bound <= 0:  #dist < self.target_radius:
      #self.steps_inside_target += 1
      info["inside_target"] = True
    else:
      #self.steps_inside_target = 0
      info["inside_target"] = False

    # Check if target is inside target area with (close to) zero velocity
    if info["inside_target"] and np.abs(self.sim.data.get_body_xvelp("car")[0 if self.direction == "vertical" else 1]) <= self.car_velocity_threshold:  #and self.steps_inside_target >= self.dwell_threshold:
      finished = True
      info["target_hit"] = True
    else:
      info["target_hit"] = False

    # Check if time limit has been reached
    #if ((self.steps >= self.max_episode_steps) and not info["fingertip_at_joystick"]) or (self.steps >= self.max_episode_steps_with_extratime):
    if self.steps >= self.max_episode_steps_with_extratime:
      finished = True
      info["termination"] = "time_limit_reached"

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # car has reached the target
    # Reward to incentivize keeping the hand at the joystick
    reward = self.reward_function_joystick.get(self, self.dist_fingertip_to_joystick)
    reward += self.reward_function_joystick_bonus.get(info["fingertip_at_joystick"])  #should only be given once!

    # Reward to incentivize moving the car inside the target area and stopping there
    reward += self.reward_function_target.get(self, self.dist_car_to_bound)
    reward += self.reward_function_target_bonus.get(info["inside_target"])

    # Add an effort cost to reward
    reward -= self.effort_term.get(self)

    return self.get_observation(), reward, finished, info


  def spawn_car(self):
    # Choose qpos value of slide joint in x-direction uniformly from joint angle range
    new_car_qpos = np.random.uniform(*self.sim.model.jnt_range[self.sim.model._joint_name2id[self.car_joint]])

    self.sim.data.qpos[self.sim.model._joint_name2id[self.car_joint]] = new_car_qpos
    self.sim.forward()

    # Get car position
    self.car_position = self.sim.data.get_body_xpos(self.car_body).copy()

  def spawn_target(self):

    # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
    for _ in range(10):
      target_rel = self.rng.uniform(*self.target_limits)
      new_position = np.array([target_rel, 0, 0]) if self.direction == "vertical" else np.array([0, -target_rel, 0])  #negative sign required, since y goes to left but car looks to the right
      distance = np.linalg.norm(self.car_position - (new_position + self.target_origin))
      if distance > self.new_target_distance_threshold:
        break
    self.target_position = new_position

    # Set location
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position

    # Sample target half-size
    #self.target_halfsize = self.rng.uniform(*self.target_halfsize_limit, 2)
    self.target_halfsize = self.rng.uniform(*self.target_halfsize_limit)

    # Set target half-size
    #self.model.geom_size[self.model._geom_name2id["target"]][:2] = self.target_halfsize
    self.model.geom_size[self.model._geom_name2id["target"]][0] = self.target_halfsize

    self.sim.forward()

  def set_target_position(self, position):
    self.target_position = position.copy()
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position
    self.sim.forward()

  def set_target_halfsize(self, halfsize):
    self.target_halfsize = halfsize
    #self.model.geom_size[self.model._geom_name2id["target"]][:2] = self.target_halfsize
    self.model.geom_size[self.model._geom_name2id["target"]][0] = self.target_halfsize
    self.sim.forward()

  def get_state(self):
    state = super().get_state()
    state["joystick_xpos"] = self.sim.data.get_geom_xpos(self.joystick_geom).copy()
    state["joystick_xmat"] = self.sim.data.get_geom_xmat(self.joystick_geom).copy()
    state["joystick_xvelp"] = self.sim.data.get_geom_xvelp(self.joystick_geom).copy()
    state["joystick_xvelr"] = self.sim.data.get_geom_xvelr(self.joystick_geom).copy()
    state["fingertip_at_joystick"] = False
    state["dist_fingertip_to_joystick"] = self.dist_fingertip_to_joystick

    state["car_xpos"] = self.sim.data.get_body_xpos(self.car_body).copy()
    state["car_xmat"] = self.sim.data.get_body_xmat(self.car_body).copy()
    state["car_xvelp"] = self.sim.data.get_body_xvelp(self.car_body).copy()
    state["car_xvelr"] = self.sim.data.get_body_xvelr(self.car_body).copy()
    state["target_position"] = self.target_origin.copy()+self.target_position.copy()
    state["target_radius"] = self.target_halfsize
    state["target_hit"] = False
    state["inside_target"] = False
    state["dist_car_to_bound"] = self.dist_car_to_bound
    return state

  def reset(self):

    # Reset counters
    self.steps_since_last_hit = 0
    self.steps = 0
    self.steps_inside_target = 0
    self.trial_idx = 0
    self.targets_hit = 0
    self.max_episode_steps_with_extratime = self.max_episode_steps

    # Reset bonus reward functions
    self.reward_function_joystick_bonus.reset()
    self.reward_function_target_bonus.reset()

    ## Modified reset() from base environment:
    self.sim.reset()
    # Randomly sample qpos, qvel, act
    nq = len(self.independent_joints)
    qpos = np.zeros((nq,))  #TODO: set to valid initial posture
    qvel = np.zeros((nq,))  #TODO: set to valid initial posture
    act = np.zeros((self.model.na,))  #TODO: set to valid initial state
    # Set qpos and qvel
    self.sim.data.qpos.fill(0)
    self.sim.data.qpos[self.independent_joints] = qpos
    self.sim.data.qvel.fill(0)
    self.sim.data.qvel[self.independent_joints] = qvel
    self.sim.data.act[:] = act
    # Do a forward so everything will be set
    self.sim.forward()

    # Spawn a new car location
    # WARNING: needs to be executed AFTER setting qpos above!
    self.spawn_car()

    # Spawn a new target location (depending on current car location)
    self.spawn_target()

    return self.get_observation()


class ProprioceptionAndVisual(RemoteDrivingEnv):

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
    observation = super().get_observation()  #TODO: exclude fingertip from observation space?
    #TODO: include fingertip force measured via some sensor in observation space?

    # Use only R channel
    r = observation["visual"][:, :, 0, None]

    if len(self.visual_buffer) > 0:
      self.visual_buffer.pop()

    while len(self.visual_buffer) < self.visual_buffer.maxlen:
      self.visual_buffer.appendleft(r)

    # Use stacked images
    observation["visual"] = np.concatenate([self.visual_buffer[0], self.visual_buffer[-1],
                                            self.visual_buffer[-1] - self.visual_buffer[0]], axis=2)

    return observation

class hashabledict(dict):
    def __hash__(self):
      return hash(tuple(sorted(self.items())))

class XMLCombiner(object):
    def __init__(self, filenames):
      assert len(filenames) > 0, 'No filenames!'
      # save all the roots, in order, to be processed later
      self.roots = []
      for f in filenames:
        try:
          self.roots.append(ET.parse(f).getroot())
        except:
          # if f is not a file, it is already a root
          self.roots.append(f)

    def combine(self):
      for r in self.roots[1:]:
        # combine each element with the first one, and update that
        self.combine_element(self.roots[0], r)
      # return the string representation
      return self.roots[0]

    def combine_element(self, one, other):
      """
      This function recursively updates either the text or the children
      of an element if another element is found in `one`, or adds it
      from `other` if not found.
      """
      # Create a mapping from tag name to element, as that's what we are fltering with
      mapping = {(el.tag, hashabledict(el.attrib)): el for el in one}
      for el in other:
        if len(el) == 0:
          # Not nested
          try:
            # Update the text
            mapping[(el.tag, hashabledict(el.attrib))].text = el.text
          except KeyError:
            # An element with this name is not in the mapping
            mapping[(el.tag, hashabledict(el.attrib))] = el
            # Add it
            one.append(el)
        else:
          try:
            # Recursively process the element, and update it in the same way
            self.combine_element(mapping[(el.tag, hashabledict(el.attrib))], el)
          except KeyError:
            # Not in the mapping
            mapping[(el.tag, hashabledict(el.attrib))] = el
            # Just add it
            one.append(el)