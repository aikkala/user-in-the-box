import numpy as np
import mujoco_py
from gym import spaces
from collections import deque
import xml.etree.ElementTree as ET
import os

from UIB.envs.mobl_arms.models.FixedEye import FixedEye
from UIB.envs.mobl_arms.remote_driving.reward_functions import NegativeExpDistanceWithHitBonus, RewardJoystick
from UIB.utils.functions import project_path



class DrivingEnv(FixedEye):
    
    
  metadata = {'render.modes': ['human']}

  # Joystick
  joystick_geom = "thumb-stick-1"
  joystick_joint = "thumb-stick-1:rot-x"

  # Car
  engine_joint = "axis-back:rot-x"
  left_wheel_joint = "wheel4:rot-x"
  right_wheel_joint = "wheel3:rot-x"
  wheels = [f"wheel{i}" for i in range(1,5)]

  # Target
  target = "target"

  def __init__(self, **kwargs):

    # Modify the xml file first
    tree = ET.parse(self.xml_file)
    root = tree.getroot()
    
    worldbody = root.find('worldbody')
    worldbody.remove(worldbody.find("body[@name='target']"))
    worldbody.remove(worldbody.find("body[@name='target-estimate']"))
    worldbody.remove(worldbody.find("body[@name='target-plane']"))

    # Include xml file with simulation options
    #include_options = ET.Element('include', file='options.xml')
    #root.append(include_options)

    # Include additional xml files
    #include_car_scene = ET.Element('include',  file=os.path.join(project_path(), 'envs/mobl_arms/remote_driving/models/scene.xml'))
    #root.append(include_car_scene)
    
    xml_file_2 = os.path.join(project_path(), 'envs/mobl_arms/remote_driving/models/scene_complete.xml')
    
    root = XMLCombiner((root, xml_file_2)).combine()


    # Save the modified XML file and replace old one
    xml_file = os.path.join(project_path(), 'envs/mobl_arms/models/variants/remote_driving_env.xml')
    with open(xml_file, 'w') as file:
      file.write(ET.tostring(root, encoding='unicode'))
    self.xml_file = xml_file

    # Now initialise variants with the modified XML file
    super().__init__(**kwargs)

    # Define joystick parameters
    self.throttle_factor = 50

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

    # Halfsize limits for target
    self.target_halfsize_limit = kwargs.get('target_halfsize_limit', np.array([0.15, 0.5]))

    # Minimum distance to new spawned targets is twice the max target radius limit #(or the size of the car, if larger)
    #self.new_target_distance_threshold = 2*np.max(np.hstack((self.target_halfsize_limit, self.model.geom_size[self.model._geom_name2id[self.car]])))
    self.new_target_distance_threshold = 2*np.max(self.target_halfsize_limit)

    # Define a default reward function (used for the task controlled via joystick)
    if self.reward_function is None:
      self.reward_function = NegativeExpDistanceWithHitBonus()

    # Define an additional reward function (used to keep the hand at the joystick)
    self.reward_function_joystick = RewardJoystick()

    # Do a forward step so stuff like geom and body positions are calculated
    self.sim.forward()

    # Define plane where targets will be spawned: 0.5m in front of shoulder, or the "humphant" body. Note that this
    # body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    #self.target_origin = np.array([0.5, 0.0, 0.8])
    self.target_origin = self.sim.data.get_body_xpos("eye") * np.array([1, 1, 0]) + np.array([3, 0, 0])
    self.target_position = 1000 * self.target_origin.copy()
    self.target_limits_x = np.array([-2, 2])
    #self.target_limits_y = np.array([-5, 5])

    # Update plane location
    self.target_plane_geom_idx = self.model._geom_name2id["target-plane"]
    self.target_plane_body_idx = self.model._body_name2id["target-plane"]
    self.model.geom_size[self.target_plane_geom_idx] = np.array([(self.target_limits_x[1] - self.target_limits_x[0])/2,
                                                                0.15, #(self.target_limits_y[1] - self.target_limits_y[0])/2
                                                                0.005,
                                                                ])
    self.model.body_pos[self.target_plane_body_idx] = self.target_origin

  def update_car_dynamics(self):
    """
    Sets the speed of the car based on the joystick state.
    """
    throttle_control = self.sim.data.qpos[self.model._joint_name2id[self.joystick_joint]]
    throttle = throttle_control * self.throttle_factor
    self.sim.data.qfrc_applied[self.model._joint_name2id[self.engine_joint]] = throttle
    self.sim.data.qfrc_applied[self.model._joint_name2id[self.left_wheel_joint]] = -throttle
    self.sim.data.qfrc_applied[self.model._joint_name2id[self.right_wheel_joint]] = throttle

  def step(self, action):

    # Set muscle control
    self.set_ctrl(action)

    finished = False
    info = {"termination": False, "target_spawned": False}
    try:
      # Update car dynamics
      self.update_car_dynamics()

      # Run forward step for both body/joystick and car
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      
      print("MuJoCo Exception!") #DEBUG
      finished = True
      info["termination"] = "MujocoException"

    # Get car position
    #car_position = self.sim.data.get_geom_xpos(self.car)

    # Distance between car and target
    #dist = np.linalg.norm(self.target_position - (car_position - self.target_origin))
    car_target_contact = []
    
    #[contact for contact in self.sim.data.contact if "target" in {contact.geom1, contact.geom2} and any([f"wheel{i}" in {contact.geom1, contact.geom2} for i in range(1, 5)])]
    
    
    #DEBUG
    for contact in self.sim.data.contact:
        if (self.sim.model.geom_id2name(contact.geom1) in self.wheels or self.sim.model.geom_id2name(contact.geom2) in self.wheels) and (self.sim.model.geom_id2name(contact.geom1) == self.target or self.sim.model.geom_id2name(contact.geom2) == self.target):
            car_target_contact.append(contact)
    print(car_target_contact)
    
    assert len(car_target_contact) == 4, "Cannot compute distance between car and target, since no contact is detected! Try increasing attribute 'margin' of geom 'target'."
    dist_car_to_bound = max(map(lambda x: x.dist, car_target_contact))  #WARNING: not differentiable!
    #ALTERNATIVE:
    #dist_car_to_bound = np.mean(list(map(lambda x: x.dist, car_target_contact)))  #linear distance costs that keep sign (negative if inside target)

    # Contact between body and joystick
    body_joystick_contact = [contact for contact in self.sim.data.contact if self.joystick_geom in {contact.geom1, contact.geom2}]
    body_at_joystick = len(body_joystick_contact) >= 1

    # Check if car has reached target
    if dist_car_to_bound < 0:  #dist < self.target_radius:
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

    # # Check if hand is kept at joystick  #might be very hard...
    # if not body_at_joystick:
    #   finished = True
    #   info["termination"] = "joystick_released"

    # Check if max number trials reached
    if self.trial_idx >= self.max_trials:
      finished = True
      info["termination"] = "max_trials_reached"

    # Increase counter
    self.steps += 1

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # car has reached the target
    reward = self.reward_function.get(self, dist_car_to_bound, info)

    # Add a bonus reward to incentivize keeping the hand at the joystick
    #reward += self.reward_function.get(self, dist_body_to_joystick, info)
    reward += self.reward_function_joystick.get(body_at_joystick)

    # Add an effort cost to reward
    reward -= self.effort_term.get(self)

    return self.get_observation(), reward, finished, info

  def spawn_target(self):

    # Sample a location; try 10 times then give up (if e.g. self.new_target_distance_threshold is too big)
    for _ in range(10):
      target_x = self.rng.uniform(*self.target_limits_x)
      #target_y = self.rng.uniform(*self.target_limits_y)
      #new_position = np.array([target_x, target_y, 0])
      new_position = np.array([target_x, 0, 0])
      distance = np.linalg.norm(self.target_position - new_position)
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

  def reset(self):

    # Reset counters
    self.steps_since_last_hit = 0
    self.steps = 0
    self.steps_inside_target = 0
    self.trial_idx = 0
    self.targets_hit = 0

    # Spawn a new location
    self.spawn_target()

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

    return self.get_observation()


class ProprioceptionAndVisual(DrivingEnv):

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

    # Time features (time left to reach target, time spent inside target)
    targets_hit = -1.0 + 2*(self.trial_idx/self.max_trials)
    dwell_time = -1.0 + 2*np.min([1.0, self.steps_inside_target/self.dwell_threshold])

    # Append to proprioception since those will be handled with a fully-connected layer
    observation["proprioception"] = np.concatenate([observation["proprioception"], np.array([dwell_time, targets_hit])])

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