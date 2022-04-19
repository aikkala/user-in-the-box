import numpy as np
import mujoco_py
from gym import spaces
from collections import deque
import xml.etree.ElementTree as ET
import os

from UIB.envs_old_to_be_removed.mobl_arms.models.FixedEye import FixedEye
from UIB.envs_old_to_be_removed.mobl_arms.tracking.reward_functions import NegativeDistance
from UIB.utils.functions import project_path


def add_target(worldbody):
  target = ET.Element('body', name='target', pos="0.5 0 0.8")
  target.append(ET.Element('geom', name='target', type="sphere", size="0.025", rgba="0.1 0.8 0.1 1.0"))
  worldbody.append(target)

def add_target_plane(worldbody):
  target_plane = ET.Element('body', name='target-plane', pos='0.5 0 0.8')
  target_plane.append(ET.Element('geom', name='target-plane', type='box', size='0.005 0.3 0.3', rgba='0.1 0.8 0.1 0'))
  worldbody.append(target_plane)


class TrackingEnv(FixedEye):

  def __init__(self, **kwargs):

    # Modify the xml file first
    tree = ET.parse(os.path.join(project_path(), self.xml_file))
    root = tree.getroot()
    worldbody = root.find('worldbody')

    # Add target and target plane -- exact coordinates and size don't matter, they are set later
    add_target(worldbody)
    add_target_plane(worldbody)

    # Save the modified XML file and replace old one
    self.xml_file = os.path.join(project_path(), f'envs/mobl_arms/models/variants/tracking_env.xml')
    with open(self.xml_file, 'w') as file:
      file.write(ET.tostring(tree.getroot(), encoding='unicode'))

    # Initialise base model
    super().__init__(**kwargs)

    # Define episode length
    episode_length_seconds = kwargs.get('episode_length_seconds', 4)
    self.max_episode_steps = kwargs.get('max_episode_steps', self.action_sample_freq*episode_length_seconds)
    self.steps = 0

    # Define some limits for target movement speed
    self.min_frequency = 0.0
    self.max_frequency = 0.5
    self.freq_curriculum = kwargs.get('freq_curriculum', lambda : 1.0)

    # Define a default reward function
    if self.reward_function is None:
      self.reward_function = NegativeDistance()

    # Define a vision buffer; use a size equivalent to 0.1 seconds
    maxlen = 1 + int(0.1/self.dt)
    self.visual_buffer = deque(maxlen=maxlen)

    # Target radius
    self.target_radius = kwargs.get('target_radius', 0.05)
    self.model.geom_size[self.model._geom_name2id["target"]][0] = self.target_radius

    # Do a forward step so stuff like geom and body positions are calculated
    self.sim.forward()

    # Define plane where targets will move: 0.5m in front of shoulder, or the "humphant" body. Note that this
    # body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    self.target_origin = self.sim.data.get_body_xpos("humphant") + np.array([0.5, 0, 0])
    self.target_position = self.target_origin.copy()
    self.target_limits_y = np.array([-0.3, 0.3])
    self.target_limits_z = np.array([-0.3, 0.3])

    # Update plane location
    self.target_plane_geom_idx = self.model._geom_name2id["target-plane"]
    self.target_plane_body_idx = self.model._body_name2id["target-plane"]
    self.model.geom_size[self.target_plane_geom_idx] = np.array([0.005,
                                                            (self.target_limits_y[1] - self.target_limits_y[0])/2,
                                                            (self.target_limits_z[1] - self.target_limits_z[0])/2])
    self.model.body_pos[self.target_plane_body_idx] = self.target_origin

    # Generate trajectory
    self.sin_y, self.sin_z = self.generate_trajectory()

    self.sim.model.cam_pos[self.sim.model._camera_name2id['for_testing']] = np.array([-0.8, -0.6, 1.5])
    self.sim.model.cam_quat[self.sim.model._camera_name2id['for_testing']] = np.array(
      [0.718027, 0.4371043, -0.31987, -0.4371043])

  def step(self, action):

    info = {}

    # Set muscle control
    self.set_ctrl(action)

    finished = False
    info["termination"] = False
    try:
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      finished = True
      info["termination"] = "MujocoException"

    # Get finger position
    finger_position = self.sim.data.get_geom_xpos(self.fingertip)

    # Distance to target origin
    dist = np.linalg.norm(self.target_position - (finger_position - self.target_origin))

    # Is fingertip inside target?
    if dist <= self.target_radius:
      info["inside_target"] = True
    else:
      info["inside_target"] = False

    # Check if time limit has been reached
    self.steps += 1
    if self.steps >= self.max_episode_steps:
      finished = True
      info["termination"] = "time_limit_reached"

    # Calculate reward; note, inputting distance to surface into reward function, hence distance can be negative if
    # fingertip is inside target
    reward = self.reward_function.get(self, dist-self.target_radius, info)

    # Add an effort cost to reward
    reward -= self.effort_term.get(self)

    # Update target location
    self.update_target_location()

    return self.get_observation(), reward, finished, info

  def reset(self):

    # Reset counters
    self.steps = 0

    # Reset vision buffer
    self.visual_buffer.clear()

    # Generate a new trajectory
    self.sin_y, self.sin_z = self.generate_trajectory()

    # Update target location
    self.update_target_location()

    return super().reset()

  def generate_trajectory(self):
    sin_y = self.generate_sine_wave(self.target_limits_y, num_components=5)
    sin_z = self.generate_sine_wave(self.target_limits_z, num_components=5)
    return sin_y, sin_z

  def generate_sine_wave(self, limits, num_components=5, min_amplitude=1, max_amplitude=5):

    max_frequency = self.min_frequency + (self.max_frequency-self.min_frequency) * self.freq_curriculum()

    # Generate a sine wave with multiple components
    t = np.arange(self.max_episode_steps+1) * self.dt
    sine = np.zeros((t.size,))
    sum_amplitude = 0
    for _ in range(num_components):
      amplitude = self.rng.uniform(min_amplitude, max_amplitude)
      sine +=  amplitude *\
              np.sin(self.rng.uniform(self.min_frequency, max_frequency)*2*np.pi*t + self.rng.uniform(0, 2*np.pi))
      sum_amplitude += amplitude

    # Normalise to fit limits
    sine = (sine + sum_amplitude) / (2*sum_amplitude)
    sine = limits[0] + (limits[1] - limits[0])*sine

    return sine

  def update_target_location(self):
    self.target_position[0] = 0
    self.target_position[1] = self.sin_y[self.steps]
    self.target_position[2] = self.sin_z[self.steps]
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position
    self.sim.forward()


class ProprioceptionAndVisual(TrackingEnv):

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

    depth = observation["vision"][:, :, 3, None]

    if len(self.visual_buffer) > 0:
      self.visual_buffer.pop()

    while len(self.visual_buffer) < self.visual_buffer.maxlen:
      self.visual_buffer.appendleft(depth)

    # Use only depth image
    #observation["vision"] = np.concatenate([self.visual_buffer[0], self.visual_buffer[-1]], axis=2)
    observation["vision"] = np.concatenate([self.visual_buffer[0], self.visual_buffer[-1],
                                            self.visual_buffer[-1] - self.visual_buffer[0]], axis=2)

    return observation
