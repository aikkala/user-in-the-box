import gym
from gym import spaces
import mujoco_py
import numpy as np
import os
import pathlib


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class MuscleActuatedWithCamera(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):

    # Get project path
    self.project_path = pathlib.Path(__file__).parent.absolute()

    # Model file
    xml_file = "models/mobl_arms_muscles.xml"

    # Set action sampling frequency
    self.action_sample_freq = 10
    self.timestep = 0.002
    self.frame_skip = int(1/(self.timestep*self.action_sample_freq))

    # Use early termination if target is not hit in time
    self.steps_since_last_hit = 0
    self.max_steps_without_hit = self.action_sample_freq*4

    # Max episode length
    self.steps = 0
    self.max_episode_length = self.action_sample_freq*60

    # Define area where targets will be spawned
    self.target_origin = np.array([0.5, 0.0, 0.8])
    self.target_position = self.target_origin.copy()
    self.target_limits_y = np.array([-0.3, 0.3])
    self.target_limits_z = np.array([-0.3, 0.3])

    # Radius limits for target
    if "target_radius_limit" in kwargs:
      self.target_radius_limit = kwargs["target_radius_limit"]
    else:
      self.target_radius_limit = np.array([0.01, 0.05])

    # Minimum distance to new spawned targets is twice the max target radius limit
    self.new_target_distance_threshold = 2*self.target_radius_limit[1]

    # RNG in case we need it
    self.rng = np.random.default_rng()

    # Initialise model and sim
    self.model = mujoco_py.load_model_from_path(os.path.join(self.project_path, xml_file))
    self.sim = mujoco_py.MjSim(self.model, nsubsteps=self.frame_skip)

    # Get indices of dependent and independent joints
    self.dependent_joints = np.unique(self.model.eq_obj1id[self.model.eq_active.astype(bool)])
    self.independent_joints = list(set(np.arange(self.model.nq)) - set(self.dependent_joints))

    # Number of muscles and motors
    self.nmuscles = np.sum(self.model.actuator_trntype == 3)
    self.nmotors = np.sum(self.model.actuator_trntype == 0)

    # Separate nq for eye and arm
    self.eye_nq = sum(self.model.jnt_bodyid[self.independent_joints]==self.model._body_name2id["eye"])
    self.arm_nq = len(self.independent_joints) - self.eye_nq

    # Set action space -- motor actuators are always first
    motors_limits = np.ones((self.nmotors,2)) * np.array([float('-inf'), float('inf')])
    muscles_limits = np.ones((self.nmuscles,2)) * np.array([float('-inf'), float('inf')])
    self.action_space = spaces.Box(low=np.float32(muscles_limits[:, 0]), high=np.float32(muscles_limits[:, 1]))

    # Fingertip is tracked for e.g. reward calculation and logging
    self.fingertip = "hand_2distph"

    # Set camera stuff, self._viewers needs to be initialised before self.get_observation() is called
    self.viewer = None
    self._viewers = {}
    self.metadata = {
      'render.modes': ['human', 'rgb_array', 'depth_array'],
      'video.frames_per_second': int(np.round(1.0 / (self.model.opt.timestep * self.frame_skip))),
      "imagesize": (1280, 800)
    }
    self.sim.model.cam_pos[self.sim.model._camera_name2id['for_testing']] = np.array([1.5, -1.5, 0.9])
    self.sim.model.cam_quat[self.sim.model._camera_name2id['for_testing']] = np.array([0.6582, 0.6577, 0.2590, 0.2588])

    # Reset
    observation = self.reset()

    # Set observation space
    self.observation_space = spaces.Dict({
      'proprioception': spaces.Box(low=-float('inf'), high=float('inf'), shape=observation['proprioception'].shape,
                                   dtype=np.float32),
      'visual': spaces.Box(low=-1, high=1, shape=observation['visual'].shape, dtype=np.float32),
      'ocular': spaces.Box(low=-float('inf'), high=float('inf'), shape=observation['ocular'].shape, dtype=np.float32)})


  def is_contact(self, idx1, idx2):
    for contact in self.sim.data.contact:
      if (contact.geom1 == idx1 and contact.geom2 == idx2) or (contact.geom1 == idx2 and contact.geom2 == idx1):
        return True
    return False

  def step(self, action):

    info = {}

    # Set motor and muscle control
    # Don't do anything with eyes for now
    self.sim.data.ctrl[2:] = np.clip(self.sim.data.act[:] + action, 0, 1)

    finished = False
    try:
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      finished = True
      info["termination"] = "MujocoException"

    # Get finger position
    finger_position = self.sim.data.get_geom_xpos(self.fingertip)

    # Distance to target
    dist = np.linalg.norm(self.target_position - (finger_position - self.target_origin))

    if dist < self.target_radius:

      # Spawn a new target
      self.spawn_target()

      # Reset counter, add hit bonus to reward
      self.steps_since_last_hit = 0
      velocity_factor = np.exp(-(self.sim.data.get_geom_xvelp(self.fingertip)**2).sum()*10)
      reward = 4 + velocity_factor*4

    else:

      # Estimate reward
      reward = np.exp(-dist*10)/10

      self.steps_since_last_hit += 1
      if self.steps_since_last_hit >= self.max_steps_without_hit:
        finished = True
        info["termination"] = "time_limit_reached"
      self.steps += 1
      if self.steps >= self.max_episode_length:
        finished = True
        info["termination"] = "episode_length_reached"

    return self.get_observation(), reward, finished, info

  def get_observation(self):
    # Ignore eye qpos and qvel for now
    jnt_range = self.sim.model.jnt_range[self.independent_joints]

    # Normalise qpos
    qpos = self.sim.data.qpos[self.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5)*2
    qvel = self.sim.data.qvel[self.independent_joints].copy()
    qacc = self.sim.data.qacc[self.independent_joints].copy()

    # Normalise act
    act = (self.sim.data.act.copy() - 0.5)*2

    # Estimate fingertip position, normalise to target_origin
    finger_position = self.sim.data.get_geom_xpos(self.fingertip) - self.target_origin

    # Get depth array and normalise
    render = self.sim.render(width=80, height=120, camera_name='oculomotor', depth=True)
    depth = render[1]
    depth = np.flipud((depth - 0.5)*2)
    rgb = render[0]
    rgb = np.flipud((rgb/255.0 - 0.5)*2)

    return {'proprioception': np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position, act]),
            #'visual': np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), (2, 0, 1)),
            'visual': np.expand_dims(depth, 0),
            'ocular': np.concatenate([qpos[:2], qvel[:2], qacc[:2]])}

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
    self.model.geom_size[self.model._geom_name2id["target-sphere"]][0] = self.target_radius

    self.sim.forward()

  def reset(self):

    self.sim.reset()
    self.steps_since_last_hit = 0
    self.steps = 0

    # Randomly sample qpos, qvel, act
    nq = len(self.independent_joints)
    qpos = self.rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    qvel = self.rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    act = self.rng.uniform(low=np.zeros((self.nmuscles,)), high=np.ones((self.nmuscles,)))

    # Set eye qpos and qvel to zero for now
    qpos[:2] = 0
    qvel[:2] = 0

    # Set qpos and qvel
    self.sim.data.qpos.fill(0)
    self.sim.data.qpos[self.independent_joints] = qpos
    self.sim.data.qvel.fill(0)
    self.sim.data.qvel[self.independent_joints] = qvel
    self.sim.data.act[:] = act

    # Spawn target
    self.spawn_target()

    # Do a forward so everything will be set
    self.sim.forward()

    return self.get_observation()

  def get_state(self):
    state = {"step": self.steps, "timestep": self.sim.data.time,
             "qpos": self.sim.data.qpos[self.independent_joints],
             "qvel": self.sim.data.qvel[self.independent_joints],
             "qacc": self.sim.data.qacc[self.independent_joints],
             "act": self.sim.data.act,
             "fingertip_xpos": self.sim.data.get_geom_xpos(self.fingertip),
             "fingertip_xmat": self.sim.data.get_geom_xmat(self.fingertip),
             "fingertip_xvelp": self.sim.data.get_geom_xvelp(self.fingertip),
             "fingertip_xvelr": self.sim.data.get_geom_xvelr(self.fingertip)}
    return state

  def render(self, mode='human', width=1280, height=800, camera_id=None, camera_name=None):

    if mode == 'rgb_array' or mode == 'depth_array':
        if camera_id is not None and camera_name is not None:
            raise ValueError("Both `camera_id` and `camera_name` cannot be"
                             " specified at the same time.")

        no_camera_specified = camera_name is None and camera_id is None
        if no_camera_specified:
            camera_name = 'track'

        if camera_id is None and camera_name in self.model._camera_name2id:
            camera_id = self.model.camera_name2id(camera_name)

        self._get_viewer(mode).render(width, height, camera_id=camera_id)

    if mode == 'rgb_array':
        # window size used for old mujoco-py:
        data = self._get_viewer(mode).read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]
    elif mode == 'depth_array':
        #self._get_viewer(mode).render(width, height)
        # window size used for old mujoco-py:
        # Extract depth part of the read_pixels() tuple
        data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
        # original image is upside-down, so flip it
        return data[::-1, :]
    elif mode == 'human':
        self._get_viewer(mode).render()

  def _get_viewer(self, mode):
    self.viewer = self._viewers.get(mode)
    if self.viewer is None:
      if mode == 'human':
        self.viewer = mujoco_py.MjViewer(self.sim)
      elif mode == 'rgb_array' or mode == 'depth_array':
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

      self._viewers[mode] = self.viewer
    return self.viewer

  def close(self):
    pass

  @property
  def dt(self):
    return self.model.opt.timestep * self.frame_skip

  def write_video(self, imgs, filepath):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, self.metadata["video.frames_per_second"], self.metadata["imagesize"])
    for img in imgs:
      out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()
