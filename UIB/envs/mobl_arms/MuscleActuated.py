import gym
from gym import spaces
import mujoco_py
import numpy as np
import os
import pathlib


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class MuscleActuated(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, xml_file=None, sample_target=True):

    # Get project path
    self.project_path = pathlib.Path(__file__).parent.absolute()

    # Model file
    if not xml_file:
      xml_file = "models/mobl_arms_muscles.xml"

    # Set action sampling frequency
    self.action_sample_freq = 10
    self.timestep = 0.002
    self.frame_skip = int(1/(self.timestep*self.action_sample_freq))

    # Use early termination if target is not hit in time
    self.steps_since_last_hit = 0
    self.max_steps_without_hit = self.action_sample_freq*4

    # Define area where targets will be spawned
    self.target_origin = np.array([0.5, 0.0, 0.8])
    self.target_position = self.target_origin.copy()
    self.target_limits_y = np.array([-0.3, 0.3])
    self.target_limits_z = np.array([-0.3, 0.3])

    # Minimum distance to new spawned targets
    self.new_target_distance_threshold = 0.1

    # Radius limits for target
    self.target_radius_limit = np.array([0.01, 0.05])

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
    muscles_limits = np.ones((self.nmuscles,2)) * np.array([float('-inf'), float('inf')])#np.array([0, 1])
    limits = np.concatenate([motors_limits, muscles_limits])
    self.action_space = spaces.Box(low=np.float32(muscles_limits[:, 0]), high=np.float32(muscles_limits[:, 1]))
    #self.action_space = spaces.MultiBinary(self.nmuscles)

    # Reset
    observation = self.reset(sample_target=sample_target)

    # Set observation space
    low = np.ones_like(observation)*-float('inf')
    high = np.ones_like(observation)*float('inf')
    self.observation_space = spaces.Box(low=np.float32(low), high=np.float32(high))

    # Set camera stuff
    self.viewer = None
    self._viewers = {}
    self.metadata = {
      'render.modes': ['human', 'rgb_array', 'depth_array'],
      'video.frames_per_second': int(np.round(1.0 / (self.model.opt.timestep * self.frame_skip))),
      "imagesize": (1280, 800)
    }

  def is_contact(self, idx1, idx2):
    for contact in self.sim.data.contact:
      if (contact.geom1 == idx1 and contact.geom2 == idx2) or (contact.geom1 == idx2 and contact.geom2 == idx1):
        return True
    return False

  def step(self, action):

    info = {}

    # Set motor and muscle control
    # Don't do anything with eyes for now
    #self.sim.data.ctrl[:] = sigmoid(action)
    self.sim.data.ctrl[2:] = np.clip(self.sim.data.act[:] + action*0.2, 0, 1)
    #self.sim.data.ctrl[:2] = 0
    #self.sim.data.ctrl[2:] = np.clip(self.sim.data.ctrl[2:] + (action-0.5)*0.4, 0, 1)

    finished = False
    try:
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      finished = True
      info["termination"] = "MujocoException"

    # Check if target is hit
    fingertip_idx = self.model._geom_name2id["hand_2distph"]

    # Get finger position
    finger_position = self.sim.data.geom_xpos[fingertip_idx]

    # Distance to target
    dist = np.linalg.norm(self.target_position - (finger_position - self.target_origin))

    if dist < self.target_radius:

      # Spawn a new target
      self.spawn_target()

      # Reset counter, add hit bonus to reward
      self.steps_since_last_hit = 0
      reward = 20

    else:

      # Estimate reward
      reward = np.exp(-dist*10)

      self.steps_since_last_hit += 1
      if self.steps_since_last_hit >= self.max_steps_without_hit:
        finished = True
        info["termination"] = "time_limit_reached"

    # A small cost on controls
    reward -= 1e-3 * np.sum(self.sim.data.ctrl)

    return self.get_observation(), reward, finished, info

  def get_state(self):

    # Get cartesian positions of target geoms
    xpos = self.sim.data.geom_xpos[self.geom_target_indices, :].copy()

    # Get quaternions of target geoms
    xquat = R.from_matrix(np.reshape(self.sim.data.geom_xmat[self.geom_target_indices, :].copy(),
                                     (len(self.geom_target_indices), 3, 3))).as_quat()

    # Get translation velocity of target geoms
    xvelp = self.sim.data.geom_xvelp[self.geom_target_indices, :].copy()

    # Get get angular velocity of target geoms
    #xvelr = self.sim.data.geom_xvelr[self.geom_target_indices, :].copy()

    # Needed for egocentric observaitons
    v_pelvis = self.sim.data.geom_xpos[self.model._geom_name2id["pelvis"], : ] - \
               self.sim.data.geom_xpos[self.model._geom_name2id["l_pelvis"], :]

    # State includes everything
    state = {"xpos": xpos, "xquat": xquat, "xvelp": xvelp, "root_xpos": xpos[0, :], "root_xquat": xquat[0, :],
             "qpos": self.sim.data.qpos.copy(), "qvel": self.sim.data.qvel.copy(), "qacc": self.sim.data.qacc.copy(),
             "v_pelvis": v_pelvis, "act": self.sim.data.act.copy()}

    return state

  def get_observation(self):
    # Ignore eye qpos and qvel for now
    jnt_range = self.sim.model.jnt_range[self.independent_joints]

    qpos = self.sim.data.qpos[self.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5)*2
    qvel = self.sim.data.qvel[self.independent_joints].copy()
    qacc = self.sim.data.qacc[self.independent_joints].copy()

    act = (self.sim.data.act.copy() - 0.5)*2
    #act = self.sim.data.act.copy()
    fingertip = "hand_2distph"
    finger_position = self.sim.data.geom_xpos[self.model._geom_name2id[fingertip]]
    return np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position-self.target_origin, self.target_position,
                           np.array([self.target_radius]), act])

  def spawn_target(self, sample_target=True):

    # Sample a location
    if sample_target:
      distance = self.new_target_distance_threshold
      while distance <= self.new_target_distance_threshold:
        target_y = self.rng.uniform(*self.target_limits_y)
        target_z = self.rng.uniform(*self.target_limits_z)
        new_position = np.array([0, target_y, target_z])
        distance = np.linalg.norm(self.target_position - new_position)
      self.target_position = new_position
    else:
      self.target_position = np.zeros((3,))

    # Set location
    self.model.body_pos[self.model._body_name2id["target"]] = self.target_origin + self.target_position

    # Sample target radius
    if sample_target:
      self.target_radius = self.rng.uniform(*self.target_radius_limit)
    else:
      self.target_radius = 0.05

    # Set target radius
    self.model.geom_size[self.model._geom_name2id["target-sphere"]][0] = self.target_radius

    self.sim.forward()

  def reset(self, sample_target=True):

    self.sim.reset()
    self.steps_since_last_hit = 0

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
    self.spawn_target(sample_target=sample_target)

    # Do a forward so everything will be set
    self.sim.forward()

    return self.get_observation()

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
        self._get_viewer(mode).render(width, height)
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
