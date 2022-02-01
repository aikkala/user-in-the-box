import gym
from gym import spaces
import mujoco_py
import numpy as np
import os
import pathlib
from abc import ABC, abstractmethod


class BaseModel(ABC, gym.Env):

  def __init__(self, **kwargs):

    # Get project path
    self.project_path = pathlib.Path(__file__).parent.absolute()

    # Model file
    xml_file = "models/mobl_arms_muscles.xml"

    # Set action sampling
    self.action_sample_freq = kwargs.get('action_sample_freq', 10)
    self.timestep = 0.002
    self.frame_skip = int(1/(self.timestep*self.action_sample_freq))

    # Use early termination if target is not hit in time
    self.steps_since_last_hit = 0
    self.max_steps_without_hit = self.action_sample_freq*4
    self.steps = 0

    # Define a maximum number of trials (if needed for e.g. evaluation / visualisation)
    self.trial_idx = 0
    self.max_trials = kwargs.get('max_trials', 10)

    # Dwelling based selection -- fingertip needs to be inside target for some time
    self.steps_inside_target = 0
    self.dwell_threshold = int(0.3*self.action_sample_freq)

    # Radius limits for target
    self.target_radius_limit = kwargs.get('target_radius_limit', np.array([0.05, 0.15]))

    # Minimum distance to new spawned targets is twice the max target radius limit
    self.new_target_distance_threshold = 2*self.target_radius_limit[1]

    # RNG in case we need it
    self.rng = np.random.default_rng()

    # Initialise model and sim
    self.model = mujoco_py.load_model_from_path(os.path.join(self.project_path, xml_file))
    self.sim = mujoco_py.MjSim(self.model, nsubsteps=self.frame_skip)

    # Do a forward step so stuff like geom and body positions are calculated
    self.sim.forward()

    # Define plane where targets will be spawned: 0.5m in front of shoulder, or the "humphant" body. Note that this
    # body is not fixed but moves with the shoulder, so the model is assumed to be in initial position
    #self.target_origin = np.array([0.5, 0.0, 0.8])
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

    # Fix gaze towards the plane
    #self.oculomotor_camera_idx = self.model._camera_name2id["oculomotor"]
    #self.model.cam_mode[self.oculomotor_camera_idx] = 3
    #self.model.cam_targetbodyid[self.oculomotor_camera_idx] = self.target_plane_body_idx

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
    motors_limits = np.ones((self.nmotors,2)) * np.array([-1.0, 1.0])
    muscles_limits = np.ones((self.nmuscles,2)) * np.array([-1.0, 1.0])
    self.action_space = spaces.Box(low=np.float32(muscles_limits[:, 0]), high=np.float32(muscles_limits[:, 1]))

    # Fingertip is tracked for e.g. reward calculation and logging
    self.fingertip = "hand_2distph"

    # Define a cost function
    self.cost_function = kwargs.get('cost_function', "")

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

  def step(self, action):

    info = {}

    # Set motor and muscle control
    # Don't do anything with eyes for now
    self.sim.data.ctrl[2:] = np.clip(self.sim.data.act[:] + action, 0, 1)

    finished = False
    info["termination"] = False
    try:
      self.sim.step()
    except mujoco_py.builder.MujocoException:
      finished = True
      info["termination"] = "MujocoException"

    # Get finger position
    finger_position = self.sim.data.get_geom_xpos(self.fingertip)

    # Distance to target
    dist = np.linalg.norm(self.target_position - (finger_position - self.target_origin))

    if dist < self.target_radius and self.steps_inside_target>=self.dwell_threshold:

      # Spawn a new target
      self.spawn_target()

      # Reset counter, add hit bonus to reward
      self.steps_since_last_hit = 0
      self.steps_inside_target = 0
      reward = 2
      info["target_hit"] = True
      info["inside_target"] = True
      self.trial_idx += 1

    else:

      # Estimate reward
      reward = np.exp(-dist*10)/10
      info["target_hit"] = False

      # Check if fingertip is inside target
      if dist < self.target_radius:
        self.steps_inside_target += 1
        info["inside_target"] = True
      else:
        self.steps_inside_target = 0
        info["inside_target"] = False

      # Check if time limit has been reached
      self.steps_since_last_hit += 1
      if self.steps_since_last_hit >= self.max_steps_without_hit:
        finished = True
        info["termination"] = "time_limit_reached"

      if self.max_trials is not None and self.trial_idx >= self.max_trials:
        finished = True
        info["termination"] = "max_trials_reached"

      # Increment steps
      self.steps += 1

      # Add an effort cost to reward
      if self.cost_function == "neural_effort":
        reward -= 1e-4 * np.sum(self.sim.data.ctrl**2)
      elif self.cost_function == "composite":
        angle_acceleration = np.sum(self.sim.data.qacc[self.independent_joints]**2)
        energy = np.sum(self.sim.data.qacc[self.independent_joints]**2 * self.sim.data.qfrc_unc[self.independent_joints]**2)
        reward -= 1e-7 * (energy + 0.05*angle_acceleration)

    return self.get_observation(), reward, finished, info

  @abstractmethod
  def get_observation(self):
    pass

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
    self.steps_inside_target = 0
    self.trial_idx = 0

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

  def grab_image(self, height, width):

    # Make sure estimate is not in the image
    self.model.geom_rgba[self.model._geom_name2id["target-sphere-estimate"]][-1] = 0

    rendered = self.sim.render(height=height, width=width, camera_name='oculomotor', depth=True)
    rgb = ((np.flipud(rendered[0]) / 255.0) - 0.5) * 2
    depth = (np.flipud(rendered[1]) - 0.5) * 2
    #return np.expand_dims(np.flipud(depth), 0)
    #return np.concatenate([rgb.transpose([2, 0, 1]), np.expand_dims(depth, 0)])
    return np.concatenate([np.expand_dims(rgb[:, :, 1], 0), np.expand_dims(depth, 0)])

  def grab_proprioception(self):

    # Ignore eye qpos and qvel for now
    jnt_range = self.sim.model.jnt_range[self.independent_joints]

    qpos = self.sim.data.qpos[self.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5) * 2
    qvel = self.sim.data.qvel[self.independent_joints].copy()
    qacc = self.sim.data.qacc[self.independent_joints].copy()

    finger_position = self.sim.data.get_geom_xpos(self.fingertip).copy()
    return np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position])
    #return np.concatenate([qpos[2:], qvel[2:], qacc[2:]])

  def grab_target(self):
    # Use self.target_position for normalised position around self.target_origin
    # Make target radius zero mean using known limits
    normalised_radius = self.target_radius - (self.target_radius_limit[1]-self.target_radius_limit[0])
    return np.concatenate([self.target_position[1:].copy(), np.array([normalised_radius])])

  def get_state(self):
    state = {"step": self.steps, "timestep": self.sim.data.time,
             "qpos": self.sim.data.qpos[self.independent_joints].copy(),
             "qvel": self.sim.data.qvel[self.independent_joints].copy(),
             "qacc": self.sim.data.qacc[self.independent_joints].copy(),
             "act": self.sim.data.act.copy(),
             "fingertip_xpos": self.sim.data.get_geom_xpos(self.fingertip).copy(),
             "fingertip_xmat": self.sim.data.get_geom_xmat(self.fingertip).copy(),
             "fingertip_xvelp": self.sim.data.get_geom_xvelp(self.fingertip).copy(),
             "fingertip_xvelr": self.sim.data.get_geom_xvelr(self.fingertip).copy(),
             "target_position": self.target_origin.copy()+self.target_position.copy(),
             "target_radius": self.target_radius}
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
