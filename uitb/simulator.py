import gym
from gym import spaces
import mujoco
import os
import numpy as np
import dill
import sys
import importlib

try:
  from gym.envs.mujoco.mujoco_rendering import Viewer, RenderContextOffscreen
except ImportError:
  from uitb.mujoco_rendering import Viewer, RenderContextOffscreen

from uitb.perception.base import Perception

def get_clone_class(src_class):

  module = src_class.__module__.split(".")
  module[0] = "simulator"
  module = importlib.import_module(".".join(module))
  return getattr(module, src_class.__name__)


class Simulator(gym.Env):

  @staticmethod
  def build(config):

    # Make sure required things are defined in config
    assert "simulator" in config, "Simulator (simulator) must be defined in config"
    assert "bm_model" in config["simulator"], "Biomechanical model (bm_model) must be defined in config"
    assert "task" in config["simulator"], "task (task) must be defined in config"

    assert "run_parameters" in config, "Run parameters (run_parameters) must be defined in config"
    run_parameters = config["run_parameters"]
    assert "action_sample_freq" in run_parameters, "Action sampling frequency (action_sample_freq) must be defined " \
                                                   "in run parameters"

    # Create a folder for the simulator
    os.makedirs(config["run_folder"], exist_ok=True)

    # Load the environment
    config["simulator"]["task"].clone(config["run_folder"])
    task = config["simulator"]["task"].initialise_task(config)

    # Load biomechanical model
    config["simulator"]["bm_model"].clone(config["run_folder"])
    config["simulator"]["bm_model"].insert(task, config)

    # Add perception modules
    for module, kwargs in config["simulator"].get("perception_modules", {}).items():
      module.clone(config["run_folder"])
      module.insert(task, config, **kwargs)

    # We need to save the mujoco xml file and then load it, because when reading the xml string directly we would need
    # to copy assets to /tmp for which we might not have permissions
    task_file = os.path.join(config["run_folder"], "simulator", "task.xml")
    with open(task_file, 'w') as file:
      task.write(file, encoding='unicode')

    # Update classes in config
    sys.path.insert(0, config["run_folder"])
    config["simulator"]["task"] = get_clone_class(config["simulator"]["task"])
    config["simulator"]["bm_model"] = get_clone_class(config["simulator"]["bm_model"])
    perception_modules = {}
    for module, kwargs in config["simulator"].get("perception_modules", {}).items():
      perception_modules[get_clone_class(module)] = kwargs
    config["simulator"]["perception_modules"] = perception_modules

    # Load the mujoco model
    model = mujoco.MjModel.from_xml_path(task_file)

    # Initialise MjData
    frame_skip = int(1 / (model.opt.timestep * run_parameters["action_sample_freq"]))
    data = mujoco.MjData(model) #, nsubsteps=frame_skip)

    # Add an rng to run parameters
    run_parameters["rng"] = np.random.default_rng(run_parameters.get("random_seed", None))

    # Add dt to run parameters
    run_parameters["dt"] = model.opt.timestep*frame_skip

    # Now initialise the actual classes; sim is input to the inits so that stuff can be modified if needed
    # (e.g. move target to a specific position wrt to a body part)
    task = config["simulator"]["task"](model, data, **{**config["simulator"].get("task_kwargs", {}),
                                               **run_parameters})
    bm_model = config["simulator"]["bm_model"](model, data, **{**config["simulator"].get("bm_model_kwargs", {}),
                                                       **run_parameters})
    perception = Perception(model, data, bm_model, config["simulator"]["perception_modules"], run_parameters)

    # Could save above classes here (with dill) and load them later in Simulator.__init__() if they take a long time
    # to initialise

    # TODO There's an interesting bug here in mujoco_py. Because we load the original bm_model in the BaseBMModel
    #  initialisator it somehow replaces the model we load above (with mujoco_py.load_model_from_path), and messes up
    #  the save operation below. This happens even though sim.model still refers to the model we load above. To fix
    #  this bug we have to load the model from task_file again. It's almost as there could be only one model at a time
    #  loaded into mujoco_py. Need to check later that the modifications we put into sim.model in the initialisators
    #  of env and bm_model are really saved
    #mujoco_py.load_model_from_path(task_file)
    model = mujoco.MjModel.from_xml_path(task_file)

    # Save updated mujoco xml file
    # with open(task_file, 'w') as file:
    #   sim.save(file, 'xml', True)
    mujoco.mj_saveLastXML(task_file, model)

    # Save configs
    with open(os.path.join(config["run_folder"], "config.dill"), 'wb') as file:
      dill.dump(config, file)

  def __init__(self, run_folder):

    # Add run folder to python path if not there already
    if run_folder not in sys.path:
      sys.path.insert(0, run_folder)

    self.id = "uitb:simulator-v0"

    # Read config
    with open(os.path.join(run_folder, "config.dill"), "rb") as file:
      self.config = dill.load(file)

    #with open(os.path.join(self.config["run_folder"], "bm_model.dill"), 'rb') as file:
    #  self.bm_model = dill.load(file)

    #with open(os.path.join(self.config["run_folder"], "task.dill"), 'rb') as file:
    #  self.task = dill.load(file)

    #with open(os.path.join(self.config["run_folder"], "perception.dill"), 'rb') as file:
    #  self.perception = dill.load(file)

    # Get run parameters
    run_parameters = self.config["run_parameters"]

    # Add rng to run parameters
    run_parameters["rng"] = np.random.default_rng(run_parameters.get("random_seed", None))

    # Load the mujoco model
    #self.model = mujoco_py.load_model_from_path(os.path.join(run_folder, "simulator", "task.xml"))
    self.model = mujoco.MjModel.from_xml_path(os.path.join(run_folder, "simulator", "task.xml"))

    # Initialise data
    self.frame_skip = int(1 / (self.model.opt.timestep * run_parameters["action_sample_freq"]))
    self.data = mujoco.MjData(self.model) #, nsubsteps=self.frame_skip)

    # Add dt to run parameters
    run_parameters["dt"] = self.dt()

    # Initialise classes
    self.task = self.config["simulator"]["task"](self.model, self.data, **{
      **self.config["simulator"].get("task_kwargs", {}), **run_parameters})
    self.bm_model = self.config["simulator"]["bm_model"](self.model, self.data, **{
      **self.config["simulator"].get("bm_model_kwargs", {}), **run_parameters})
    self.perception = Perception(self.model, self.data, self.bm_model, self.config["simulator"]["perception_modules"], run_parameters)

    # Set action space TODO for now we assume all actuators have control signals between [-1, 1]
    self.action_space = self.initialise_action_space()

    # Set observation space
    self.observation_space = self.initialise_observation_space()

    # Collect some episode statistics
    self._episode_statistics = {"length (seconds)": 0, "length (steps)": 0, "reward": 0}

    # Set camera stuff, self._viewers needs to be initialised before self.get_observation() is called
    self.viewer = None
    self._viewers = {}
    self.metadata = {
      'render.modes': ['human', 'rgb_array', 'depth_array'],
      'video.frames_per_second': int(np.round(1.0 / (self.model.opt.timestep * self.frame_skip))),
      "imagesize": (1600, 1280)
    }

    # Get callbacks
    #self.callbacks = {callback.name: callback for callback in run_parameters.get('callbacks', [])}

  def initialise_action_space(self):
    num_actuators = self.bm_model.nu + self.perception.nu
    actuator_limits = np.ones((num_actuators,2)) * np.array([-1.0, 1.0])
    self.action_space = spaces.Box(low=np.float32(actuator_limits[:, 0]), high=np.float32(actuator_limits[:, 1]))
    return self.action_space

  def initialise_observation_space(self):
    observation = self.get_observation()
    obs_dict = dict()
    for module in self.perception.perception_modules:
      obs_dict[module.modality] = spaces.Box(dtype=np.float32, **module.get_observation_space_params())
    if "stateful_information" in observation:
      obs_dict["stateful_information"] = spaces.Box(dtype=np.float32,
                                                    **self.task.get_stateful_information_space_params())

    return spaces.Dict(obs_dict)

  def step(self, action):

    # Set control for the bm model
    self.bm_model.set_ctrl(self.model, self.data, action[:self.bm_model.nu])

    # Set control for perception modules (e.g. eye movements)
    self.perception.set_ctrl(self.model, self.data, action[self.bm_model.nu:])

    # Advance the simulation
    mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    # Find out what happened
    reward, finished, info = self.task.update(self.model, self.data)

    # Add an effort cost to reward
    #reward -= self.bm_model.effort_term.get(self)

    # Get observation
    obs = self.get_observation()

    return obs, reward, finished, info

  def get_observation(self):

    # Get observation from perception
    observation = self.perception.get_observation(self.model, self.data)

    # Add any stateful information that is required
    stateful_information = self.task.get_stateful_information(self.model, self.data)
    if stateful_information is not None:
      observation["stateful_information"] = stateful_information

    return observation

  def reset(self):

    # Reset sim
    mujoco.mj_resetData(self.model, self.data)

    # Reset all models
    self.bm_model.reset(self.sim)
    self.perception.reset(self.sim)
    self.task.reset(self.sim)

    # Do a forward so everything will be set
    mujoco.mj_forward(self.model, self.data)

    return self.get_observation()

  #def callback(self, callback_name, num_timesteps):
  #  self.callbacks[callback_name].update(num_timesteps)

  @property
  def dt(self):
    return self.model.opt.timestep * self.frame_skip


  def get_state(self):
    state = {"step": self.steps, "timestep": self.data.time,
             "qpos": self.data.qpos[self.independent_joints].copy(),
             "qvel": self.data.qvel[self.independent_joints].copy(),
             "qacc": self.data.qacc[self.independent_joints].copy(),
             "act": self.data.act.copy(),
             "fingertip_xpos": self.data.get_geom_xpos(self.fingertip).copy(),
             "fingertip_xmat": self.data.get_geom_xmat(self.fingertip).copy(),
             "fingertip_xvelp": self.data.get_geom_xvelp(self.fingertip).copy(),
             "fingertip_xvelr": self.data.get_geom_xvelr(self.fingertip).copy(),
             "termination": False}
    return state

  # def render(self, width=1280, height=800, camera_id=None, camera_name=None):
  #
  #   # Get rgb and depth arrays
  #   # TODO: compare to https://github.com/rodrigodelazcano/gym/blob/906789e0fd5b11e3c7979065e091a1abc00d1b35/gym/envs/mujoco/mujoco_rendering.py
  #   #  (def render() and def read_pixels())
  #
  #   rect = mujoco.MjrRect(left=0, bottom=0, width=width, height=height)
  #
  #   # Sometimes buffers are too small.
  #   if width > self.con.offWidth or height > self.con.offHeight:
  #     new_width = max(width, self.model.vis.global_.offwidth)
  #     new_height = max(height, self.model.vis.global_.offheight)
  #     self.update_offscreen_size(new_width, new_height)
  #
  #   if camera_name is not None:
  #     self.camera.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
  #   elif camera_id is not None:
  #     if camera_id == -1:
  #       self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  #     else:
  #       self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
  #     self.cam.fixedcamid = camera_id
  #
  #   mujoco.mjv_updateScene(
  #     self.model, self.data, mujoco.MjvOption(), None,
  #     self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
  #   mujoco.mjr_render(self.viewport, self.scene, self.context)
  #
  #   mujoco.mjr_rectangle(self.viewport, 0, 0, 0, 1)
  #   rgb_arr = np.zeros(3 * self.viewport.width * self.viewport.height, dtype=np.uint8)
  #   depth_arr = np.zeros(self.viewport.width * self.viewport.height, dtype=np.float32)
  #   mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.context)
  #   rgb = rgb_arr.reshape(self.viewport.height, self.viewport.width, 3)
  #   depth = depth_arr.reshape(self.viewport.height, self.viewport.width)
  #
  #   # Normalise
  #   #depth = render[1]
  #   depth = np.flipud((depth - 0.5) * 2)
  #   #rgb = render[0]
  #   rgb = np.flipud((rgb / 255.0 - 0.5) * 2)
  #
  #   return np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), [2, 0, 1])

  def render(self, mode='human', width=1280, height=800, camera_id=None, camera_name=None):

    if mode == 'rgb_array' or mode == 'depth_array':
        if camera_id is not None and camera_name is not None:
            raise ValueError("Both `camera_id` and `camera_name` cannot be"
                             " specified at the same time.")

        no_camera_specified = camera_name is None and camera_id is None
        if no_camera_specified:
            camera_name = 'track'

        if camera_id is None:
          camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

          self._get_viewer(mode).render(width, height, camera_id=camera_id)

    if mode == 'rgb_array':
        data = self._get_viewer(mode).read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]
    elif mode == 'depth_array':
        self._get_viewer(mode).render(width, height)
        # Extract depth part of the read_pixels() tuple
        data = self._get_viewer(mode).read_pixels(width, height, depth=True)[1]
        # original image is upside-down, so flip it
        return data[::-1, :]
    elif mode == 'human':
        self._get_viewer(mode).render()

  def _get_viewer(self, mode, width=1280, height=800):
    self.viewer = self._viewers.get(mode)
    if self.viewer is None:
      if mode == 'human':
        self.viewer = Viewer(self.model, self.data)
      elif mode == 'rgb_array' or mode == 'depth_array':
        self.viewer = RenderContextOffscreen(width, height, self.model, self.data)
      self._viewers[mode] = self.viewer
    return self.viewer