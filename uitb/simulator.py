import gym
from gym import spaces
import mujoco
import os
import numpy as np
import sys
import importlib
import shutil
import inspect
import pathlib
from ruamel.yaml import YAML

from .perception.base import Perception
from .utils.rendering import Camera, Context
from .utils.functions import output_path, parent_path


class Simulator(gym.Env):

  id = "uitb:simulator-v0"

  @staticmethod
  def get_class(modules, cls_str):
    # TODO check for incorrect module names etc
    src = __name__.split(".")[0]
    if "." in cls_str:
      mods = cls_str.split(".")
      modules += "." + ".".join(mods[:-1])
      cls_name = mods[-1]
    else:
      cls_name = cls_str
    module = importlib.import_module(src + "." + modules)
    return getattr(module, cls_name)

  @classmethod
  def build(cls, config):

    # Make sure required things are defined in config
    assert "simulation" in config, "Simulation specs (simulation) must be defined in config"
    assert "bm_model" in config["simulation"], "Biomechanical model (bm_model) must be defined in config"
    assert "task" in config["simulation"], "task (task) must be defined in config"

    assert "run_parameters" in config["simulation"], "Run parameters (run_parameters) must be defined in config"
    run_parameters = config["simulation"]["run_parameters"].copy()
    assert "action_sample_freq" in run_parameters, "Action sampling frequency (action_sample_freq) must be defined " \
                                                   "in run parameters"

    # Set simulator id
    config["id"] = cls.id

    # Save outputs to uitb/outputs if run folder is not defined
    if "run_folder" not in config:
      config["run_folder"] = os.path.join(output_path(), config["run_name"])

    # If 'package_name' is not defined use 'run_name' TODO check that package_name is suitable for a python module
    if "package_name" not in config:
      config["package_name"] = config["run_name"]

    # The name used in gym has a suffix -v0
    config["gym_name"] = "uitb:" + config["package_name"] + "-v0"

    # Initialise a simulator in the run folder
    cls.initialise(config["run_folder"], config["package_name"])

    # Load task class
    task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
    task_cls.clone(config["run_folder"], config["package_name"])
    simulation = task_cls.initialise(config["simulation"]["task"].get("kwargs", {}))

    # Load biomechanical model class
    bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
    bm_cls.clone(config["run_folder"], config["package_name"])
    bm_cls.insert(simulation)

    # Add perception modules
    perception_modules = {}
    for module_cfg in config["simulation"].get("perception_modules", []):
      module_cls = cls.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      module_cls.clone(config["run_folder"], config["package_name"])
      module_cls.insert(simulation, config, **module_kwargs)
      perception_modules[module_cls] = module_kwargs

    # Clone also RL library files so the package will be completely standalone
    rl_cls = cls.get_class("rl", config["rl"]["algorithm"])
    rl_cls.clone(config["run_folder"], config["package_name"])

    # TODO read the xml file directly from task.getroot() instead of writing it to a file first; need to input a dict
    #  of assets to mujoco.MjModel.from_xml_path
    simulation_file = os.path.join(config["run_folder"], config["package_name"], "simulation")
    with open(simulation_file+".xml", 'w') as file:
      simulation.write(file, encoding='unicode')

    # Load the mujoco model
    model = mujoco.MjModel.from_xml_path(simulation_file+".xml")

    # Initialise MjData
    data = mujoco.MjData(model)

    # Add dt to run parameters
    frame_skip = int(1 / (model.opt.timestep * run_parameters["action_sample_freq"]))
    run_parameters["dt"] = model.opt.timestep*frame_skip

    # Initialise a rendering context, required for e.g. some vision modules
    run_parameters["rendering_context"] = Context(model,
                                                  max_resolution=run_parameters.get("max_resolution", [1280, 960]))

    # Now initialise the actual classes; model and data are input to the inits so that stuff can be modified if needed
    # (e.g. move target to a specific position wrt to a body part)
    task = task_cls(model, data, **{**config["simulation"]["task"].get("kwargs", {}), **run_parameters})
    bm_model = bm_cls(model, data, **{**config["simulation"]["bm_model"].get("kwargs", {}), **run_parameters})
    perception = Perception(model, data, bm_model, perception_modules, run_parameters)

    # It would be nice to save an xml here in addition to saving a binary model file. But seems like there's only one
    # function to save an xml file: mujoco.mj_saveLastXML, which doesn't work here because we read the original
    # task and bm_model xml files. I've tried to read again the task_file with mujoco.MjModel.from_xml_path(task_file)
    # and then call mujoco.mj_saveLastXML(task_file, model) but it doesn't save the changes into the model (like
    # setting target-plane to 55cm in front of and 10cm to the right of the biomechanical model's shoulder)

    # Save the modified model as binary
    mujoco.mj_saveModel(model, simulation_file+".mjcf", None)

    # Save config
    yaml = YAML()
    with open(os.path.join(config["run_folder"], "config.yaml"), "w") as stream:
      yaml.dump(config, stream)

  @classmethod
  def initialise(cls, run_folder, package_name):

    # Create the folder
    dst = os.path.join(run_folder, package_name)
    os.makedirs(dst, exist_ok=True)

    # Copy simulator
    src = pathlib.Path(inspect.getfile(cls))
    shutil.copyfile(src, os.path.join(dst, src.name))

    # Create __init__.py with env registration
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from .simulator import Simulator\n\n")
      file.write("from gym.envs.registration import register\n")
      file.write("import pathlib\n\n")
      file.write("module_folder = pathlib.Path(__file__).parent\n")
      file.write("run_folder = module_folder.parent\n")
      file.write("kwargs = {'run_folder': run_folder}\n")
      file.write("register(id=f'{module_folder.stem}-v0', entry_point=f'{module_folder.stem}.simulator:Simulator', kwargs=kwargs)\n")

    # Copy utils
    shutil.copytree(os.path.join(parent_path(src), "utils"), os.path.join(run_folder, package_name, "utils"),
                    dirs_exist_ok=True)

  @classmethod
  def get(cls, run_folder, package_name=None, use_cloned=True):
    # TODO make sure simulator has been built before calling this method

    # If package_name is not given just find the folder with an __init__.py file
    if package_name is None:
      package_name = []
      for name in os.listdir(run_folder):
        if os.path.isdir(os.path.join(run_folder, name)):
          files = os.listdir(os.path.join(run_folder, name))
          if "__init__.py" in files:
            package_name.append(name)
      assert len(package_name) == 1, "Found zero or multiple packages"
      package_name = package_name[0]

    if use_cloned:
      # Make sure run_folder is in path
      if run_folder not in sys.path:
        sys.path.insert(0, run_folder)

      # Get Simulator class
      cls = getattr(importlib.import_module(package_name), "Simulator")
    else:
      pass

    # Return Simulator object
    return cls(run_folder)

  def __init__(self, run_folder, run_parameters=None):

    # Read
    yaml = YAML()
    with open(os.path.join(run_folder, "config.yaml"), "r") as stream:
      #self.config = dill.load(file)
      self.config = yaml.load(stream)

    # Get run parameters: these parameters can be used to override parameters used during training
    if run_parameters is None:
      run_parameters = self.config["simulation"]["run_parameters"].copy()

    # Load the mujoco model
    self.model = mujoco.MjModel.from_binary_path(os.path.join(run_folder, self.config["package_name"], "simulation.mjcf"))

    # Initialise data
    self.data = mujoco.MjData(self.model)

    # Get frame skip
    self.frame_skip = int(1 / (self.model.opt.timestep * run_parameters["action_sample_freq"]))

    # Add dt to run parameters so it's easier to pass to models
    run_parameters["dt"] = self.dt

    # Initialise a rendering context, required for e.g. some vision modules
    run_parameters["rendering_context"] = Context(self.model,
                                                  max_resolution=run_parameters.get("max_resolution", [1280, 960]))

    # Initialise task object
    task_cls = self.get_class("tasks", self.config["simulation"]["task"]["cls"])
    self.task = task_cls(self.model, self.data,
                         **{**self.config["simulation"]["task"].get("kwargs", {}), **run_parameters})

    # Initialise bm_model object
    bm_cls = self.get_class("bm_models", self.config["simulation"]["bm_model"]["cls"])
    self.bm_model = bm_cls(self.model, self.data,
                           **{**self.config["simulation"]["bm_model"].get("kwargs", {}), **run_parameters})

    # Initialise perception object
    perception_modules = {}
    for module_cfg in self.config["simulation"].get("perception_modules", []):
      module_cls = self.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      perception_modules[module_cls] = module_kwargs
    self.perception = Perception(self.model, self.data, self.bm_model, perception_modules, run_parameters)

    # Set action space TODO for now we assume all actuators have control signals between [-1, 1]
    self.action_space = self.initialise_action_space()

    # Set observation space
    self.observation_space = self.initialise_observation_space()

    # Collect some episode statistics
    self._episode_statistics = {"length (seconds)": 0, "length (steps)": 0, "reward": 0}

    # Initialise viewer
    self.camera = Camera(run_parameters["rendering_context"], self.model, self.data, camera_id='for_testing')

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

    # Update bm model (e.g. update constraints)
    self.bm_model.update(self.model, self.data)

    # Update perception modules
    self.perception.update(self.model, self.data)

    # Update environment
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
    self.bm_model.reset(self.model, self.data)
    self.perception.reset(self.model, self.data)
    self.task.reset(self.model, self.data)

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
