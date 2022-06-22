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
from datetime import datetime
import copy

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
    run_folder = os.path.join(output_path(), config["run_name"])

    # If 'package_name' is not defined use 'run_name' TODO check that package_name is suitable for a python package
    if "package_name" not in config:
      config["package_name"] = config["run_name"]

    # The name used in gym has a suffix -v0
    config["gym_name"] = "uitb:" + config["package_name"] + "-v0"

    # Create a simulator in the run folder
    cls._clone(run_folder, config["package_name"])

    # Load task class
    task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
    task_cls.clone(run_folder, config["package_name"])
    simulation = task_cls.initialise(config["simulation"]["task"].get("kwargs", {}))

    # Load biomechanical model class
    bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
    bm_cls.clone(run_folder, config["package_name"])
    bm_cls.insert(simulation)

    # Add perception modules
    for module_cfg in config["simulation"].get("perception_modules", []):
      module_cls = cls.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      module_cls.clone(run_folder, config["package_name"])
      module_cls.insert(simulation, **module_kwargs)

    # Clone also RL library files so the package will be completely standalone
    rl_cls = cls.get_class("rl", config["rl"]["algorithm"])
    rl_cls.clone(run_folder, config["package_name"])

    # TODO read the xml file directly from task.getroot() instead of writing it to a file first; need to input a dict
    #  of assets to mujoco.MjModel.from_xml_path
    simulation_file = os.path.join(run_folder, config["package_name"], "simulation")
    with open(simulation_file+".xml", 'w') as file:
      simulation.write(file, encoding='unicode')

    # Initialise the simulator
    model, _, _, _, _ = \
      cls._initialise(config, run_folder, run_parameters)

    # Now that simulator has been initialised, everything should be set. Now we want to save the xml file again, but
    # mujoco only is able to save the latest loaded xml file (which is either the task or bm model xml files which are
    # are read in their __init__ functions), hence we need to read the file we've generated again before saving the
    # modified model
    mujoco.MjModel.from_xml_path(simulation_file+".xml")
    mujoco.mj_saveLastXML(simulation_file+".xml", model)

    # Save the modified model also as binary for faster loading
    mujoco.mj_saveModel(model, simulation_file+".mjcf", None)

    # Input built time into config
    config["built"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save config
    yaml = YAML()
    with open(os.path.join(run_folder, "config.yaml"), "w") as stream:
      yaml.dump(config, stream)

    return run_folder

  @classmethod
  def _clone(cls, run_folder, package_name):

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
  def _initialise(cls, config, run_folder, run_parameters):

    # Get task class and kwargs
    task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
    task_kwargs = config["simulation"]["task"].get("kwargs", {})

    # Get bm class and kwargs
    bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
    bm_kwargs = config["simulation"]["bm_model"].get("kwargs", {})

    # Initialise perception modules
    perception_modules = {}
    for module_cfg in config["simulation"].get("perception_modules", []):
      module_cls = cls.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      perception_modules[module_cls] = module_kwargs

    # Get xml file
    simulation_file = os.path.join(run_folder, config["package_name"], "simulation.xml")

    # Load the mujoco model
    model = mujoco.MjModel.from_xml_path(simulation_file)

    # Initialise MjData
    data = mujoco.MjData(model)

    # Add frame skip and dt to run parameters
    run_parameters["frame_skip"] = int(1 / (model.opt.timestep * run_parameters["action_sample_freq"]))
    run_parameters["dt"] = model.opt.timestep*run_parameters["frame_skip"]

    # Initialise a rendering context, required for e.g. some vision modules
    run_parameters["rendering_context"] = Context(model,
                                                  max_resolution=run_parameters.get("max_resolution", [1280, 960]))

    # Now initialise the actual classes; model and data are input to the inits so that stuff can be modified if needed
    # (e.g. move target to a specific position wrt to a body part)
    task = task_cls(model, data, **{**task_kwargs, **run_parameters})
    bm_model = bm_cls(model, data, **{**bm_kwargs, **run_parameters})
    perception = Perception(model, data, bm_model, perception_modules, run_parameters)

    return model, data, task, bm_model, perception

  @classmethod
  def get(cls, run_folder, run_parameters=None, use_cloned=True):

    # Read config file
    config_file = os.path.join(run_folder, "config.yaml")
    yaml = YAML()
    try:
      with open(config_file, "r") as stream:
        config = yaml.load(stream)
    except:
      raise FileNotFoundError(f"Could not open file {config_file}")

    # Make sure the simulator has been built
    if "built" not in config:
      raise RuntimeError("Simulator has not been built")

    if use_cloned:
      # Make sure run_folder is in path
      if run_folder not in sys.path:
        sys.path.insert(0, run_folder)

      # Get Simulator class
      gen_cls = getattr(importlib.import_module(config["package_name"]), "Simulator")
    else:
      gen_cls = cls

    # Return Simulator object
    return gen_cls(run_folder, run_parameters=run_parameters)

  def __init__(self, run_folder, run_parameters=None):

    # Read configs
    self._run_folder = run_folder
    yaml = YAML()
    with open(os.path.join(self._run_folder, "config.yaml"), "r") as stream:
      self._config = yaml.load(stream)

    # Get run parameters: these parameters can be used to override parameters used during training
    self._run_parameters = self._config["simulation"]["run_parameters"].copy()
    self._run_parameters.update(run_parameters or {})

    # Initialise simulation
    self._model, self._data, self.task, self.bm_model, self.perception = \
      self._initialise(self._config, self._run_folder, self._run_parameters)

    # Set action space TODO for now we assume all actuators have control signals between [-1, 1]
    self.action_space = self._initialise_action_space()

    # Set observation space
    self.observation_space = self._initialise_observation_space()

    # Collect some episode statistics
    self._episode_statistics = {"length (seconds)": 0, "length (steps)": 0, "reward": 0}

    # Initialise viewer
    self._camera = Camera(self._run_parameters["rendering_context"], self._model, self._data, camera_id='for_testing',
                         dt=self._run_parameters["dt"])

    # Get callbacks
    #self.callbacks = {callback.name: callback for callback in run_parameters.get('callbacks', [])}

  def _initialise_action_space(self):
    num_actuators = self.bm_model.nu + self.perception.nu
    actuator_limits = np.ones((num_actuators,2)) * np.array([-1.0, 1.0])
    return spaces.Box(low=np.float32(actuator_limits[:, 0]), high=np.float32(actuator_limits[:, 1]))

  def _initialise_observation_space(self):
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
    self.bm_model.set_ctrl(self._model, self._data, action[:self.bm_model.nu])

    # Set control for perception modules (e.g. eye movements)
    self.perception.set_ctrl(self._model, self._data, action[self.bm_model.nu:])

    # Advance the simulation
    mujoco.mj_step(self._model, self._data, nstep=self._run_parameters["frame_skip"])

    # Update bm model (e.g. update constraints); updates also effort model
    self.bm_model.update(self._model, self._data)

    # Update perception modules
    self.perception.update(self._model, self._data)

    # Update environment
    reward, finished, info = self.task.update(self._model, self._data)

    # Add an effort cost to reward
    reward -= self.bm_model.get_effort_cost(self._model, self._data)

    # Get observation
    obs = self.get_observation(info)

    return obs, reward, finished, info

  def get_observation(self, info=None):

    # Get observation from perception
    observation = self.perception.get_observation(self._model, self._data, info)

    # Add any stateful information that is required
    stateful_information = self.task.get_stateful_information(self._model, self._data)
    if stateful_information is not None:
      observation["stateful_information"] = stateful_information

    return observation

  def reset(self):

    # Reset sim
    mujoco.mj_resetData(self._model, self._data)

    # Reset all models
    self.bm_model.reset(self._model, self._data)
    self.perception.reset(self._model, self._data)
    info = self.task.reset(self._model, self._data)

    # Do a forward so everything will be set
    mujoco.mj_forward(self._model, self._data)

    return self.get_observation(info)

  #def callback(self, callback_name, num_timesteps):
  #  self.callbacks[callback_name].update(num_timesteps)

  def get_config(self):
    return copy.deepcopy(self._config)

  def get_run_parameters(self):
    # Context cannot be deep copied
    exclude = {"rendering_context"}
    run_params = {k: copy.deepcopy(self._run_parameters[k]) for k in self._run_parameters.keys() - exclude}
    run_params["rendering_context"] = self._run_parameters["rendering_context"]
    return run_params

  def get_run_folder(self):
    return self._run_folder

  def get_state(self):
    # TODO fix this
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
