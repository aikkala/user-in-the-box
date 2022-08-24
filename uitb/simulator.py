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
from datetime import datetime
import copy

from .perception.base import Perception
from .utils.rendering import Camera, Context
from .utils.functions import output_path, parent_path, is_suitable_package_name, parse_yaml, write_yaml


class Simulator(gym.Env):
  """
  The Simulator class contains functionality to build a standalone Python package from a config file. The built package
   integrates a biomechanical model, a task model, and a perception model into one simulator that implements a gym.Env
   interface.
  """

  # May be useful for later, the three digit number suffix is of format X.Y.Z where X is a major version.
  id = "uitb:simulator-v100"

  @classmethod
  def get_class(cls, *args):
    """ Returns a class from given strings. The last element in args should contain the class name. """
    # TODO check for incorrect module names etc
    modules = ".".join(args[:-1])
    if "." in args[-1]:
      splitted = args[-1].split(".")
      if modules == "":
        modules = ".".join(splitted[:-1])
      else:
        modules += "." + ".".join(splitted[:-1])
      cls_name = splitted[-1]
    else:
      cls_name = args[-1]
    module = cls.get_module(modules)
    return getattr(module, cls_name)

  @classmethod
  def get_module(cls, *args):
    """ Returns a module from given strings. """
    src = __name__.split(".")[0]
    modules = ".".join(args)
    return importlib.import_module(src + "." + modules)

  @classmethod
  def build(cls, config):
    """ Builds a simulator based on a config. The input 'config' may be a dict (parsed from YAML) or path to a YAML file

    Args:
      config:
        - A dict containing configuration information. See example configs in folder uitb/configs/
        - A path to a config file
    """

    # If config is a path to the config file, parse it first
    if isinstance(config, str):
      if not os.path.isfile(config):
        raise FileNotFoundError(f"Given config file {config} does not exist")
      config = parse_yaml(config)

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

    # Save generated simulators to uitb/simulators
    simulator_folder = os.path.join(output_path(), config["simulator_name"])

    # If 'package_name' is not defined use 'simulator_name'
    if "package_name" not in config:
      config["package_name"] = config["simulator_name"]
    if not is_suitable_package_name(config["package_name"]):
      raise NameError("Package name defined in the config file (either through 'package_name' or 'simulator_name') is "
                      "not a suitable Python package name. Use only lower-case letters and underscores instead of "
                      "spaces, and the name cannot start with a number.")

    # The name used in gym has a suffix -v0
    config["gym_name"] = "uitb:" + config["package_name"] + "-v0"

    # Create a simulator in the simulator folder
    cls._clone(simulator_folder, config["package_name"])

    # Load task class
    task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
    task_cls.clone(simulator_folder, config["package_name"])
    simulation = task_cls.initialise(config["simulation"]["task"].get("kwargs", {}))

    # Load biomechanical model class
    bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
    bm_cls.clone(simulator_folder, config["package_name"])
    bm_cls.insert(simulation)

    # Add perception modules
    for module_cfg in config["simulation"].get("perception_modules", []):
      module_cls = cls.get_class("perception", module_cfg["cls"])
      module_kwargs = module_cfg.get("kwargs", {})
      module_cls.clone(simulator_folder, config["package_name"])
      module_cls.insert(simulation, **module_kwargs)

    # Clone also RL library files so the package will be completely standalone
    rl_cls = cls.get_class("rl", config["rl"]["algorithm"])
    rl_cls.clone(simulator_folder, config["package_name"])

    # TODO read the xml file directly from task.getroot() instead of writing it to a file first; need to input a dict
    #  of assets to mujoco.MjModel.from_xml_path
    simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")
    with open(simulation_file+".xml", 'w') as file:
      simulation.write(file, encoding='unicode')

    # Initialise the simulator
    model, _, _, _, _, _ = \
      cls._initialise(config, simulator_folder, run_parameters)

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
    write_yaml(config, os.path.join(simulator_folder, "config.yaml"))

    return simulator_folder

  @classmethod
  def _clone(cls, simulator_folder, package_name):
    """ Create a folder for the simulator being built, and copy or create relevant files.

    Args:
       simulator_folder: Location of the simulator.
       package_name: Name of the simulator (which is a python package).
    """

    # Create the folder
    dst = os.path.join(simulator_folder, package_name)
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
      file.write("simulator_folder = module_folder.parent\n")
      file.write("kwargs = {'simulator_folder': simulator_folder}\n")
      file.write("register(id=f'{module_folder.stem}-v0', entry_point=f'{module_folder.stem}.simulator:Simulator', kwargs=kwargs)\n")

    # Copy utils
    shutil.copytree(os.path.join(parent_path(src), "utils"), os.path.join(simulator_folder, package_name, "utils"),
                    dirs_exist_ok=True)

  @classmethod
  def _initialise(cls, config, simulator_folder, run_parameters):
    """ Initialise a simulator -- i.e., create a MjModel, MjData, and initialise all necessary variables.

    Args:
        config: A config dict.
        simulator_folder: Location of the simulator.
        run_parameters: Important run time variables that may also be used to override parameters.
    """

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
    simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation.xml")

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

    # Initialise callbacks
    callbacks = {}
    for cb in run_parameters.get("callbacks", []):
      callbacks[cb["name"]] = cls.get_class(cb["cls"])(cb["name"], **cb["kwargs"])

    # Now initialise the actual classes; model and data are input to the inits so that stuff can be modified if needed
    # (e.g. move target to a specific position wrt to a body part)
    task = task_cls(model, data, **{**task_kwargs, **callbacks, **run_parameters})
    bm_model = bm_cls(model, data, **{**bm_kwargs, **callbacks, **run_parameters})
    perception = Perception(model, data, bm_model, perception_modules, {**callbacks, **run_parameters})

    return model, data, task, bm_model, perception, callbacks

  @classmethod
  def get(cls, simulator_folder, run_parameters=None, use_cloned=True):
    """ Returns a Simulator that is located in given folder.

    Args:
      simulator_folder: Location of the simulator.
      run_parameters: Can be used to override parameters.
      use_cloned: Can be useful for debugging. Set to False to use original files instead of the ones that have been
        cloned/copied during building phase.
    """

    # Read config file
    config_file = os.path.join(simulator_folder, "config.yaml")
    try:
      config = parse_yaml(config_file)
    except:
      raise FileNotFoundError(f"Could not open file {config_file}")

    # Make sure the simulator has been built
    if "built" not in config:
      raise RuntimeError("Simulator has not been built")

    if use_cloned:
      # Make sure simulator_folder is in path
      if simulator_folder not in sys.path:
        sys.path.insert(0, simulator_folder)

      # Get Simulator class
      gen_cls = getattr(importlib.import_module(config["package_name"]), "Simulator")
    else:
      gen_cls = cls

    # Return Simulator object
    return gen_cls(simulator_folder, run_parameters=run_parameters)

  def __init__(self, simulator_folder, run_parameters=None):
    """ Initialise a new `Simulator`.

    Args:
      simulator_folder: Location of a simulator.
      run_parameters: Can be used to override parameters during run time.
    """

    # Make sure simulator exists
    if not os.path.exists(simulator_folder):
      raise FileNotFoundError(f"Simulator folder {simulator_folder} does not exists")
    self._simulator_folder = simulator_folder

    # Read config
    self._config = parse_yaml(os.path.join(self._simulator_folder, "config.yaml"))

    # Get run parameters: these parameters can be used to override parameters used during training
    self._run_parameters = self._config["simulation"]["run_parameters"].copy()
    self._run_parameters.update(run_parameters or {})

    # Initialise simulation
    self._model, self._data, self.task, self.bm_model, self.perception, self.callbacks = \
      self._initialise(self._config, self._simulator_folder, self._run_parameters)

    # Set action space TODO for now we assume all actuators have control signals between [-1, 1]
    self.action_space = self._initialise_action_space()

    # Set observation space
    self.observation_space = self._initialise_observation_space()

    # Collect some episode statistics
    self._episode_statistics = {"length (seconds)": 0, "length (steps)": 0, "reward": 0}

    # Initialise viewer
    self._camera = Camera(self._run_parameters["rendering_context"], self._model, self._data, camera_id='for_testing',
                         dt=self._run_parameters["dt"])

  def _initialise_action_space(self):
    """ Initialise action space. """
    num_actuators = self.bm_model.nu + self.perception.nu
    actuator_limits = np.ones((num_actuators,2)) * np.array([-1.0, 1.0])
    return spaces.Box(low=np.float32(actuator_limits[:, 0]), high=np.float32(actuator_limits[:, 1]))

  def _initialise_observation_space(self):
    """ Initialise observation space. """
    observation = self.get_observation()
    obs_dict = dict()
    for module in self.perception.perception_modules:
      obs_dict[module.modality] = spaces.Box(dtype=np.float32, **module.get_observation_space_params())
    if "stateful_information" in observation:
      obs_dict["stateful_information"] = spaces.Box(dtype=np.float32,
                                                    **self.task.get_stateful_information_space_params())

    return spaces.Dict(obs_dict)

  def step(self, action):
    """ Step simulation forward with given actions.

    Args:
      action: Actions sampled from a policy. Limited to range [-1, 1].
    """


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
    obs = self.get_observation()

    return obs, reward, finished, info

  def get_observation(self):
    """ Returns an observation from the perception model.

    Returns:
      A dict with observations from individual perception modules. May also contain stateful information from a task.
    """

    # Get observation from perception
    observation = self.perception.get_observation(self._model, self._data)

    # Add any stateful information that is required
    stateful_information = self.task.get_stateful_information(self._model, self._data)
    if stateful_information is not None:
      observation["stateful_information"] = stateful_information

    return observation

  def reset(self):
    """ Reset the simulator and return an observation. """

    # Reset sim
    mujoco.mj_resetData(self._model, self._data)

    # Reset all models
    self.bm_model.reset(self._model, self._data)
    self.perception.reset(self._model, self._data)
    self.task.reset(self._model, self._data)

    # Do a forward so everything will be set
    mujoco.mj_forward(self._model, self._data)

    return self.get_observation()

  def callback(self, callback_name, num_timesteps):
    """ Update a callback -- may be useful during training, e.g. for curriculum learning. """
    self.callbacks[callback_name].update(num_timesteps)

  def update_callbacks(self, num_timesteps):
    """ Update all callbacks. """
    for callback_name in self.callbacks:
      self.callback(callback_name, num_timesteps)

  @property
  def config(self):
    """ Return config. """
    return copy.deepcopy(self._config)

  @property
  def run_parameters(self):
    """ Return run parameters. """
    # Context cannot be deep copied
    exclude = {"rendering_context"}
    run_params = {k: copy.deepcopy(self._run_parameters[k]) for k in self._run_parameters.keys() - exclude}
    run_params["rendering_context"] = self._run_parameters["rendering_context"]
    return run_params

  @property
  def simulator_folder(self):
    """ Return simulator folder. """
    return self._simulator_folder

  def get_state(self):
    """ Return a state of the simulator / individual components (biomechanical model, perception model, task).

    This function is used for logging/evaluation purposes, not for RL training.

    Returns:
      A dict with one float or numpy vector per keyword.
    """

    # Get time, qpos, qvel, qacc, act, ctrl of the current simulation
    state = {"timestep": self._data.time,
             "qpos": self._data.qpos.copy(),
             "qvel": self._data.qvel.copy(),
             "qacc": self._data.qacc.copy(),
             "act": self._data.act.copy(),
             "ctrl": self._data.ctrl.copy()}

    # Add state from the task
    state.update(self.task.get_state(self._model, self._data))

    # Add state from the biomechanical model
    state.update(self.bm_model.get_state(self._model, self._data))

    # Add state from the perception model
    state.update(self.perception.get_state(self._model, self._data))

    return state
