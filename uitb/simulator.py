import gym
from gym import spaces
import mujoco
import os
import numpy as np
import dill
import sys
import importlib

from uitb.perception.base import Perception
from uitb.utils.viewers import Viewer

def get_clone_class(src_class):

  module = src_class.__module__.split(".")
  module[0] = "simulation"
  module = importlib.import_module(".".join(module))
  return getattr(module, src_class.__name__)


class Simulator(gym.Env):

  id = "uitb:simulator-v0"

  @classmethod
  def build(cls, config):

    # Make sure required things are defined in config
    assert "simulation" in config, "Simulation specs (simulation) must be defined in config"
    assert "bm_model" in config["simulation"], "Biomechanical model (bm_model) must be defined in config"
    assert "task" in config["simulation"], "task (task) must be defined in config"

    assert "run_parameters" in config["simulation"], "Run parameters (run_parameters) must be defined in config"
    run_parameters = config["simulation"]["run_parameters"]
    assert "action_sample_freq" in run_parameters, "Action sampling frequency (action_sample_freq) must be defined " \
                                                   "in run parameters"

    # Set simulator id
    config["id"] = cls.id

    # By default use cloned files (set to False in config for debugging)
    if "use_cloned_files" not in run_parameters:
      config["run_parameters"]["use_cloned_files"] = True

    # Create a folder for the simulator
    os.makedirs(config["run_folder"], exist_ok=True)

    # Load the environment
    config["simulation"]["task"].clone(config["run_folder"])
    task = config["simulation"]["task"].initialise_task(config)

    # Load biomechanical model
    config["simulation"]["bm_model"].clone(config["run_folder"])
    config["simulation"]["bm_model"].insert(task, config)

    # Add perception modules
    for module, kwargs in config["simulation"].get("perception_modules", {}).items():
      module.clone(config["run_folder"])
      module.insert(task, config, **kwargs)

    # TODO read the xml file directly from task.getroot() instead of writing it to a file first; need to input a dict
    #  of assets to mujoco.MjModel.from_xml_path
    task_file = os.path.join(config["run_folder"], "simulation", "task")
    with open(task_file+".xml", 'w') as file:
      task.write(file, encoding='unicode')

    # Load the mujoco model
    model = mujoco.MjModel.from_xml_path(task_file+".xml")

    # Initialise MjData
    data = mujoco.MjData(model)

    # Add dt to run parameters
    frame_skip = int(1 / (model.opt.timestep * run_parameters["action_sample_freq"]))
    run_parameters["dt"] = model.opt.timestep*frame_skip

    # Now initialise the actual classes; model and data are input to the inits so that stuff can be modified if needed
    # (e.g. move target to a specific position wrt to a body part)
    task = config["simulation"]["task"](model, data, **{**config["simulation"].get("task_kwargs", {}),
                                                        **run_parameters})
    bm_model = config["simulation"]["bm_model"](model, data, **{**config["simulation"].get("bm_model_kwargs", {}),
                                                                **run_parameters})
    perception = Perception(model, data, bm_model, config["simulation"]["perception_modules"], run_parameters)

    # It would be nice to save an xml here in addition to saving a binary model file. But seems like there's only one
    # function to save an xml file: mujoco.mj_saveLastXML, which doesn't work here because we read the original
    # task and bm_model xml files. I've tried to read again the task_file with mujoco.MjModel.from_xml_path(task_file)
    # and then call mujoco.mj_saveLastXML(task_file, model) but it doesn't save the changes into the model (like
    # setting target-plane to 55cm in front of and 10cm to the right of the biomechanical model's shoulder)

    # Save the modified model as binary
    mujoco.mj_saveModel(model, task_file+".mjcf", None)

    # Save configs
    with open(os.path.join(config["run_folder"], "config.dill"), 'wb') as file:
      dill.dump(config, file)

  def __init__(self, run_folder, run_parameters=None):

    # Read config
    with open(os.path.join(run_folder, "config.dill"), "rb") as file:
      self.config = dill.load(file)

    # Get run parameters: these parameters can be used to override parameters used during training
    if run_parameters is None:
      run_parameters = self.config["simulation"]["run_parameters"].copy()

    # Use cloned files?
    if run_parameters["use_cloned_files"]:

      # Add cloned files to python path if not there already
      if run_folder not in sys.path:
        sys.path.insert(0, self.config["run_folder"])

      # Update classes in config
      self.config["simulation"]["task"] = get_clone_class(self.config["simulation"]["task"])
      self.config["simulation"]["bm_model"] = get_clone_class(self.config["simulation"]["bm_model"])
      perception_modules = {}
      for module, kwargs in self.config["simulation"].get("perception_modules", {}).items():
        perception_modules[get_clone_class(module)] = kwargs
      self.config["simulation"]["perception_modules"] = perception_modules

    # Load the mujoco model
    self.model = mujoco.MjModel.from_binary_path(os.path.join(run_folder, "simulation", "task.mjcf"))

    # Initialise data
    self.data = mujoco.MjData(self.model)

    # Get frame skip
    self.frame_skip = int(1 / (self.model.opt.timestep * run_parameters["action_sample_freq"]))

    # Add dt to run parameters so it's easier to pass to models
    run_parameters["dt"] = self.dt

    # Initialise classes
    self.task = self.config["simulation"]["task"](self.model, self.data, **{
      **self.config["simulation"].get("task_kwargs", {}), **run_parameters})
    self.bm_model = self.config["simulation"]["bm_model"](self.model, self.data, **{
      **self.config["simulation"].get("bm_model_kwargs", {}), **run_parameters})
    self.perception = Perception(self.model, self.data, self.bm_model, self.config["simulation"]["perception_modules"], run_parameters)

    # Set action space TODO for now we assume all actuators have control signals between [-1, 1]
    self.action_space = self.initialise_action_space()

    # Set observation space
    self.observation_space = self.initialise_observation_space()

    # Collect some episode statistics
    self._episode_statistics = {"length (seconds)": 0, "length (steps)": 0, "reward": 0}

    # Initialise viewer
    #self.viewer = Viewer(self.model, self.data, camera_name='for_testing',
    #                     resolution=[640, 480], dt=self.dt)

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