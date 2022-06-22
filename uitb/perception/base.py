from abc import ABC, abstractmethod
import os
import shutil
import inspect
import numpy as np
import mujoco
import pathlib

from ..utils.functions import parent_path


class BaseModule(ABC):

  def __init__(self, model, data, bm_model, **kwargs):
    self.module_folder = None
    self.bm_model = bm_model
    self.actuator_names = []
    self.joint_names = []

    # Get an rng
    self.rng = np.random.default_rng(kwargs.get("random_seed", None))

    # Get modality
    self.modality = parent_path(inspect.getfile(self.__class__)).parent.stem

    # Get observation shape
    self.observation_shape = self.get_observation(model, data).shape

  @staticmethod
  @abstractmethod
  def insert(simulation, **kwargs):
    pass

  @abstractmethod
  def get_observation(self, model, data, info=None):
    pass

  @abstractmethod
  def _get_observation_range(self):
    return {"low": None, "high": None}

  def get_observation_space_params(self):
    return {**self._get_observation_range(), "shape": self.observation_shape}

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass

  def set_ctrl(self, model, data, action):
    pass

  @property
  def encoder(self):
    return None

  @property
  def nu(self):
    return len(self.actuator_names)

  @classmethod
  def clone(cls, run_folder, package_name):

    # Create "perception" folder if needed
    dst = os.path.join(run_folder, package_name, "perception")
    os.makedirs(dst, exist_ok=True)

    # Copy this file and __init__.py
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))
    shutil.copyfile(os.path.join(base_file.parent, "__init__.py"), os.path.join(dst, "__init__.py"))

    # Get src for this module
    src = parent_path(inspect.getfile(cls))

    # Create upper level folder (if needed)
    modality = os.path.join(dst, src.parent.stem)
    os.makedirs(modality, exist_ok=True)

    # Copy the encoders file (if it exists)
    encoders_file = os.path.join(src.parent, "encoders.py")
    if os.path.isfile(encoders_file):
      shutil.copyfile(encoders_file, os.path.join(modality, "encoders.py"))

    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(modality, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[3:]) + " import " + cls.__name__)

    # Copy module folder
    shutil.copytree(src, os.path.join(modality, src.stem), dirs_exist_ok=True)

    # Copy assets if they exist
    if os.path.isdir(os.path.join(src, "assets")):
      shutil.copytree(os.path.join(src, "assets"), os.path.join(run_folder, package_name, "assets"),
                      dirs_exist_ok=True)


class Perception:

  def __init__(self, model, data, bm_model, perception_modules, run_parameters):

    # Get names of any (controllable) actuators that might be used in perception modules, like for e.g. eye movements
    self.actuator_names = []
    self.joint_names = []

    # Get encoders
    self.encoders = dict()

    self.perception_modules = []
    for module_cls, kwargs in perception_modules.items():
      module = module_cls(model, data, bm_model, **{**kwargs, **run_parameters})
      self.perception_modules.append(module)
      self.actuator_names.extend(module.actuator_names)
      self.joint_names.extend(module.joint_names)
      self.encoders[module.modality] = module.encoder

    # Find actuators in the simulation
    self.actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                      for actuator_name in self.actuator_names]
    self.nu = len(self.actuators)

    # Find joints in the simulation
    self.joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                   for joint_name in self.joint_names]

  def set_ctrl(self, model, data, action):
    num = 0
    for module in self.perception_modules:
      module.set_ctrl(model, data, action[num:num+module.nu])
      num += module.nu

  def reset(self, model, data):
    for module in self.perception_modules:
      module.reset(model, data)
    self.update(model, data)

  def update(self, model, data):
    for module in self.perception_modules:
      module.update(model, data)

  def get_observation(self, model, data, info=None):
    observations = {}
    for module in self.perception_modules:
      observations[module.modality] = module.get_observation(model, data, info)
    return observations