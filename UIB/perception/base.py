from abc import ABC, abstractmethod
import os
import shutil
import inspect
import numpy as np

from UIB.utils.functions import parent_path


class BaseModule(ABC):

  def __init__(self, sim, bm_model):
    self.module_folder = None
    self.bm_model = bm_model
    self.actuator_names = []
    self.joint_names = []

    # Get modality
    self.modality = parent_path(inspect.getfile(self.__class__)).parent.stem

    # Get observation shape
    self.observation_shape = self.get_observation(sim).shape

  @property
  def nu(self):
    return len(self.actuator_names)

  @staticmethod
  @abstractmethod
  def insert(task, config, **kwargs):
    pass

  @classmethod
  def clone(cls, run_folder):

    # Create "perception" folder if needed
    dst = os.path.join(run_folder, "simulator", "perception")
    os.makedirs(dst, exist_ok=True)

    # Get src for this module
    src = parent_path(inspect.getfile(cls))

    # Create upper level folder (if needed)
    modality = os.path.join(dst, src.parent.stem)
    os.makedirs(modality, exist_ok=True)

    # Copy the extractors file
    shutil.copyfile(os.path.join(src.parent, "extractors.py"), os.path.join(modality, "extractors.py"))

    # Copy module folder
    shutil.copytree(src, os.path.join(modality, src.stem), dirs_exist_ok=True)

    # Copy assets if they exist
    if os.path.isdir(os.path.join(src, "assets")):
      shutil.copytree(os.path.join(src, "assets"), os.path.join(run_folder, "simulator", "assets"),
                      dirs_exist_ok=True)

  @abstractmethod
  def reset(self, sim, rng):
    pass

  def set_ctrl(self, sim, action):
    pass

  @abstractmethod
  def get_observation(self, sim):
    pass

  @property
  @abstractmethod
  def extractor(self):
    pass

  @abstractmethod
  def get_observation_space_params(self):
    return {"low": None, "high": None, "shape": None}


class Perception:

  def __init__(self, sim, bm_model, perception_modules, run_parameters):

    # Get names of any (controllable) actuators that might be used in perception modules, like for e.g. eye movements
    self.actuator_names = []
    self.joint_names = []

    # Get extractors
    self.extractors = dict()

    self.perception_modules = []
    for module_cls, kwargs in perception_modules.items():
      kwargs = {} if kwargs is None else kwargs
      module = module_cls(sim, bm_model, **{**kwargs, **run_parameters})
      self.perception_modules.append(module)
      self.actuator_names.extend(module.actuator_names)
      self.joint_names.extend(module.joint_names)
      self.extractors[module.modality] = module.extractor()

    # Find actuators in the simulation
    actuator_names_array = np.array(sim.model.actuator_names)
    self.actuators = [np.where(actuator_names_array==actuator)[0][0] for actuator in self.actuator_names]
    self.nu = len(self.actuators)

  def set_ctrl(self, sim, action):
    num = 0
    for module in self.perception_modules:
      module.set_ctrl(sim, action[num:num+module.nu])
      num += module.nu

  def reset(self, sim, rng):
    for module in self.perception_modules:
      module.reset(sim, rng)

  def get_observation(self, sim):

    observations = {}
    for module in self.perception_modules:
      observations[module.modality] = module.get_observation(sim)

    return observations