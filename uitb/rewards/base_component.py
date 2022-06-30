from abc import ABC, abstractmethod
import mujoco
import numpy as np
import os
import shutil
import inspect
import pathlib

from ..utils.functions import parent_path


class BaseRewardComponent(ABC):
  _name = "UndefinedReward"

  def __init__(self, model, data, bm_model, perception_modules, task, run_parameters, **kwargs):
    self.model = model
    self.data = data

    self.bm_model = bm_model
    self.perception_modules = perception_modules
    self.task = task
    self.run_parameters = run_parameters

  @abstractmethod
  def reward(self, model, data):
    pass

  @property
  def name(self):
      return self._name

  @classmethod
  def clone(cls, run_folder, package_name):
    # Create 'rewards' folder
    dst = os.path.join(run_folder, package_name, "rewards")
    os.makedirs(dst, exist_ok=True)

    # Copy this file and __init__.py
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))

    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(dst, "__init__.py"), "a") as file:
      file.write("from ." + ".".join(modules[2:]) + " import " + cls.__name__ + " as " + cls.__name__.split("RewardComponent")[0] + "\n")

    # Copy reward component folder
    src = parent_path(inspect.getfile(cls))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True)

    # # Copy effort models
    # shutil.copytree(os.path.join(base_file.parent, "effort_models.py"), os.path.join(dst, "effort_models.py"))

class Zero(BaseRewardComponent):
  _name = "ZeroReward"

  def reward(self, model, data):
    return 0

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass