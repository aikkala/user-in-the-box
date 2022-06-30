from abc import ABC, abstractmethod
import numpy as np
import os
import shutil
import inspect
import pathlib

class Reward(ABC):

  def __init__(self, **kwargs):
    self._components = []
    self._components_rewards = []

  def add_component(self, component):
    self._components.append(component)

  def get_reward(self):
    self._components_rewards = [component.reward() for component in self._components]
    return np.sum(self._components_rewards)

  @property
  def components(self):
    return self._components

  @property
  def components_rewards(self):
    return self._components_rewards

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
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[2:]) + " import " + cls.__name__ + "\n\n")
