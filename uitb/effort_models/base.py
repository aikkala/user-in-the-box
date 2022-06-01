from abc import ABC, abstractmethod
import os
import inspect
import shutil
import pathlib


class BaseEffortModel(ABC):

  def __init__(self, bm_model, **kwargs):
    self._bm_model = bm_model

  @abstractmethod
  def cost(self, model, data):
    pass

  @abstractmethod
  def reset(self):
    pass

  @classmethod
  def clone(cls, run_folder, package_name):

    # Create 'effort_models' folder
    dst = os.path.join(run_folder, package_name, "effort_models")
    os.makedirs(dst, exist_ok=True)

    # Copy this file
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))

    # Copy the relevant effort model file
    src = pathlib.Path(inspect.getfile(cls))
    shutil.copyfile(src, os.path.join(dst, src.name))

    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[2:]) + " import " + cls.__name__)