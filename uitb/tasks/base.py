from abc import ABC, abstractmethod
import os
import shutil
import inspect
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import pathlib
import importlib

from ..utils.functions import parent_path


class BaseTask(ABC):

  def __init__(self, model, data, **kwargs):

    # Initialise mujoco model of the task, easier to manipulate things
    task_model = mujoco.MjModel.from_xml_path(self.get_xml_file())

    # Get action sample freq
    self.action_sample_freq = kwargs["action_sample_freq"]


    # Get an rng
    self.rng = np.random.default_rng(kwargs.get("random_seed", None))

    # Get actuator names and joint names (if any)
    self.actuator_names = [mujoco.mj_id2name(task_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(task_model.nu)]
    self.joint_names = [mujoco.mj_id2name(task_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(task_model.njnt)]

    # Find actuator indices in the simulation
    self.actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                      for actuator_name in self.actuator_names]

    # Find joint indices in the simulation
    self.joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                   for joint_name in self.joint_names]

  @classmethod
  def get_xml_file(cls):
    return os.path.join(parent_path(inspect.getfile(cls)), "task.xml")

  @abstractmethod
  def update(self, model, data):
    pass

  @abstractmethod
  def reset(self, model, data):
    return dict()

  def get_stateful_information(self, model, data):
    return None

  def get_stateful_information_space_params(self):
    return None

  @property
  def stateful_information_encoder(self):
    return None

  @classmethod
  def initialise(cls, task_kwargs):
    # Parse xml file and return the tree
    return ET.parse(cls.get_xml_file())

  @classmethod
  def clone(cls, run_folder, package_name):

    # Create 'tasks' folder
    dst = os.path.join(run_folder, package_name, "tasks")
    os.makedirs(dst, exist_ok=True)

    # Copy this file and __init__.py
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))

    # Create an __init__.py file with the relevant import
    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[2:]) + " import " + cls.__name__)

    # Copy env folder
    src = parent_path(inspect.getfile(cls))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True)

    # Copy assets if they exist
    if os.path.isdir(os.path.join(src, "assets")):
      shutil.copytree(os.path.join(src, "assets"), os.path.join(run_folder, package_name, "assets"),
                      dirs_exist_ok=True)

  @classmethod
  def get_reward_function(cls, specs, module_name="reward_functions"):
    module = importlib.import_module(".".join(cls.__module__.split(".")[:-1]) + f".{module_name}")
    return getattr(module, specs["cls"])(**specs.get("kwargs", {}))

  def _get_body_xvelp_xvelr(self, model, data, bodyname):
    # TODO: test this reimplementation of mujoco-py
    jacp = np.zeros(3 * model.nv)
    jacr = np.zeros(3 * model.nv)
    mujoco.mj_jacBody(model, data, jacp[:], jacr[:],
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bodyname))
    xvelp = jacp.reshape((3, model.nv)).dot(data.qvel[:])
    xvelr = jacr.reshape((3, model.nv)).dot(data.qvel[:])

    return xvelp, xvelr

  def _get_geom_xvelp_xvelr(self, model, data, geomname):
    # TODO: test this reimplementation of mujoco-py
    jacp = np.zeros(3 * model.nv)
    jacr = np.zeros(3 * model.nv)
    mujoco.mj_jacGeom(model, data, jacp[:], jacr[:],
                      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geomname))
    xvelp = jacp.reshape((3, model.nv)).dot(data.qvel[:])
    xvelr = jacr.reshape((3, model.nv)).dot(data.qvel[:])

    return xvelp, xvelr
