from abc import ABC, abstractmethod
import os
import shutil
import inspect
import mujoco
import numpy as np
import xml.etree.ElementTree as ET

from ..utils.functions import parent_path


class BaseTask(ABC):

  def __init__(self, model, data, **kwargs):

    # Initialise mujoco model of the task, easier to manipulate things
    task_model = mujoco.MjModel.from_xml_path(self.get_xml_file())

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
    pass

  def get_stateful_information(self, model, data):
    return None

  def get_stateful_information_space_params(self):
    return None

  @property
  def stateful_information_encoder(self):
    return None

  @classmethod
  def initialise_task(cls, config):
    # Parse xml file and return the tree
    return ET.parse(cls.get_xml_file())

  @classmethod
  def clone(cls, run_folder):

    # Create 'tasks' folder
    dst = os.path.join(run_folder, "simulation", "tasks")
    os.makedirs(dst, exist_ok=True)

    # Copy env folder
    src = parent_path(inspect.getfile(cls))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True)

    # Copy assets if they exist
    if os.path.isdir(os.path.join(src, "assets")):
      shutil.copytree(os.path.join(src, "assets"), os.path.join(run_folder, "simulation", "assets"),
                      dirs_exist_ok=True)

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
