from abc import ABC, abstractmethod
import os
import shutil
import inspect
import mujoco_py
import numpy as np
import xml.etree.ElementTree as ET

from uitb.utils.functions import parent_path


class BaseTask(ABC):

  xml_file = None

  def __init__(self, sim, **kwargs):

    # Initialise the mujoco model, easier to manipulate things
    model = mujoco_py.load_model_from_path(self.xml_file)

    # Get actuator names and joint names (if any)
    self.actuator_names = model.actuator_names
    self.joint_names = model.joint_names

    # Find actuators in the simulation
    actuator_names_array = np.array(sim.model.actuator_names)
    self.actuators = [np.where(actuator_names_array==actuator)[0][0] for actuator in self.actuator_names]

    # Get dependent and independent joint names
    self.dependent_joint_names = {self.joint_names[idx] for idx in
                                  np.unique(model.eq_obj1id[model.eq_active.astype(bool)])} \
      if model.eq_obj1id is not None else set()
    self.independent_joint_names = set(self.joint_names) - self.dependent_joint_names

    # Find dependent and independent joints in the simulation
    sim_joint_names = np.array(sim.model.joint_names)
    self.dependent_joints = [np.where(sim_joint_names==joint)[0][0] for joint in self.dependent_joint_names]
    self.independent_joints = [np.where(sim_joint_names==joint)[0][0] for joint in self.independent_joint_names]


  @abstractmethod
  def update(self, sim):
    pass

  @abstractmethod
  def reset(self, sim, rng):
    pass

  def get_stateful_information(self, sim):
    return None

  def get_stateful_information_space_params(self):
    return None

  @property
  def stateful_information_extractor(self):
    return None

  @classmethod
  def initialise_task(cls, config):
    # Parse xml file and return the tree
    return ET.parse(cls.xml_file)

  @classmethod
  def clone(cls, run_folder):

    # Create 'tasks' folder
    dst = os.path.join(run_folder, "simulator", "tasks")
    os.makedirs(dst, exist_ok=True)

    # Copy env folder
    src = parent_path(inspect.getfile(cls))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True)

    # Copy assets if they exist
    if os.path.isdir(os.path.join(src, "assets")):
      shutil.copytree(os.path.join(src, "assets"), os.path.join(run_folder, "simulator", "assets"),
                      dirs_exist_ok=True)
