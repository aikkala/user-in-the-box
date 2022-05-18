import numpy as np
import xml.etree.ElementTree as ET
import pathlib
import os
import re
import shutil
import inspect
import mujoco
from abc import ABC, abstractmethod, abstractproperty

from ..utils.functions import parent_path
from ..utils import element_tree as ETutils
#from uitb.utils import effort_terms


class BaseBMModel:

  def __init__(self, model, data, **kwargs):

    # Initialise mujoco model of the biomechanical model, easier to manipulate things
    bm_model = mujoco.MjModel.from_xml_path(self.get_xml_file())

    # Get an rng
    self.rng = np.random.default_rng(kwargs.get("random_seed", None))

    # Total number of actuators
    self.nu = bm_model.nu

    # Number of muscle actuators
    self.na = bm_model.na

    # Number of motor actuators
    self.nm = self.nu - self.na
    self.motor_smooth_avg = np.zeros((self.nm,))
    self.motor_alpha = 0.9

    # Get actuator names (muscle and motor)
    self.actuator_names = [mujoco.mj_id2name(bm_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(bm_model.nu)]
    self.muscle_actuator_names = set(np.array(self.actuator_names)[model.actuator_trntype==3])
    self.motor_actuator_names = set(self.actuator_names) - self.muscle_actuator_names

    # Sort the names to preserve original ordering (not really necessary but looks nicer)
    self.muscle_actuator_names = sorted(self.muscle_actuator_names, key=self.actuator_names.index)
    self.motor_actuator_names = sorted(self.motor_actuator_names, key=self.actuator_names.index)

    # Find actuator indices in the simulation
    self.muscle_actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                             for actuator_name in self.muscle_actuator_names]
    self.motor_actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                            for actuator_name in self.motor_actuator_names]

    # Get joint names (dependent and independent)
    self.joint_names = [mujoco.mj_id2name(bm_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(bm_model.njnt)]
    self.dependent_joint_names = {self.joint_names[idx] for idx in
                                  np.unique(bm_model.eq_obj1id[bm_model.eq_active.astype(bool)])} \
      if bm_model.eq_obj1id is not None else set()
    self.independent_joint_names = set(self.joint_names) - self.dependent_joint_names

    # Sort the names to preserve original ordering (not really necessary but looks nicer)
    self.dependent_joint_names = sorted(self.dependent_joint_names, key=self.joint_names.index)
    self.independent_joint_names = sorted(self.independent_joint_names, key=self.joint_names.index)

    # Find dependent and independent joint indices in the simulation
    self.dependent_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                             for joint_name in self.dependent_joint_names]
    self.independent_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                               for joint_name in self.independent_joint_names]

    #self.effort_term = kwargs.get('effort_term', effort_terms.Zero())

    # If the model has a floor/ground, it should be defined so we can ignore it when cloning
    self.floor = kwargs.get("floor", None)

  def update(self, model, data):
    pass

  @classmethod
  def get_xml_file(cls):
    return os.path.join(parent_path(inspect.getfile(cls)), "bm_model.xml")

  @classmethod
  def get_floor(cls):
    return None

  @classmethod
  def insert(cls, task_tree):

    # Parse xml file
    bm_tree = ET.parse(cls.get_xml_file())
    bm_root = bm_tree.getroot()

    # Get task root
    task_root = task_tree.getroot()

    # Add defaults
    ETutils.copy_or_append("default", bm_root, task_root)

    # Add assets, except skybox
    ETutils.copy_children("asset", bm_root, task_root, exclude={"tag": "texture", "attrib": "type", "name": "skybox"})

    # Add bodies, except floor/ground
    if cls.get_floor() is not None:
      ETutils.copy_children("worldbody", bm_root, task_root,
                   exclude={"tag": cls.get_floor().tag, "attrib": "name",
                            "name": cls.get_floor().attrib.get("name", None)})
    else:
      ETutils.copy_children("worldbody", bm_root, task_root)

    # Add tendons
    ETutils.copy_children("tendon", bm_root, task_root)

    # Add actuators
    ETutils.copy_children("actuator", bm_root, task_root)

    # Add equality constraints
    ETutils.copy_children("equality", bm_root, task_root)

  @classmethod
  def clone(cls, run_folder, package_name):

    # Create 'bm_models' folder
    dst = os.path.join(run_folder, package_name, "bm_models")
    os.makedirs(dst, exist_ok=True)

    # Copy this file
    base_file = pathlib.Path(__file__)
    shutil.copyfile(base_file, os.path.join(dst, base_file.name))

    # Create an __init__.py file with the relevant import
    modules = cls.__module__.split(".")
    with open(os.path.join(dst, "__init__.py"), "w") as file:
      file.write("from ." + ".".join(modules[2:]) + " import " + cls.__name__)

    # Copy bm-model folder
    src = parent_path(inspect.getfile(cls))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True)

    # Copy assets
    shutil.copytree(os.path.join(src, "assets"), os.path.join(run_folder, package_name, "assets"),
                    dirs_exist_ok=True)

  def set_ctrl(self, model, data, action):
    data.ctrl[self.motor_actuators] = np.clip(self.motor_smooth_avg + action[:self.nm], 0, 1)
    data.ctrl[self.muscle_actuators] = np.clip(data.act[self.muscle_actuators] + action[self.nm:], 0, 1)

    # Update smoothed online estimate of motor actuation
    self.motor_smooth_avg = (1-self.motor_alpha)*self.motor_smooth_avg \
                            + self.motor_alpha*data.ctrl[self.motor_actuators]

  def reset(self, model, data):

    # TODO add kwargs for setting initial positions

    # Randomly sample qpos, qvel, act
    nq = len(self.independent_joints)
    qpos = self.rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    qvel = self.rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    act = self.rng.uniform(low=np.zeros((self.na,)), high=np.ones((self.na,)))

    # Set qpos and qvel
    data.qpos[self.dependent_joints] = 0
    data.qpos[self.independent_joints] = qpos
    data.qvel[self.dependent_joints] = 0
    data.qvel[self.independent_joints] = qvel
    data.act[self.muscle_actuators] = act

    # Reset smoothed average of motor actuator activation
    self.motor_smooth_avg = np.zeros((self.nm,))

    # Some effort terms may be stateful and need to be reset
    #self.effort_term.reset()

    # Finally update whatever needs to be updated
    self.update(model, data)