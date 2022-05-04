import numpy as np
import xml.etree.ElementTree as ET
import pathlib
import os
import shutil
import inspect
import mujoco_py
from abc import ABC, abstractmethod, abstractproperty

from uitb.utils.functions import project_path, parent_path
from uitb.utils import effort_terms
import uitb.utils.element_tree as ETutils


class BaseBMModel:

  xml_file = None
  floor = None

  def __init__(self, sim, **kwargs):

    # Initialise the mujoco model, easier to manipulate things
    model = mujoco_py.load_model_from_path(self.xml_file)

    # Get actuator names and joint names
    self.muscle_actuator_names = set(np.array(model.actuator_names)[model.actuator_trntype==3])
    self.motor_actuator_names = set(model.actuator_names) - self.muscle_actuator_names
    self.joint_names = model.joint_names

    # Total number of actuators
    self.nu = model.nu

    # Number of muscle actuators
    self.na = model.na

    # Number of motor actuators
    self.nm = self.nu - self.na
    self.motor_smooth_avg = np.zeros((self.nm,))
    self.motor_alpha = 0.9

    # Find actuators in the simulation
    actuator_names_array = np.array(sim.model.actuator_names)
    self.muscle_actuators = [np.where(actuator_names_array==actuator)[0][0] for actuator in self.muscle_actuator_names]
    self.motor_actuators = [np.where(actuator_names_array==actuator)[0][0] for actuator in self.motor_actuator_names]

    # Get dependent and independent joint names
    self.dependent_joint_names = {self.joint_names[idx] for idx in
                                  np.unique(model.eq_obj1id[model.eq_active.astype(bool)])} \
      if model.eq_obj1id is not None else set()
    self.independent_joint_names = set(self.joint_names) - self.dependent_joint_names

    # Find dependent and independent joints in the simulation
    sim_joint_names = np.array(sim.model.joint_names)
    self.dependent_joints = [np.where(sim_joint_names==joint)[0][0] for joint in self.dependent_joint_names]
    self.independent_joints = [np.where(sim_joint_names==joint)[0][0] for joint in self.independent_joint_names]

    #self.effort_term = kwargs.get('effort_term', effort_terms.Zero())

    # If the model has a floor/ground, it should be defined so we can ignore it when cloning
    self.floor = kwargs.get("floor", None)

  @staticmethod
  def insert(task_tree, config):

    C = config["simulator"]["bm_model"]

    # Parse xml file
    bm_tree = ET.parse(C.xml_file)
    bm_root = bm_tree.getroot()

    # Get task root
    task_root = task_tree.getroot()

    # Add defaults
    ETutils.copy_or_append("default", bm_root, task_root)

    # Add assets, except skybox
    ETutils.copy_children("asset", bm_root, task_root, exclude={"tag": "texture", "attrib": "type", "name": "skybox"})

    # Add bodies, except floor/ground
    if C.floor is not None:
      ETutils.copy_children("worldbody", bm_root, task_root,
                   exclude={"tag": C.floor.tag, "attrib": "name",
                            "name": C.floor.attrib.get("name", None)})
    else:
      ETutils.copy_children("worldbody", bm_root, task_root)

    # Add tendons
    ETutils.copy_children("tendon", bm_root, task_root)

    # Add actuators
    ETutils.copy_children("actuator", bm_root, task_root)

    # Add equality constraints
    ETutils.copy_children("equality", bm_root, task_root)

  @classmethod
  def clone(cls, run_folder):

    # Create 'bm_models' folder
    dst = os.path.join(run_folder, "simulator", "bm_models")
    os.makedirs(dst, exist_ok=True)

    # Copy bm-model folder
    src = parent_path(inspect.getfile(cls))
    shutil.copytree(src, os.path.join(dst, src.stem), dirs_exist_ok=True)

    # Copy assets
    shutil.copytree(os.path.join(src, "assets"), os.path.join(run_folder, "simulator", "assets"),
                    dirs_exist_ok=True)

  def set_ctrl(self, sim, action):
    sim.data.ctrl[self.motor_actuators] = np.clip(self.motor_smooth_avg + action[:self.nm], 0, 1)
    sim.data.ctrl[self.muscle_actuators] = np.clip(sim.data.act[self.muscle_actuators] + action[self.nm:], 0, 1)

    # Update smoothed online estimate of motor actuation
    self.motor_smooth_avg = (1-self.motor_alpha)*self.motor_smooth_avg \
                            + self.motor_alpha*sim.data.ctrl[self.motor_actuators]

  @abstractmethod
  def update(self, sim):
    pass

  def reset(self, sim, rng):

    # TODO add kwargs for setting initial positions

    # Randomly sample qpos, qvel, act
    nq = len(self.independent_joints)
    qpos = rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    qvel = rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    act = rng.uniform(low=np.zeros((self.na,)), high=np.ones((self.na,)))

    # Set qpos and qvel
    sim.data.qpos[self.dependent_joints] = 0
    sim.data.qpos[self.independent_joints] = qpos
    sim.data.qvel[self.dependent_joints] = 0
    sim.data.qvel[self.independent_joints] = qvel
    sim.data.act[self.muscle_actuators] = act

    # Reset smoothed average of motor actuator activation
    self.motor_smooth_avg = np.zeros((self.nm,))

    # Some effort terms may be stateful and need to be reset
    #self.effort_term.reset()