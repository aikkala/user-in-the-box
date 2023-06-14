import numpy as np
import xml.etree.ElementTree as ET
import pathlib
import os
import shutil
import inspect
import mujoco
from abc import ABC, abstractmethod
import importlib
from typing import final

from ..utils.functions import parent_path
from ..utils import element_tree as ETutils


class BaseBMModel(ABC):

  def __init__(self, model, data, **kwargs):
    """Initializes a new `BaseBMModel`.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      **kwargs: Many keywords that should be documented somewhere
    """

    # Initialise mujoco model of the biomechanical model, easier to manipulate things
    bm_model = mujoco.MjModel.from_xml_path(self.get_xml_file())

    # Get an rng
    self._rng = np.random.default_rng(kwargs.get("random_seed", None))

    # Total number of actuators
    self._nu = bm_model.nu

    # Number of muscle actuators
    self._na = bm_model.na

    # Number of motor actuators
    self._nm = self._nu - self._na
    self._motor_smooth_avg = np.zeros((self._nm,))
    self._motor_alpha = 0.9

    # Get actuator names (muscle and motor)
    self._actuator_names = [mujoco.mj_id2name(bm_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(bm_model.nu)]
    self._muscle_actuator_names = set(np.array(self._actuator_names)[model.actuator_trntype==3])
    self._motor_actuator_names = set(self._actuator_names) - self._muscle_actuator_names

    # Sort the names to preserve original ordering (not really necessary but looks nicer)
    self._muscle_actuator_names = sorted(self._muscle_actuator_names, key=self._actuator_names.index)
    self._motor_actuator_names = sorted(self._motor_actuator_names, key=self._actuator_names.index)

    # Find actuator indices in the simulation
    self._muscle_actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                             for actuator_name in self._muscle_actuator_names]
    self._motor_actuators = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                            for actuator_name in self._motor_actuator_names]

    # Get joint names (dependent and independent)
    self._joint_names = [mujoco.mj_id2name(bm_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(bm_model.njnt)]
    self._dependent_joint_names = {self._joint_names[idx] for idx in
                                  np.unique(bm_model.eq_obj1id[bm_model.eq_active.astype(bool)])} \
      if bm_model.eq_obj1id is not None else set()
    self._independent_joint_names = set(self._joint_names) - self._dependent_joint_names

    # Sort the names to preserve original ordering (not really necessary but looks nicer)
    self._dependent_joint_names = sorted(self._dependent_joint_names, key=self._joint_names.index)
    self._independent_joint_names = sorted(self._independent_joint_names, key=self._joint_names.index)

    # Find dependent and independent joint indices in the simulation
    self._dependent_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                             for joint_name in self._dependent_joint_names]
    self._independent_joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                               for joint_name in self._independent_joint_names]

    # Get the effort model; some models might need to know dt
    self._effort_model = self.get_effort_model(kwargs.get("effort_model", {"cls": "Zero"}), dt=kwargs["dt"])


  ############ The methods below you should definitely overwrite as they are important ############

  @classmethod
  @abstractmethod
  def _get_floor(cls):
    """ If there's a floor in the bm_model.xml file it should be defined here.

    Returns:
      * None if there is no floor in the file
      * A dict like {"tag": "geom", "name": "name-of-the-geom"}, where "tag" indicates what kind of element the floor
      is, and "name" is the name of the element.
    """
    pass


  ############ The methods below are overwritable but often don't need to be overwritten ############

  def _reset(self, model, data):
    """ Resets the biomechanical model. """

    # Randomly sample qpos, qvel, act around zero values
    nq = len(self._independent_joints)
    qpos = self._rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    qvel = self._rng.uniform(low=np.ones((nq,))*-0.05, high=np.ones((nq,))*0.05)
    act = self._rng.uniform(low=np.zeros((self._na,)), high=np.ones((self._na,)))

    # Set qpos and qvel
    data.qpos[self._dependent_joints] = 0
    data.qpos[self._independent_joints] = qpos
    data.qvel[self._dependent_joints] = 0
    data.qvel[self._independent_joints] = qvel
    data.act[self._muscle_actuators] = act

    # Reset smoothed average of motor actuator activation
    self._motor_smooth_avg = np.zeros((self._nm,))

  def _update(self, model, data):
    """ Update the biomechanical model after a step has been taken in the simulator. """
    pass

  def _get_state(self, model, data):
    """ Return the state of the biomechanical model. These states are used only for logging/evaluation, not for RL
    training

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.

    Returns:
      A dict where each key should have a float or a numpy vector as their value
    """
    return dict()

  def set_ctrl(self, model, data, action):
    """ Set control values for the biomechanical model.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      action: Action values between [-1, 1]

    """
    data.ctrl[self._motor_actuators] = np.clip(self._motor_smooth_avg + action[:self._nm], 0, 1)
    data.ctrl[self._muscle_actuators] = np.clip(data.act[self._muscle_actuators] + action[self._nm:], 0, 1)

    # Update smoothed online estimate of motor actuation
    self._motor_smooth_avg = (1 - self._motor_alpha) * self._motor_smooth_avg \
                             + self._motor_alpha * data.ctrl[self._motor_actuators]

  @classmethod
  def get_xml_file(cls):
    """ Overwrite this method if you want to call the mujoco xml file something other than 'bm_model.xml'. """
    return os.path.join(parent_path(inspect.getfile(cls)), "bm_model.xml")

  def get_effort_model(self, specs, dt):
    """ Returns an initialised object of the effort model class.

    Overwrite this method if you want to define your effort models somewhere else. But note that in that case you need
    to overwrite the 'clone' method as well since it assumes the effort models are defined in
    uitb.bm_models.effort_models.

    Args:
      specs: Specifications of the effort model, in format of
        {"cls": "name-of-class", "kwargs": {"kw1": value1, "kw2": value2}}}
      dt: Elapsed time between two consecutive simulation steps

    Returns:
       An instance of a class that inherits from the uitb.bm_models.effort_models.BaseEffortModel class
    """
    module = importlib.import_module(".".join(BaseBMModel.__module__.split(".")[:-1]) + ".effort_models")
    return getattr(module, specs["cls"])(self, **{**specs.get("kwargs", {}), **{"dt": dt}})

  @classmethod
  def clone(cls, simulator_folder, package_name):
    """ Clones (i.e. copies) the relevant python files into a new location.

    Args:
       simulator_folder: Location of the simulator.
       package_name: Name of the simulator (which is a python package)
    """

    # Create 'bm_models' folder
    dst = os.path.join(simulator_folder, package_name, "bm_models")
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
    shutil.copytree(os.path.join(src, "assets"), os.path.join(simulator_folder, package_name, "assets"),
                    dirs_exist_ok=True)

    # Copy effort models
    shutil.copyfile(os.path.join(base_file.parent, "effort_models.py"), os.path.join(dst, "effort_models.py"))

  @classmethod
  def insert(cls, simulator_tree):
    """ Inserts the biomechanical model into the simulator by integrating the xml files together.

     Args:
       simulator_tree: An `xml.etree.ElementTree` containing the parsed simulator xml file
    """

    # Parse xml file
    bm_tree = ET.parse(cls.get_xml_file())
    bm_root = bm_tree.getroot()

    # Get simulator root
    simulator_root = simulator_tree.getroot()

    # Add defaults
    ETutils.copy_or_append("default", bm_root, simulator_root)

    # Add assets, except skybox
    ETutils.copy_children("asset", bm_root, simulator_root,
                          exclude={"tag": "texture", "attrib": "type", "name": "skybox"})

    # Add bodies, except floor/ground  TODO this might not be currently working
    if cls._get_floor() is not None:
      floor = cls._get_floor()
      ETutils.copy_children("worldbody", bm_root, simulator_root,
                            exclude={"tag": floor["tag"], "attrib": "name", "name": floor["name"]})
    else:
      ETutils.copy_children("worldbody", bm_root, simulator_root)

    # Add tendons
    ETutils.copy_children("tendon", bm_root, simulator_root)

    # Add actuators
    ETutils.copy_children("actuator", bm_root, simulator_root)

    # Add equality constraints
    ETutils.copy_children("equality", bm_root, simulator_root)


  ############ The methods below you should not overwrite ############

  @final
  def update(self, model, data):
    """ Updates the biomechanical model and effort model. """
    self._update(model, data)
    self._effort_model.update(model, data)

  @final
  def reset(self, model, data):
    """ Resets the biomechanical model and effort model. """
    self._reset(model, data)
    self._effort_model.reset(model, data)
    self.update(model, data)

  @final
  def get_state(self, model, data):
    """ Returns the state of the biomechanical model (as a dict). """
    state = dict()
    state.update(self._get_state(model, data))
    return state

  @final
  def get_effort_cost(self, model, data):
    """ Returns effort cost from the effort model. """
    return self._effort_model.cost(model, data)

  @property
  @final
  def independent_joints(self):
    """ Returns indices of independent joints. """
    return self._independent_joints.copy()

  @property
  @final
  def nu(self):
    """ Returns number of actuators (both muscle and motor). """
    return self._nu