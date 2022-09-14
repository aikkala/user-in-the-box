import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import cv2
import xml.etree.ElementTree as ET

from UnityClient import UnityClient


from ..base import BaseTask


class UnityDemo(BaseTask):

  def __init__(self, model, data, end_effector, root_name, root_position, root_quaternion, **kwargs):
    super().__init__(model, data, **kwargs)

    # Fire up the Unity client if we're really stasrting the simulation and not just building
    if not kwargs.get("build", False):
      # Start a Unity client
      self._unity_client = UnityClient(step_size=kwargs["dt"], port=5555)

      # Wait until app is up and running. Ping the app and receive initial state for resetting
      self.initial_state = self._unity_client.handshake()

    # This task requires an end-effector to be defined; also, it must be a body
    # Would be nicer to have this check in the "initialise" method of this class, but not currently possible because
    # of the order in which mujoco xml files are merged (task -> bm_model -> perception).
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector) == -1:
      raise KeyError(f"'end_effector' must be a body, no body called {end_effector} found in the model")
    self._end_effector = end_effector

    # Use early termination if target is not hit in time
    self._max_steps = self._action_sample_freq*10

    # Assumes the bm model is facing towards positive x axis
    #rotate = Rotation.from_quat(np.array([0.7071068, 0, 0, -0.7071068]))
    #quat = Rotation.from_quat(model.body(root_name).quat)
    #quat = (quat * rotate).as_quat()
    #model.body(root_name).quat = quat

    # Set root position and orientation
    #data.body(root_name).xpos = root_position
    #data.body(root_name).xquat = root_quaternion

    # Set camera angle TODO need to rethink how cameras are implemented
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([2.2, -1.8, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])
    #model.cam_pos[model.camera_name2id('for_testing')] = np.array([-0.8, -0.6, 1.5])
    #model.cam_quat[model.camera_name2id('for_testing')] = np.array(
    #  [0.718027, 0.4371043, -0.31987, -0.4371043])

  @classmethod
  def initialise(cls, task_kwargs):

    if "end_effector" not in task_kwargs:
      raise KeyError("Key 'end_effector' is missing from task kwargs. The end-effector must be defined for this "
                     "environment")
    end_effector = task_kwargs["end_effector"]
    if "relpose" not in task_kwargs:
      raise KeyError("Key 'relpose' is missing from task kwargs. This key defines the relative pose of controller wrt "
                     "to end-effector, and it must be defined for this environment")

    # Parse xml file
    tree = ET.parse(cls.get_xml_file())
    root = tree.getroot()

    # Modify controller position and orientation
    controller_right = root.find(".//body[@name='controller-right']")
    controller_right.attrib["pos"] = np.array2string(np.array(task_kwargs["controller_right_position"]))[1:-1]
    controller_right.attrib["quat"] = np.array2string(np.array(task_kwargs["controller_right_quaternion"]))[1:-1]

    # Add a weld connect
    equality = root.find("equality")
    if equality is None:
      equality = ET.Element("equality")
      root.append(equality)
    equality.append(ET.Element("weld", name="controller-right-weld", body1="controller-right", body2=end_effector,
                               relpose=task_kwargs["relpose"], active="false"))

    return tree

  def _update(self, model, data):

    info = {}
    is_finished = False

    # Let Unity app know that the episode has terminated
    if self._steps >= self._max_steps:
      is_finished = True
      info["termination"] = "max_steps_reached"

    # Send end effector position and rotation to unity, get reward and image from camera
    reward, image_bytes, is_app_finished = self._unity_client.step(self._create_state(model, data), is_finished)

    if is_finished and not is_app_finished:
      raise RuntimeError("User simulation has terminated an episode but Unity app has not")

    # Form an image of the received bytes
    info = {"unity_observation": np.flip(cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), -1), 2)}

    return reward, is_app_finished, info

  def _create_state(self, model, data):
    #model.body("controller-right").pos = np.array([1, 0  , 1.0])
    #model.body("controller-right").quat = np.array([1, 0, 0, 0])
    pos = model.body("controller-right").pos
    quat = Rotation.from_quat(np.concatenate([model.body("controller-right").quat[1:], model.body("controller-right").quat[:1]]))

    #T_controller = np.eye(4)
    #T_controller[:3, :3] = quat.as_matrix()
    #T_controller[:3, 3] = pos

    #rotate1 = Rotation.from_quat(np.array([0.7071068, 0, 0., 0.7071068]))
    #T_controller = np.matmul(rotate1, np.linalg.inv(T_controller))

    #rotate1 = Rotation.from_quat(np.array([0.7071068, 0, 0., 0.7071068]))
    rotate1 = Rotation.from_quat(np.array([0, 0.7071068, 0.7071068, 0]))

    #pos = np.array([1, 0  , 0.5])
    #quat = Rotation.from_quat(np.array([0, 0, 0, 1]))
    #quat = model.body("controller-right").quat
    quat = quat*rotate1
    rotate = Rotation.from_matrix(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    #rotate = Rotation.from_quat(np.array([0.7071068, 0, 0.7071068, 0.])).as_matrix()
    pos = rotate.apply(pos)#np.matmul(rotate, pos)
    #quat = np.array([1.0, 0.0, 0.0, 0.0])
    #rot = np.matmul(rotate.as_matrix(), orientation.as_matrix())
    #rotate = Rotation.from_euler('y', 90, degrees=True)
    quat = quat*rotate


    quat = quat.as_quat()
    #quat = Rotation.from_matrix(T_controller[:3, :3]).as_quat()

    #quat = Rotation.from_matrix(rot).as_quat()
    state = {
      "headsetPosition": {"x": 0, "y": 1, "z": 0},
      "leftControllerPosition": {"x": 0, "y": 0, "z": 0},
#      "rightControllerPosition": {"x": -pos[1], "y": pos[2], "z": pos[0]},
      "rightControllerPosition": {"x": pos[0], "y": pos[2], "z": pos[1]},
      "headsetRotation": {"x": 0, "y": 0, "z": 0, "w": 1.0},
      "leftControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1.0},
      #"rightControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1}
      "rightControllerRotation": {"x": quat[0], "y": quat[2], "z": quat[1], "w": -quat[3]}
    }
    return state

  def _reset(self, model, data):
    # Reset and receive an observation
    image_bytes = self._unity_client.reset(self._create_state(model, data))
    info = {"unity_observation": np.flip(cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), -1), 2)}
    return info
