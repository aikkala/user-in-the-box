import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import cv2
import xml.etree.ElementTree as ET

from UnityClient import UnityClient

from ..base import BaseTask
from ...utils.transformations import transformation_matrix


class UnityDemo(BaseTask):

  def __init__(self, model, data, end_effector, relpose, **kwargs):
    super().__init__(model, data, **kwargs)

    # Fire up the Unity client if we're really stasrting the simulation and not just building
    if not kwargs.get("build", False):
      # Start a Unity client
      self._unity_client = UnityClient(unity_executable=kwargs["unity_executable"], step_size=kwargs["dt"])

      # Wait until app is up and running. Ping the app and receive initial state for resetting
      self.initial_state = self._unity_client.handshake()

    # This task requires an end-effector to be defined; also, it must be a body
    # Would be nicer to have this check in the "initialise" method of this class, but not currently possible because
    # of the order in which mujoco xml files are merged (task -> bm_model -> perception).
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector) == -1:
      raise KeyError(f"'end_effector' must be a body, no body called {end_effector} found in the model")
    self._end_effector = end_effector

    # Use early termination if target is not hit in time
    self._max_steps = self._action_sample_freq*20

    # Geom's mass property is not saved when we save the integrated model xml file (would be saved in binary mjcf
    # though). So let's set it here again just to be sure
    model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "controller-right")] = 0.503

    # Do a forward step so body positions are updated
    mujoco.mj_forward(model, data)

    # 'relpose' is defined with respect to the end-effector. We need to move the controller to the correct position
    # to make sure there aren't any quick movements in the first timesteps when mujoco starts enforcing the equality
    # constraint.
    T1 = transformation_matrix(pos=data.body(self._end_effector).xpos, quat=data.body(self._end_effector).xquat)
    T2 = transformation_matrix(pos=relpose[:3], quat=relpose[3:])
    T = np.matmul(T1, np.linalg.inv(T2))
    model.body("controller-right").pos = T[:3, 3]
    model.body("controller-right").quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)

    # Set camera angle
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([2.2, -1.8, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])

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

    # Add a weld connect
    equality = root.find("equality")
    if equality is None:
      equality = ET.Element("equality")
      root.append(equality)
    equality.append(ET.Element("weld", name="controller-right-weld", body1="controller-right", body2=end_effector, relpose=" ".join([str(x) for x in task_kwargs["relpose"]]), active="true"))

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

  @staticmethod
  def _transform_to_unity(pos, quat):

    # The coordinate axis needs to be rotated to match the one used in Unity
    rot = Rotation.from_quat(np.roll(quat, -1))
    unity_transform = Rotation.from_quat(np.array([0, 0.7071068, 0.7071068, 0]))
    rot = rot*unity_transform

    # Get the quaternion; quat is in scalar-last format
    quat = rot.as_quat()

    # Transform from MuJoCo's right hand side coordinate system to Unity's left hand side coordinate system
    pos = {"x": pos[0], "y": pos[2], "z": pos[1]}
    quat = {"x": quat[0], "y": quat[2], "z": quat[1], "w": -quat[3]}

    return pos, quat

  def _create_state(self, model, data):

    # Get position and rotation of right controller
    controller_right_pos, controller_right_quat = \
      self._transform_to_unity(data.body("controller-right").xpos, data.body("controller-right").xquat)

    # Get position and rotation of headset
#    headset_pos, headset_quat = \
#      self._transform_to_unity(data.body("headset").xpos, data.body("headset").xquat)

    # Create the state
    state = {
      "headsetPosition": {"x": 0, "y": 1, "z": 0},
      "leftControllerPosition": {"x": 0, "y": 0, "z": 0},
      "rightControllerPosition": controller_right_pos,
      #"rightControllerPosition": {"x": .08, "y": 0.75, "z": 0.08},
      "headsetRotation": {"x": 0, "y": 0.7071068, "z": 0, "w": 0.7071068},
      "leftControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1.0},
      "rightControllerRotation": controller_right_quat
    }
    return state

  def _reset(self, model, data):
    # Reset and receive an observation
    image_bytes = self._unity_client.reset(self._create_state(model, data))
    info = {"unity_observation": np.flip(cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), -1), 2)}
    return info
