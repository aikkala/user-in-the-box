import numpy as np
import mujoco
#import VREnv
from scipy.spatial.transform import Rotation
import cv2
from UnityClient import UnityClient

from ..base import BaseTask


class UnityDemo(BaseTask):

  def __init__(self, model, data, end_effector, **kwargs):
    super().__init__(model, data, **kwargs)

    # Fire up the Unity client if we're really stasrting the simulation and not just building
    if not kwargs.get("build", False):
      # Start a Unity client
      self._unity_client = UnityClient(step_size=kwargs["dt"], port=5555)

      # Wait until app is up and running. Ping the app and receive initial state for resetting
      self.initial_state = self._unity_client.handshake()

    # This task requires an end-effector to be defined
    self._end_effector = end_effector

    # Use early termination if target is not hit in time
    self._max_steps = self._action_sample_freq*10

    # Set camera angle TODO need to rethink how cameras are implemented
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([1.1, -0.9, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])
    #model.cam_pos[model.camera_name2id('for_testing')] = np.array([-0.8, -0.6, 1.5])
    #model.cam_quat[model.camera_name2id('for_testing')] = np.array(
    #  [0.718027, 0.4371043, -0.31987, -0.4371043])

  def _update(self, model, data):

    info = {}
    is_finished = False

    # Let Unity app know that the episode has terminated
    if self._steps >= self._max_steps:
      is_finished = True
      info["termination"] = "max_steps_reached"

    # Send end effector position and rotation to unity, get reward and image from camera
    reward, image_bytes, is_app_finished = self._unity_client.step(self._create_state(data), is_finished)

    if is_finished and not is_app_finished:
      raise RuntimeError("User simulation has terminated an episode but Unity app has not")

    # Form an image of the received bytes
    info = {"unity_observation": np.flip(cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), -1), 2)}

    return reward, is_app_finished, info

  def _create_state(self, data):
    pos = data.body("index3").xpos
    quat = Rotation.from_quat(np.concatenate([data.body("index3").xquat[1:], data.body("index3").xquat[:1]]))
    rotate = Rotation.from_euler('z', 180, degrees=True)
    quat = (quat * rotate).as_quat()
    state = {
      "headsetPosition": {"x": 0, "y": 1, "z": 0},
      "leftControllerPosition": {"x": 0, "y": 0, "z": 0},
      "rightControllerPosition": {"x": -pos[1], "y": pos[2], "z": pos[0]},
      "headsetRotation": {"x": 0, "y": 0, "z": 0, "w": 1.0},
      "leftControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1.0},
      "rightControllerRotation": {"x": -quat[0], "y": -quat[2], "z": -quat[1], "w": quat[3]}
    }
    return state

  def _reset(self, model, data):
    # Reset and receive an observation
    image_bytes = self._unity_client.reset(self._create_state(data))
    info = {"unity_observation": np.flip(cv2.imdecode(np.asarray(image_bytes, dtype=np.uint8), -1), 2)}
    return info
