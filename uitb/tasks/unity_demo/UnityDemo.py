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

    # Start a Unity client
    self._unity_client = UnityClient(step_size=kwargs["dt"])

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

    # Send end effector position and rotation to unity, get reward and image from camera
    reward, screenshot_bytes, is_finished = self._unity_client.step(self._create_unity_state(data))

    # Form an image of the received bytes
    info = {"unity_observation": np.flip(cv2.imdecode(np.asarray(screenshot_bytes, dtype=np.uint8), -1), 2)}

    if self._steps >= self._max_steps:
      is_finished = True
      info["termination"] = "max_steps_reached"

    return reward, is_finished, info

  def _create_unity_state(self, data):
    pos = data.body("index3").xpos
    quat = Rotation.from_quat(np.concatenate([data.body("index3").xquat[1:], data.body("index3").xquat[:1]]))
    rotate = Rotation.from_euler('z', 180, degrees=True)
    quat = (quat * rotate).as_quat()
    positions = [
      {"x": 0, "y": 1, "z": 0},  # headset pos
      {"x": 0, "y": 0, "z": 0},  # left controller pos
      {"x": -pos[1], "y": pos[2], "z": pos[0]},  # right controller pos
    ]
    orientations = [
      {"x": 0, "y": 0, "z": 0, "w": 1.0},  # headset rot
      {"x": 0, "y": 0, "z": 0, "w": 1.0},  # left controller rot
      {"x": -quat[0], "y": -quat[2], "z": -quat[1], "w": quat[3]}  # right controller rot
    ]
    return {"positions": positions, "orientations": orientations}

  def _reset(self, model, data):

    # Reset unity position and task state
    positions = [
      {"x": 0, "y": 1, "z": 0},  # headset pos
      {"x": 0, "y": 0, "z": 0},  # left controller pos
      {"x": 0, "y": 0, "z": 0},  # right controller pos
    ]
    orientations = [
      {"x": 0, "y": 0, "z": 0, "w": 1},  # headset rot
      {"x": 0, "y": 0, "z": 0, "w": 1},  # left controller rot
      {"x": 0, "y": 0, "z": 0, "w": 1}  # right controller rot
    ]

    # Reset and receive an observation
    screenshot_bytes = self._unity_client.reset({"positions": positions, "orientations": orientations}, [0]*8)
    info = {"unity_observation": np.flip(cv2.imdecode(np.asarray(screenshot_bytes, dtype=np.uint8), -1), 2)}
    return info
