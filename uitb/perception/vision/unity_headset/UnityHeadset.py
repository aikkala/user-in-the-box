import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from collections import deque
import cv2

from ...base import BaseModule
from ....utils.functions import parent_path
from ....utils.rendering import Camera
from ..encoders import small_cnn

class UnityHeadset(BaseModule):

  def __init__(self, model, data, bm_model, resolution, pos=None, quat=None, body="worldbody", channels=None,
               buffer=None, **kwargs):
    """
    This class represents a Unity Headset camera in VR simulations. Observation is received from the (Unity) task
     instead of mujoco
    Args:
      model: A MjModel object of the simulation
      data: A MjData object of the simulation
      bm_model: A biomechanical model class object inheriting from BaseBMModel
      resolution: Resolution in pixels [width, height]
      pos: Position of the camera [x, y, z]
      quat: Orientation of the camera as a quaternion [w, x, y, z]
      body (optional): Body to which the camera is attached, default is 'worldbody'
      channels (optional): Which channels to use; 0: red, 1: green, 2: blue. Default value is None, which means that all channels are used (i.e. same as channels=[0,1,2])
      buffer (optional): Defines a buffer of given length (in seconds) that is utilized to include prior observations
      **kwargs (optional): Keyword args that may be used
    """

    self._model = model
    self._data = data

    # Probably already called
    mujoco.mj_forward(self._model, self._data)

    # Set camera specs
    if channels is None:
      channels = [0, 1, 2]
    self._channels = channels
    self._resolution = resolution
    self._pos = pos
    self._quat = quat
    self._body = body

    # Define a vision buffer for including previous visual observations
    self._buffer = None
    if buffer is not None:
      assert "dt" in kwargs, "dt must be defined in order to include prior observations"
      maxlen = 1 + int(buffer/kwargs["dt"])
      self._buffer = deque(maxlen=maxlen)

    super().__init__(model, data, bm_model, **kwargs)

  @staticmethod
  def insert(simulation, **kwargs):
    pass

  def get_observation(self, model, data, info=None):

    # The observation is transferred from task through 'info' TODO a smarter architecture?

    if not info:
      rgb = np.zeros((self._resolution[1], self._resolution[0], 3))
    else:
      # Normalise
      rgb = (info["unity_observation"] / 255.0 - 0.5) * 2

    if rgb.shape != (self._resolution[1], self._resolution[0], 3):
      # Sometimes the screen hasn't been resized yet when first screenshot arrives
      print(f"Resizing from {[rgb.shape[1], rgb.shape[0]]} to {self._resolution}")
      rgb = cv2.resize(info["unity_observation"], dsize=tuple(self._resolution), interpolation=cv2.INTER_CUBIC)

    # Transpose channels
    obs = np.transpose(rgb, [2, 0, 1])

    # Choose channels
    obs = obs[self._channels, :, :]

    # Include prior observation if needed
    if self._buffer is not None:
      # Update buffer
      if len(self._buffer) > 0:
        self._buffer.pop()
      while len(self._buffer) < self._buffer.maxlen:
        self._buffer.appendleft(obs)

      # Use latest and oldest observation, and their difference
      obs = np.concatenate([self._buffer[0], self._buffer[-1], self._buffer[-1] - self._buffer[0]], axis=2)

    return obs

  def _get_observation_range(self):
    return {"low": -1, "high": 1}

  def reset(self, model, data):
    if self._buffer is not None:
      self._buffer.clear()

  @property
  def encoder(self):
    return small_cnn(observation_shape=self.observation_shape, out_features=256)
