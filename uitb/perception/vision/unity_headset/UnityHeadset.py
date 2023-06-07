import numpy as np
import mujoco
from collections import deque
import cv2

from ...base import BaseModule
#from ..encoders import small_cnn


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
      channels = [0, 1, 2, 3]
    self._channels = channels
    self._resolution = resolution
    self._pos = pos
    self._quat = quat
    self._body = body

    # Save last observation
    self._last_obs = None

    # Define a vision buffer for including previous visual observations
    self._buffer = None
    self._buffer_difference = kwargs.get("use_buffer_difference", False)
    if buffer is not None:
      assert "dt" in kwargs, "dt must be defined in order to include prior observations"
      maxlen = 1 + int(buffer/kwargs["dt"])
      self._buffer = deque(maxlen=maxlen)

    super().__init__(model, data, bm_model, **kwargs)

    # Append the mock-up Unity camera to self._cameras to be able to display
    # their outputs in human-view/GUI mode (used by simulator.py)
    self._cameras.append(self)  #TODO: create a (mock-up) Camera instance that "renders" the rgb array obtained from Unity, instead of pretending that self would be a Camera (see self.render())

  @staticmethod
  def insert(simulation, **kwargs):
    pass

  def get_observation(self, model, data, info=None):

    # The observation is transferred from task through 'info' TODO a smarter architecture?

    if not info:
      obs = np.zeros((self._resolution[1], self._resolution[0], 4))
    else:

      # Get observation
      obs = info["unity_image"]

      # Sometimes the screen hasn't been resized yet when first screenshot arrives
      if obs.shape != (self._resolution[1], self._resolution[0], 4):
        print(f"Resizing from {[obs.shape[1], obs.shape[0]]} to {self._resolution}")
        obs = cv2.resize(obs, dsize=tuple(self._resolution), interpolation=cv2.INTER_CUBIC)

      # Normalise
      obs = (obs / 255.0 - 0.5) * 2

      # Make a copy for rendering purposes
      self._last_obs = info["unity_image"].copy()

    # Transpose channels
    obs = np.transpose(obs, [2, 0, 1])

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
      if self._buffer_difference:
        obs = np.concatenate([self._buffer[0], self._buffer[-1], self._buffer[-1] - self._buffer[0]], axis=0)
      else:
        obs = np.concatenate([self._buffer[0], self._buffer[-1]], axis=0)

    return obs

  def _reset(self, model, data):
    if self._buffer is not None:
      self._buffer.clear()
  
  @property
  def _default_encoder(self):
    return {"module": "rl.encoders", "cls": "SmallCNN", "kwargs": {"out_features": 256}}
  
  # @property
  # def encoder(self):
  #   return small_cnn(observation_shape=self._observation_shape, out_features=256)

  def render(self):
    # Return only rgb channels
    # NOTE: need to return (rgb_image, depth_image) to keep the format of utils.rendering.Camera.render()
    if self._last_obs is not None:
      rgb_image = self._last_obs[:,:,:3].copy()
    else:
      rgb_image = None
    depth_image = None

    return rgb_image, depth_image