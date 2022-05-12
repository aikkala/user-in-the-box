import xml.etree.ElementTree as ET
import numpy as np
import os
import mujoco
import collections
from collections import deque

from uitb.perception.base import BaseModule
from uitb.utils.functions import parent_path
from ..extractors import small_cnn

class FixedEye(BaseModule):

  def __init__(self, model, data, bm_model, resolution, pos, quat, body="worldbody", channels=None, buffer=None,
               **kwargs):
    """
    A simple eye model using a fixed camera.

    Args:
        model: A MjModel object of the simulation
        data: A MjData object of the simulation
        bm_model: A biomechanical model class object inheriting from BaseBMModel
        resolution: Resolution in pixels [width, height]
        pos: Position of the camera [x, y, z]
        quat: Orientation of the camera as a quaternion [w, x, y, z]
        body (optional): Body to which the camera is attached, default is 'worldbody'
        channels (optional): Which channels to use; 0-2 refer to RGB, 3 is depth. Default value is None, which means that all channels are used (i.e. same as channels=[0,1,2,3])
        buffer (optional): Defines a buffer of given length (in seconds) that is utilized to include prior observations
        **kwargs (optional): Keyword args that may be used
    """

    self.model = model
    self.data = data

    # Probably already called
    mujoco.mj_forward(self.model, self.data)

    # Set camera specs
    if channels is None:
      channels = [0, 1, 2, 3]
    self.channels = channels
    self.resolution = resolution
    self.pos = pos
    self.quat = quat
    self.body = body

    # Initialise camera
    self.gl = mujoco.GLContext(*self.resolution)
    self.gl.make_current()
    self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
    self.camera = mujoco.MjvCamera()
    self.voptions = mujoco.MjvOption()
    self.perturb = mujoco.MjvPerturb()
    self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.context)
    self.viewport = mujoco.MjrRect(left=0, bottom=0, width=self.resolution[0], height=self.resolution[1])
    self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    self.camera.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'fixed-eye')

    # Define a vision buffer for including previous visual observations
    if buffer is None:
      self.buffer = None
    else:
      assert "dt" in kwargs, "dt must be defined in order to include prior observations"
      maxlen = 1 + int(buffer/kwargs["dt"])
      self.buffer = deque(maxlen=maxlen)

    super().__init__(model, data, bm_model, **kwargs)
    self.module_folder = parent_path(__file__)

  def render(self):

    # Update scene
    mujoco.mjv_updateScene(
      self.model,
      self.data,
      self.voptions,
      self.perturb,
      self.camera,
      mujoco.mjtCatBit.mjCAT_ALL,
      self.scene,
    )

    # Render
    mujoco.mjr_render(self.viewport, self.scene, self.context)

    # Initialise rgb and depth arrays
    rgb_arr = np.zeros(3 * self.viewport.width * self.viewport.height, dtype=np.uint8)
    depth_arr = np.zeros(self.viewport.width * self.viewport.height, dtype=np.float32)

    # Read pixels into arrays
    mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.context)

    # Reshape and flip
    rgb_img = np.flipud(rgb_arr.reshape(self.viewport.height, self.viewport.width, 3))
    depth_img = np.flipud(depth_arr.reshape(self.viewport.height, self.viewport.width))

    return rgb_img, depth_img

  @staticmethod
  def insert(task, config, **kwargs):

    assert "pos" in kwargs, "pos needs to be defined for this perception module"
    assert "quat" in kwargs, "quat needs to be defined for this perception module"

    # Get task root
    task_root = task.getroot()

    # Add assets
    task_root.find("asset").append(ET.Element("mesh", name="eye", scale="0.05 0.05 0.05",
                                              file="assets/basic_eye_2.stl"))
    task_root.find("asset").append(ET.Element("texture", name="blue-eye", type="cube", gridsize="3 4",
                                              gridlayout=".U..LFRB.D..",
                                              file="assets/blue_eye_texture_circle.png"))
    task_root.find("asset").append(ET.Element("material", name="blue-eye", texture="blue-eye", texuniform="true"))

    # Create eye
    eye = ET.Element("body", name="fixed-eye", pos=kwargs["pos"], quat=kwargs["quat"])
    eye.append(ET.Element("geom", name="fixed-eye", type="mesh", mesh="eye", euler="0.69 1.43 0",
                          material="blue-eye", size="0.025"))
    eye.append(ET.Element("camera", name="fixed-eye", fovy="90"))

    # Add eye to a body
    body = kwargs.get("body", "worldbody")
    if body == "worldbody":
      task_root.find("worldbody").append(eye)
    else:
      eye_body = task_root.find(f".//body[@name='{body}'")
      assert eye_body is not None, f"Body with name {body} was not found"
      eye_body.append(eye)

  def get_observation(self, model, data):

    # Get rgb and depth arrays
    rgb, depth = self.render()
    assert not np.all(rgb==0), "There's still something wrong with rendering"

    # Normalise
    depth = (depth - 0.5) * 2
    rgb = (rgb / 255.0 - 0.5) * 2

    # Transpose channels
    obs = np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), [2, 0, 1])

    # Choose channels
    obs = obs[self.channels, :, :]

    # Include prior observation if needed
    if self.buffer is not None:
      # Update buffer
      if len(self.buffer) > 0:
        self.buffer.pop()
      while len(self.buffer) < self.buffer.maxlen:
        self.buffer.appendleft(obs)

      # Use latest and oldest observation, and their difference
      obs = np.concatenate([self.buffer[0], self.buffer[-1], self.buffer[-1] - self.buffer[0]], axis=2)

    return obs

  def get_observation_space_params(self):
    return {"low": -1, "high": 1, "shape": self.observation_shape}

  def reset(self, model, data):
    if self.buffer is not None:
      self.buffer.clear()

  def extractor(self):
    return small_cnn(observation_shape=self.observation_shape, out_features=256)
