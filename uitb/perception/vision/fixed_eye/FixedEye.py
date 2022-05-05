import xml.etree.ElementTree as ET
import numpy as np
import os
import mujoco
import collections

from uitb.perception.base import BaseModule
from uitb.utils.functions import parent_path
from ..extractors import small_cnn

try:
  from gym.envs.mujoco.mujoco_rendering import Viewer, RenderContextOffscreen
except ImportError:
  from uitb.mujoco_rendering import Viewer, RenderContextOffscreen


# def _import_egl(width, height):
#   from mujoco.egl import GLContext
#
#   return GLContext(width, height)
#
#
# def _import_glfw(width, height):
#   from mujoco.glfw import GLContext
#
#   return GLContext(width, height)
#
#
# def _import_osmesa(width, height):
#   from mujoco.osmesa import GLContext
#
#   return GLContext(width, height)
#
#
# _ALL_RENDERERS = collections.OrderedDict(
#     [
#         ("glfw", _import_glfw),
#         ("egl", _import_egl),
#         ("osmesa", _import_osmesa),
#     ]
# )
#
# def _get_opengl_backend(width, height):
#   backend = os.environ.get("MUJOCO_GL")
#   if backend is not None:
#     try:
#       opengl_context = _ALL_RENDERERS[backend](width, height)
#     except KeyError:
#       raise RuntimeError(
#         "Environment variable {} must be one of {!r}: got {!r}.".format(
#           "MUJOCO_GL", _ALL_RENDERERS.keys(), backend
#         )
#       )
#
#   else:
#     for name, import_func in _ALL_RENDERERS.items():
#       try:
#         opengl_context = _ALL_RENDERERS["osmesa"](width, height)
#         backend = name
#         break
#       except:
#         pass
#     if backend is None:
#       raise RuntimeError(
#         "No OpenGL backend could be imported. Attempting to create a "
#         "rendering context will result in a RuntimeError."
#       )
#
#   return opengl_context

class FixedEye(BaseModule):

  def __init__(self, model, data, bm_model, resolution, pos, quat, body="worldbody", channels=None, **kwargs):
    if channels is None:
      channels = [0, 1, 2, 3]
    self.channels = channels
    self._viewers = {}
    self.resolution = resolution
    self.pos = pos
    self.quat = quat
    self.body = body
    super().__init__(model, data, bm_model)
    self.module_folder = parent_path(__file__)

    # TODO keyword for using prior observations

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
    render = self.render(model=model, data=data, mode="depth_array", width=self.resolution[0], height=self.resolution[1], camera_name='fixed-eye')

    # Normalise
    depth = render[1]
    depth = np.flipud((depth - 0.5) * 2)
    rgb = render[0]
    rgb = np.flipud((rgb / 255.0 - 0.5) * 2)

    # Transpose channels
    obs = np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), [2, 0, 1])

    return obs[self.channels, :, :]

  # def get_observation(self, model, data):
  #
  #   # Get rgb and depth arrays
  #
  #   # render = sim.render(width=self.resolution[0], height=self.resolution[1], camera_name='fixed-eye',
  #   #                     depth=True)
  #
  #   #TODO
  #   # ctx = mujoco.GLContext(self.resolution[0], self.resolution[1])
  #   # ctx.make_current()
  #   # ###
  #   # with _MjSim_render_lock:
  #   #   if self._render_context_offscreen is None:
  #   #     render_context = MjRenderContextOffscreen(
  #   #       self, device_id=-1)
  #   #   else:
  #   #     render_context = self._render_context_offscreen
  #   # render_context.render(width=self.resolution[0], height=self.resolution[1], camera_id=model.camera_name2id('fixed-eye'), segmentation=False)
  #   # render = render_context.read_pixels(width, height, depth=True, segmentation=False)
  #
  #   self.camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'fixed-eye')
  #   mujoco.mjv_updateScene(
  #     model, data, mujoco.MjvOption(), None,
  #     self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
  #   mujoco.mjr_render(self.viewport, self.scene, self.context)
  #
  #   # TODO: compare to https://github.com/rodrigodelazcano/gym/blob/906789e0fd5b11e3c7979065e091a1abc00d1b35/gym/envs/mujoco/mujoco_rendering.py
  #   #  (def render() and def read_pixels())
  #
  #   mujoco.mjr_rectangle(self.viewport, 0, 0, 0, 1)
  #   rgb_arr = np.zeros(3 * self.viewport.width * self.viewport.height, dtype=np.uint8)
  #   depth_arr = np.zeros(self.viewport.width * self.viewport.height, dtype=np.float32)
  #   mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.context)
  #   rgb = rgb_arr.reshape(self.viewport.height, self.viewport.width, 3)
  #   depth = depth_arr.reshape(self.viewport.height, self.viewport.width)
  #
  #   # Normalise
  #   #depth = render[1]
  #   depth = np.flipud((depth - 0.5) * 2)
  #   #rgb = render[0]
  #   rgb = np.flipud((rgb / 255.0 - 0.5) * 2)
  #
  #   return np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), [2, 0, 1])

  def get_observation_space_params(self):
    return {"low": -1, "high": 1, "shape": self.observation_shape}

  def reset(self, model, data, rng):
    # TODO reset once prior observations are implemented
    pass

  def extractor(self):
    return small_cnn(observation_shape=self.observation_shape, out_features=256)

  def render(self, model, data, mode='depth_array', width=1280, height=800, camera_id=None, camera_name=None, preprocess_output=False):

    if mode == 'rgb_array' or mode == 'depth_array':
        if camera_id is not None and camera_name is not None:
            raise ValueError("Both `camera_id` and `camera_name` cannot be"
                             " specified at the same time.")

        no_camera_specified = camera_name is None and camera_id is None
        if no_camera_specified:
            camera_name = 'track'

        if camera_id is None:
          camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

          self._get_viewer(model, data, mode).render(width, height, camera_id=camera_id)

    if mode == 'rgb_array':
        data = self._get_viewer(model, data, mode).read_pixels(width, height, depth=False)
        if preprocess_output:
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        else:
            return data
    elif mode == 'depth_array':
        self._get_viewer(model, data, mode).render(width, height)
        data = self._get_viewer(model, data, mode).read_pixels(width, height, depth=True)
        if preprocess_output:
            # Extract depth part of the read_pixels() tuple; original image is upside-down, so flip it
            return data[1][::-1, :]
        else:
            return data
    elif mode == 'human':
        self._get_viewer(model, data, mode).render()

  def _get_viewer(self, model, data, mode, width=1280, height=800):
    self.viewer = self._viewers.get(mode)
    if self.viewer is None:
      if mode == 'human':
        self.viewer = Viewer(model, data)
      elif mode == 'rgb_array' or mode == 'depth_array':
        self.viewer = RenderContextOffscreen(width, height, model, data)
      self._viewers[mode] = self.viewer
    return self.viewer