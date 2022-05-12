import mujoco
import numpy as np
import cv2


class Viewer:

  # This class still needs some work. Perhaps a better idea would be to use dm_control functionality if it is
  # available / supported, cf. https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/engine.py

  def __init__(self, model, data, camera_name, resolution, dt=None):

    self.model = model
    self.data = data
    self.resolution = resolution

    # Estimate frames per second based on dt
    self.fps = None
    if dt is not None:
      self.set_fps(dt)

    # Create GL context and make it current
    self.gl = mujoco.GLContext(*resolution)
    self.gl.make_current()

    # Update resolution (really only needed when resolution is bigger than [640,480]); these need only be set until
    # mujoco.MjrContext is created, and can be then overwritten (by e.g. a visual perception module)
    self.model.vis.global_.offwidth = self.resolution[0]
    self.model.vis.global_.offheight = self.resolution[1]

    # Use mujoco python bindings to create a viewer (scene, camera, context, viewport, etc)
    self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
    self.camera = mujoco.MjvCamera()
    self.voptions = mujoco.MjvOption()
    self.perturb = mujoco.MjvPerturb()
    self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.context)
    self.viewport = mujoco.MjrRect(left=0, bottom=0, width=resolution[0], height=resolution[1])
    self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    self.camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

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

    # Initialise rgb array
    rgb_arr = np.zeros(3 * self.viewport.width * self.viewport.height, dtype=np.uint8)

    # Read pixels into arrays
    mujoco.mjr_readPixels(rgb_arr, None, self.viewport, self.context)

    # Reshape and flip
    rgb_img = np.flipud(rgb_arr.reshape(self.viewport.height, self.viewport.width, 3))

    return rgb_img

  def set_fps(self, dt):
    self.fps = int(np.round(1.0 / dt))

  def write_video(self, imgs, filepath):

    # Make sure fps has been set
    assert self.fps is not None, "Frames per second not set, cannot record video"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, self.fps, tuple(self.resolution))
    for img in imgs:
      out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()