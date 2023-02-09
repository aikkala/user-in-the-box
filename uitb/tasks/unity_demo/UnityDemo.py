import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
import os
import pathlib
import subprocess

from ..base import BaseTask
from ...utils.transformations import transformation_matrix
from ...utils.unity import UnityClient


class UnityDemo(BaseTask):
  # IMPORTANT: This task expects that in Unity Z is forward, Y is up, and X is to the right, and
  # that in MuJoCo Z is up, Y is to the left, and X is forward. If these don't hold, then the
  # transformations will not be correct

  def __init__(self, model, data, left_controller_enabled=False, **kwargs):
    super().__init__(model, data, **kwargs)

    # Fire up the Unity client if we're really starting the simulation and not just building
    if not kwargs.get("build", False):

      # Get path for the application binary/executable file
      app_path = os.path.join(pathlib.Path(__file__).parent, kwargs["unity_executable"])

      # Check if we want to record game play videos
      self.record_options = dict()
      if kwargs.get("unity_record_gameplay", False):
        self.record_options = {"resolution": kwargs.get("unity_record_resolution",
                                                        f"{model.vis.global_.offwidth}x{model.vis.global_.offheight}"),
                               "output_folder": os.path.join(os.path.split(app_path)[0], "output")}

      # Start a Unity client
      self._unity_client = UnityClient(unity_executable=app_path, port=kwargs.get("port", None),
                                       standalone=kwargs.get("standalone", True), record_options=self.record_options)

      # Wait until app is up and running. Send time options to unity app
      time_options = {"timestep": model.opt.timestep, "sampleFrequency": kwargs["action_sample_freq"],
                      "timeScale": kwargs["time_scale"]}
      self.initial_state = self._unity_client.handshake(time_options)

    # This task requires an end-effector to be defined; also, it must be a body
    # Would be nicer to have this check in the "initialise" method of this class, but not currently possible because
    # of the order in which mujoco xml files are merged (task -> bm_model -> perception).
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kwargs["right_controller_body"]) == -1:
      raise KeyError(f"'right_controller_body' must be a body, no body called {kwargs['right_controller_body']} found in the model")

    # Left controller can be disabled or enabled
    self.left_controller_enabled = left_controller_enabled
    if left_controller_enabled:
      if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kwargs["left_controller_body"]) == -1:
        raise KeyError(
          f"'left_controller_body' must be a body, no body called {kwargs['left_controller_body']} found in the model")

    # Check for headset
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kwargs["headset_body"]) == -1:
      raise KeyError(f"'headset_body' must be a body, no body called {kwargs['headset_body']} found in the model")

    # Let's try to keep time in mujoco and unity synced
    self._current_timestep = 0

    # Use early termination if target is not hit in time
    self._max_steps = self._action_sample_freq*20

    # Geom's mass property is not saved when we save the integrated model xml file (would be saved in binary mjcf
    # though). So let's set it here again just to be sure
    model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "controller-right")] = 0.129
    if left_controller_enabled:
      model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "controller-left")] = 0.129
    model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "headset")] = 0.571

    # Do a forward step so body positions are updated
    mujoco.mj_forward(model, data)

    # '*_relpose' is defined with respect to the '*_body'. We need to move the object to the correct position to make
    # sure there aren't any quick movements in the first timesteps when mujoco starts enforcing the equality constraint
    self.initialise_pos_and_quat(model=model, data=data, aux_body=kwargs["right_controller_body"],
                                 relpose=kwargs["right_controller_relpose"], body="controller-right")
    if self.left_controller_enabled:
      self.initialise_pos_and_quat(model=model, data=data, aux_body=kwargs["left_controller_body"],
                                   relpose=kwargs["left_controller_relpose"], body="controller-left")
    self.initialise_pos_and_quat(model=model, data=data, aux_body=kwargs["headset_body"],
                                 relpose=kwargs["headset_relpose"], body="headset")

    # Set camera angle
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([2.2, -1.8, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])

  def initialise_pos_and_quat(self, model, data, aux_body, relpose, body):
    T1 = transformation_matrix(pos=data.body(aux_body).xpos, quat=data.body(aux_body).xquat)
    T2 = transformation_matrix(pos=relpose[:3], quat=relpose[3:])
    T = np.matmul(T1, np.linalg.inv(T2))
    model.body(body).pos = T[:3, 3]
    model.body(body).quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)

  @classmethod
  def initialise(cls, task_kwargs):

    # Make sure body for right controller is defined, as well as the relative position of the controller wrt to it
    if "right_controller_body" not in task_kwargs:
      raise KeyError("Key 'right_controller_body' is missing from task kwargs. The end-effector body must be defined "
                     "for this environment")
    if "right_controller_relpose" not in task_kwargs:
      raise KeyError("Key 'right_controller_relpose' is missing from task kwargs. This key defines the relative pose "
                     "of right controller wrt to end-effector, and it must be defined for this environment")

    if "left_controller_enabled" not in task_kwargs:
      raise KeyError("Key 'left_controller_enabled' is missing from task kwargs.")

    if task_kwargs["left_controller_enabled"]:
      if "left_controller_body" not in task_kwargs:
        raise KeyError("Key 'left_controller_body' is missing from task kwargs. The end-effector body must be defined "
                       "for this environment")
      if "left_controller_relpose" not in task_kwargs:
        raise KeyError("Key 'left_controller_relpose' is missing from task kwargs. This key defines the relative pose "
                       "of left controller wrt to end-effector, and it must be defined for this environment")

    # Make sure body for headset is defined, as well as the relative position of headset wrt to it
    if "headset_body" not in task_kwargs:
      raise KeyError("Key 'headset_body' is missing from task kwargs. The headset body must be defined for this "
                     "environment")
    if "headset_relpose" not in task_kwargs:
      raise KeyError("Key 'headset_relpose' is missing from task kwargs. This key defines the relative pose of "
                     "the headset wrt to a head, and it must be defined for this environment")

    # Parse xml file
    tree = ET.parse(cls.get_xml_file())
    root = tree.getroot()

    # Find equalities
    equality = root.find("equality")
    if equality is None:
      equality = ET.Element("equality")
      root.append(equality)

    # Add a weld equality for right controller
    equality.append(ET.Element("weld",
                               name="controller-right-weld",
                               body1="controller-right",
                               body2=task_kwargs["right_controller_body"],
                               relpose=" ".join([str(x) for x in task_kwargs["right_controller_relpose"]]),
                               active="true"))

    # If left controller is not enabled, remove it from the model; if it is enabled, add a weld equality
    if task_kwargs["left_controller_enabled"]:
      equality.append(ET.Element("weld",
                                 name="controller-left-weld",
                                 body1="controller-left",
                                 body2=task_kwargs["left_controller_body"],
                                 relpose=" ".join([str(x) for x in task_kwargs["left_controller_relpose"]]),
                                 active="true"))
    else:
      worldbody = root.find("worldbody")
      worldbody.remove(worldbody.find("body[@name='controller-left']"))

    # Add a weld equality for headset
    equality.append(ET.Element("weld",
                               name="headset-weld",
                               body1="headset",
                               body2=task_kwargs["headset_body"],
                               relpose=" ".join([str(x) for x in task_kwargs["headset_relpose"]]),
                               active="true"))

    return tree

  def _update(self, model, data):

    info = {}
    is_finished = False

    # Update timestep
    self._current_timestep = data.time

    # Let Unity app know that the episode has terminated
    # if self._steps >= self._max_steps:
    #   is_finished = True
    #   info["termination"] = "max_steps_reached"

    # Send end effector position and rotation to unity, get reward and image from camera
    image, reward, is_app_finished = self._unity_client.step(self._create_state(model, data), is_finished)

    if is_finished and not is_app_finished:
      raise RuntimeError("User simulation has terminated an episode but Unity app has not")

    return reward, is_app_finished, {"unity_observation": image}

  def _transform_to_unity(self, pos, quat, apply_rotation):

    # A couple of rotations to make coordinate axes match with unity. These probably could be simplified
    rot = Rotation.from_quat(np.roll(quat, -1))

    # Get the quaternion; quat is in scalar-last format
    quat = rot.as_quat()

    # Transform from MuJoCo's right hand side coordinate system to Unity's left hand side coordinate system
    pos = {"x": -pos[1], "y": pos[2], "z": pos[0]}
    quat = {"x": -quat[1], "y": quat[2], "z": quat[0], "w": -quat[3]}

    return pos, quat

  def _create_state(self, model, data):

    # Get position and rotation of right controller
    controller_right_pos, controller_right_quat = \
      self._transform_to_unity(data.body("controller-right").xpos, data.body("controller-right").xquat,
                               apply_rotation=True)

    if self.left_controller_enabled:
      controller_left_pos, controller_left_quat = \
        self._transform_to_unity(data.body("controller-left").xpos, data.body("controller-left").xquat,
                               apply_rotation=True)
    else:
      controller_left_pos = {"x": 0, "y": 0, "z": 0}
      controller_left_quat = {"x": 0, "y": 0, "z": 0, "w": 1.0}

    # Get position and rotation of headset
    headset_pos, headset_quat = \
      self._transform_to_unity(data.body("headset").xpos, data.body("headset").xquat, apply_rotation=False)

    # Create the state
    state = {
      "headsetPosition": headset_pos,
      "leftControllerPosition": controller_left_pos,
      "rightControllerPosition": controller_right_pos,
      "headsetRotation": headset_quat,
      "leftControllerRotation": controller_left_quat,
      "rightControllerRotation": controller_right_quat,
      "currentTimestep": self._current_timestep,
      "nextTimestep": self._current_timestep + 1/self._action_sample_freq
    }
    return state

  def _reset(self, model, data):

    # Reset and receive an observation
    image = self._unity_client.reset(self._create_state(model, data))

    # Set timestep
    data.time = self._current_timestep

    return {"unity_observation": image}

  def close(self):

    # Close Unity
    self._unity_client.close()

    # If we were recording, create videos from images
    if self.record_options:

      # There can be several folders with images, loop through them
      for key in os.listdir(self.record_options["output_folder"]):

        maybe_folder = os.path.join(self.record_options["output_folder"], key)

        # Only process folders (there shouldn't be anything else anyways)
        if os.path.isdir(maybe_folder):

          # Create the video
          subprocess.call([
            'ffmpeg',
            '-r', f'{self._action_sample_freq}', '-f', 'image2', '-s', self.record_options["resolution"],
            '-i', f"{os.path.join(maybe_folder, 'image%d.png')}",
            '-vcodec', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', f"{os.path.join(maybe_folder, f'{key}_video.mp4')}"])
