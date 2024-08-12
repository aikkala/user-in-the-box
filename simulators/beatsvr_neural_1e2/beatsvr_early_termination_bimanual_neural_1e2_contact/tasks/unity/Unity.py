import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
import os
import pathlib
import json
import wandb


from ..base import BaseTask
from ...utils.transformations import transformation_matrix
from ...utils.unity import UnityClient, images_to_video
from ...utils.functions import initialise_pos_and_quat


class Unity(BaseTask):
  # IMPORTANT: This task expects that in Unity Z is forward, Y is up, and X is to the right, and
  # that in MuJoCo Z is up, Y is to the left, and X is forward. If these don't hold, then the
  # transformations will not be correct

  def __init__(self, model, data, gear, left_controller_enabled=False, **kwargs):
    super().__init__(model, data, **kwargs)

    # Fire up the Unity client if we're really starting the simulation and not just building
    if not kwargs.get("build", False):

      # Get path for the application binary/executable file
      app_path = os.path.join(pathlib.Path(__file__).parent, kwargs["unity_executable"])

      # Parse app args
      app_args = kwargs.get("app_args", [])

      # Check for output folder
      self._output_folder = None
      if "unity_output_folder" in kwargs:
        self._output_folder = kwargs["unity_output_folder"]
        app_args.extend(["-outputFolder", self._output_folder])

      # Check for recording and resolution
      self._record = kwargs.get("unity_record_gameplay", False)
      self._resolution = kwargs.get("unity_record_resolution",
                                    f"{model.vis.global_.offwidth}x{model.vis.global_.offheight}")
      if self._record:
        app_args.extend(["-record", "-resolution", self._resolution])

      # Check for logging
      self._logging = kwargs.get("unity_logging", False)
      if self._logging:
        app_args.extend(["-logging"])

      # Check if we want to set the random seed
      if "unity_random_seed" in kwargs:
        app_args.extend(["-fixedSeed", f"{kwargs['unity_random_seed']}"])

      # Start a Unity client
      self._standalone = kwargs.get("standalone", True)
      self._unity_client = UnityClient(unity_executable=app_path,
                                       port=kwargs.get("port", None),
                                       standalone=self._standalone,
                                       app_args=app_args)

      # Wait until app is up and running. Send time options to unity app
      time_options = {"timestep": model.opt.timestep, "sampleFrequency": kwargs["action_sample_freq"],
                      "timeScale": kwargs["time_scale"]}
      self.initial_state = self._unity_client.handshake(time_options)

      # Used for logging states
      self._info = {"terminated": False,
                  "truncated": False, "unity_image": None}

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
    self._max_steps = self._action_sample_freq*kwargs.get("max_episode_length_seconds", 1000)

    # Unity returns a time feature to indicate how much of episode is left, scaled [-1, 1]
    self._time = -1

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
    initialise_pos_and_quat(model=model, data=data, aux_body=kwargs["right_controller_body"],
                            relpose=kwargs["right_controller_relpose"], body="controller-right")
    if self.left_controller_enabled:
      initialise_pos_and_quat(model=model, data=data, aux_body=kwargs["left_controller_body"],
                              relpose=kwargs["left_controller_relpose"], body="controller-left")
    initialise_pos_and_quat(model=model, data=data, aux_body=kwargs["headset_body"],
                            relpose=kwargs["headset_relpose"], body="headset")

    # OpenXR has a slight offset that applies to each controller (the origin of the controller is slightly offset
    # in Unity). You probably could use this same for for other VR gears, or just use an identity matrix
    self.offsets = {
      "oculus-quest-1": transformation_matrix(pos=np.array([0.053, 0, 0.002]), quat=np.array([1, 0, 0, 0]),
                                              scalar_first=True)
    }

    # Check if offset has been defined
    self.gear = gear
    if gear not in self.offsets:
      raise NotImplementedError(f"Offset has not been defined for VR gear {gear}")

    # # We may need to override headset orientation in some cases
    # self._override_headset_orientation = kwargs.get("override_headset_orientation", None)

    # Set camera angle
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([1.2, -0.8, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6369657, 0.6364587, 0.3076895, 0.3074446])

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

    # Disable contact between right controller and right_controller_body
    contact = root.find("contact")
    contact.append(ET.Element("exclude",
                              body1="controller-right",
                              body2=task_kwargs["right_controller_body"]))

    # If left controller is not enabled, remove it from the model; if it is enabled, add a weld equality, and
    # disable contact between left controller and left_controller_body
    if task_kwargs["left_controller_enabled"]:
      equality.append(ET.Element("weld",
                                 name="controller-left-weld",
                                 body1="controller-left",
                                 body2=task_kwargs["left_controller_body"],
                                 relpose=" ".join([str(x) for x in task_kwargs["left_controller_relpose"]]),
                                 active="true"))
      contact.append(ET.Element("exclude",
                                body1="controller-left",
                                body2=task_kwargs["left_controller_body"]))
    else:
      worldbody = root.find("worldbody")
      worldbody.remove(worldbody.find("body[@name='controller-left']"))
      contact.remove(contact.find("pair[@name='headset-controller-left']"))
      contact.remove(contact.find("pair[@name='controllers']"))

    # Add a weld equality for headset
    equality.append(ET.Element("weld",
                               name="headset-weld",
                               body1="headset",
                               body2=task_kwargs["headset_body"],
                               relpose=" ".join([str(x) for x in task_kwargs["headset_relpose"]]),
                               active="true"))

    # Disable contact between headset and headset_body
    contact.append(ET.Element("exclude",
                              body1="headset",
                              body2=task_kwargs["headset_body"]))

    return tree

  def _update(self, model, data):

    is_finished = False
    truncated = False  #TODO: should we allow for truncation (using self._max_trials)?

    # Update timestep
    self._current_timestep = data.time

    # Check if episode length has been exceeded
    if self._steps >= self._max_steps:
      is_finished = True
      truncated = True
      self._info["termination"] = "max_steps_reached"

    # Send end effector position and rotation to unity, get reward and image from camera
    obs, reward, is_app_finished, log_dict = self._unity_client.step(self._create_state(model, data), is_finished)
    log_dict = json.loads(log_dict)  #TODO: improve json serializer such that it directly unpacks json dict correctly
    self._time = obs["time"]

    if is_finished and not is_app_finished:
      raise RuntimeError("User simulation has terminated an episode but Unity app has not")

    terminated = is_app_finished

    self._info["terminated"] = terminated
    self._info["truncated"] = truncated
    self._info["unity_image"] = obs["image"]
    # self._info["log_dict"] = log_dict
    self._info.update(log_dict)

    if wandb.run is not None:
      wandb.log(log_dict)

    return reward, terminated, truncated, self._info 

  def get_stateful_information(self, model, data):
    return np.array([self._time])

  def _transform_to_unity(self, pos, quat, apply_offset=False):

    T_controller = transformation_matrix(pos=pos, quat=quat, scalar_first=True)

    # Check if we need to offset the position/rotation
    if apply_offset:
      T_controller = np.matmul(T_controller, self.offsets[self.gear])

    # Get (updated) pos and quat; quat is in scalar-last format
    pos = T_controller[:3, 3]
    quat = Rotation.from_matrix(T_controller[:3, :3]).as_quat()

    # Transform from MuJoCo's right hand side coordinate system to Unity's left hand side coordinate system
    pos = {"x": -pos[1], "y": pos[2], "z": pos[0]}
    quat = {"x": -quat[1], "y": quat[2], "z": quat[0], "w": -quat[3]}

    return pos, quat

  def _create_state(self, model, data):

    # Get position and rotation of right controller
    controller_right_pos, controller_right_quat = \
      self._transform_to_unity(data.body("controller-right").xpos, data.body("controller-right").xquat,
                               apply_offset=True)

    if self.left_controller_enabled:
      controller_left_pos, controller_left_quat = \
        self._transform_to_unity(data.body("controller-left").xpos, data.body("controller-left").xquat,
                                 apply_offset=True)
    else:
      controller_left_pos = {"x": 0, "y": 0, "z": 0}
      controller_left_quat = {"x": 0, "y": 0, "z": 0, "w": 1.0}

    # Get position and rotation of headset
    headset_pos, headset_quat = \
      self._transform_to_unity(data.body("headset").xpos,
                               data.body("headset").xquat) # if self._override_headset_orientation is None else
                              #  self._override_headset_orientation)

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
    obs = self._unity_client.reset(self._create_state(model, data))

    # Set timestep
    data.time = self._current_timestep

    # Reset time feature
    self._time = obs["time"]

    # Used for logging states
    self._info = {"terminated": False,
                "truncated": False, "unity_image": obs["image"]}

    return self._info

  def close(self, evaluate_dir=None):

    # Close Unity
    self._unity_client.close()

    # If we were recording, create videos from images
    if self._record:
      if self._standalone:
        recording_folder = os.path.join(self._output_folder, "recording")
      else:
        return  #TODO: get correct recording_folder (should be "$HOME/.config/unity3d/<company-name>/<application-name>/recording")

      images_to_video(recording_folder, self._action_sample_freq, self._resolution, evaluate_dir)
