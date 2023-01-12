import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
import os
import pathlib

from ..base import BaseTask
from ...utils.transformations import transformation_matrix
from ...utils.unity import UnityClient


class UnityDemo(BaseTask):

  def __init__(self, model, data, **kwargs):
    super().__init__(model, data, **kwargs)

    # Fire up the Unity client if we're really starting the simulation and not just building
    if not kwargs.get("build", False):

      # Start a Unity client
      app_path = os.path.join(pathlib.Path(__file__).parent, kwargs["unity_executable"])
      self._unity_client = UnityClient(unity_executable=app_path, port=kwargs.get("port", None),
                                       standalone=kwargs.get("standalone", True))

      # Wait until app is up and running. Send time options to unity app
      time_options = {"timestep": model.opt.timestep, "sampleFrequency": kwargs["action_sample_freq"],
                      "timeScale": kwargs["time_scale"]}
      self.initial_state = self._unity_client.handshake(time_options)

    # This task requires an end-effector to be defined; also, it must be a body
    # Would be nicer to have this check in the "initialise" method of this class, but not currently possible because
    # of the order in which mujoco xml files are merged (task -> bm_model -> perception).
    self._right_controller = kwargs["right_controller_body"]
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._right_controller) == -1:
      raise KeyError(f"'right_controller_body' must be a body, no body called {self._right_controller} found in the model")

    # Check for headset
    self._headset = kwargs["headset_body"]
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._headset) == -1:
      raise KeyError(f"'headset_body' must be a body, no body called {self._headset} found in the model")

    # Let's try to keep time in mujoco and unity synced
    self._current_timestep = 0

    # Use early termination if target is not hit in time
    self._max_steps = self._action_sample_freq*20

    # Geom's mass property is not saved when we save the integrated model xml file (would be saved in binary mjcf
    # though). So let's set it here again just to be sure
    model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "controller-right")] = 0.129
    model.body_mass[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "headset")] = 0.571

    # Do a forward step so body positions are updated
    mujoco.mj_forward(model, data)

    # 'right_controller_relpose' is defined with respect to the right controller body. We need to move the controller to
    # the correct position to make sure there aren't any quick movements in the first timesteps when mujoco starts
    # enforcing the equality constraint.
    T1 = transformation_matrix(pos=data.body(self._right_controller).xpos, quat=data.body(self._right_controller).xquat)
    T2 = transformation_matrix(pos=kwargs["right_controller_relpose"][:3], quat=kwargs["right_controller_relpose"][3:])
    T = np.matmul(T1, np.linalg.inv(T2))
    model.body("controller-right").pos = T[:3, 3]
    model.body("controller-right").quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)

    T1 = transformation_matrix(pos=data.body(self._headset).xpos, quat=data.body(self._headset).xquat)
    T2 = transformation_matrix(pos=kwargs["headset_relpose"][:3], quat=kwargs["headset_relpose"][3:])
    T = np.matmul(T1, np.linalg.inv(T2))
    #T = T1
    model.body("headset").pos = T[:3, 3]
    model.body("headset").quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)

    # Set camera angle
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array([2.2, -1.8, 0.95])
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'for_testing')] = np.array(
      [0.6582, 0.6577, 0.2590, 0.2588])

    # Need to set up some transforms from mujoco to unity. The exact specifications of these transforms remain a bit
    # of a mystery. They might be different for different biomechanical models (depending on in which orientation the
    # axis of the root body is in)
    self.rotation1 = Rotation.from_euler("xyz", np.array([90, 0, 180]), degrees=True)
    self.rotation2 = Rotation.from_euler("z", 90, degrees=True)

  @classmethod
  def initialise(cls, task_kwargs):

    # Make sure body for right controller is defined, as well as the relative position of the controller wrt to it
    if "right_controller_body" not in task_kwargs:
      raise KeyError("Key 'right_controller_body' is missing from task kwargs. The end-effector body must be defined "
                     "for this environment")
    right_controller = task_kwargs["right_controller_body"]
    if "right_controller_relpose" not in task_kwargs:
      raise KeyError("Key 'right_controller_relpose' is missing from task kwargs. This key defines the relative pose "
                     "of right controller wrt to end-effector, and it must be defined for this environment")

    # Make sure body for headset is defined, as well as the relative position of headset wrt to it
    if "headset_body" not in task_kwargs:
      raise KeyError("Key 'headset_body' is missing from task kwargs. The headset body must be defined for this "
                     "environment")
    headset = task_kwargs["headset_body"]
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

    # Add a weld connect for right controller
    equality.append(ET.Element("weld",
                               name="controller-right-weld",
                               body1="controller-right",
                               body2=right_controller,
                               relpose=" ".join([str(x) for x in task_kwargs["right_controller_relpose"]]),
                               active="true"))

    # Add a weld connect for left controller
    # equality.append(ET.Element("weld",
    #                            name="controller-right-weld",
    #                            body1="controller-right",
    #                            body2=right_controller,
    #                            relpose=" ".join([str(x) for x in task_kwargs["right_controller_relpose"]]),
    #                            active="true"))

    # Add a weld connect for headset
    equality.append(ET.Element("weld",
                               name="headset-weld",
                               body1="headset",
                               body2=headset,
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

  def _transform_to_unity(self, pos, quat):

    # A couple of rotations to make coordinate axes match with unity. These probably could be simplified
    rot = Rotation.from_quat(np.roll(quat, -1))
    rot = self.rotation2*rot*self.rotation1

    # Need to rotate the position as well to match coordinates
    pos = self.rotation2.apply(pos)

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
    headset_pos, headset_quat = \
      self._transform_to_unity(data.body("thorax").xpos+np.array([0.15, 0, 0.22]), data.body("thorax").xquat)

    # Create the state
    state = {
      "headsetPosition": headset_pos,
      "leftControllerPosition": {"x": 0, "y": 0, "z": 0},
      "rightControllerPosition": controller_right_pos,
      "headsetRotation": {"x": 0.1305262, "y": 0, "z": 0, "w": 0.9914449},#headset_quat, # tilt head down by 15 degrees
      "leftControllerRotation": {"x": 0, "y": 0, "z": 0, "w": 1.0},
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
    self._unity_client.close()