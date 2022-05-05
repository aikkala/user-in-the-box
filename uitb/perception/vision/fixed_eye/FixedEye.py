import xml.etree.ElementTree as ET
import numpy as np
import os

from uitb.perception.base import BaseModule
from uitb.utils.functions import parent_path
from ..extractors import small_cnn

class FixedEye(BaseModule):

  def __init__(self, sim, bm_model, resolution, pos, quat, body="worldbody", channels=None, **kwargs):
    if channels is None:
      channels = [0, 1, 2, 3]
    self.channels = channels
    self.resolution = resolution
    self.pos = pos
    self.quat = quat
    self.body = body
    super().__init__(sim, bm_model)
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

  def get_observation(self, sim):

    # Get rgb and depth arrays
    render = sim.render(width=self.resolution[0], height=self.resolution[1], camera_name='fixed-eye',
                        depth=True)

    # Normalise
    depth = render[1]
    depth = np.flipud((depth - 0.5) * 2)
    rgb = render[0]
    rgb = np.flipud((rgb / 255.0 - 0.5) * 2)

    # Transpose channels
    obs = np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), [2, 0, 1])

    return obs[self.channels, :, :]

  def get_observation_space_params(self):
    return {"low": -1, "high": 1, "shape": self.observation_shape}

  def reset(self, sim, rng):
    # TODO reset once prior observations are implemented
    pass

  def extractor(self):
    return small_cnn(observation_shape=self.observation_shape, out_features=256)