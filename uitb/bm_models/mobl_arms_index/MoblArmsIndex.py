from uitb.bm_models.base import BaseBMModel
from uitb.utils.functions import parent_path

import os
import numpy as np


class MoblArmsIndex(BaseBMModel):

  xml_file = os.path.join(parent_path(__file__), "mobl_arms_index.xml")

  def __init__(self, model, data, **kwargs):
    super().__init__(model, data, **kwargs)

    # Set shoulder variant
    self.shoulder_variant = kwargs.get("shoulder_variant", "patch-v1")

  def update(self, model, data):

    # Update shoulder equality constraints
    if self.shoulder_variant.startswith("patch"):
      model.equality("shoulder1_r2_con").data[1] = \
        -((np.pi - 2 * data.joint('shoulder_elv').qpos) / np.pi)

      if self.shoulder_variant == "patch-v2":
        data.joint('shoulder_rot').range[:] = \
          np.array([-np.pi / 2, np.pi / 9]) - \
          2 * np.min((data.joint('shoulder_elv').qpos,
                      np.pi - data.joint('shoulder_elv').qpos)) / np.pi \
          * data.joint('elv_angle').qpos

  def reset(self, model, data):
    super().reset(model, data)
    self.update(model, data)