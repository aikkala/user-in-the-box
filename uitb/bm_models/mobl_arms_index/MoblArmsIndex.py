from uitb.bm_models.base import BaseBMModel
from uitb.utils.functions import parent_path

import os
import numpy as np


class MoblArmsIndex(BaseBMModel):

  xml_file = os.path.join(parent_path(__file__), "mobl_arms_index.xml")

  def __init__(self, sim, **kwargs):
    super().__init__(sim, **kwargs)

    # Set shoulder variant
    self.shoulder_variant = kwargs.get("shoulder_variant", "patch-v1")

    # Get equality constraint ID of shoulder1_r2
    if self.shoulder_variant.startswith("patch"):
      cond1 = sim.model.eq_type==2
      cond2 = sim.model.eq_obj1id==sim.model.joint_name2id("shoulder1_r2")
      cond3 = sim.model.eq_obj2id==sim.model.joint_name2id("elv_angle")
      eq_ID_shoulder1_r2 = np.where(cond1&cond2&cond3)[0]
      assert eq_ID_shoulder1_r2.size == 1, "Couldn't find the equality constraint"
      self.eq_ID_shoulder1_r2 = eq_ID_shoulder1_r2[0]

  def update(self, sim):

    # Update shoulder equality constraints
    if self.shoulder_variant.startswith("patch"):
      sim.model.eq_data[self.eq_ID_shoulder1_r2, 1] = \
        -((np.pi - 2 * sim.data.qpos[sim.model.joint_name2id('shoulder_elv')]) / np.pi)

      if self.shoulder_variant == "patch-v2":
        sim.model.jnt_range[sim.model.joint_name2id('shoulder_rot'), :] = \
          np.array([-np.pi / 2, np.pi / 9]) - \
          2 * np.min((sim.data.qpos[sim.model.joint_name2id('shoulder_elv')],
                      np.pi - sim.data.qpos[sim.model.joint_name2id('shoulder_elv')])) / np.pi \
          * sim.data.qpos[sim.model.joint_name2id('elv_angle')]