import mujoco
import numpy as np

from .base import BaseEffortModel


class Composite(BaseEffortModel):

  def __init__(self, bm_model, weight=1e-7, **kwargs):
    super().__init__(bm_model)
    self._weight = weight

  def cost(self, model, data):
    mujoco.cymj._mj_inverse(model, data) # TODO does this work with new mujoco python bindings?
    angle_acceleration = np.sum(data.qacc[self._bm_model.independent_joints] ** 2)
    energy = np.sum(data.qvel[self._bm_model.independent_joints] ** 2
                    * data.qfrc_inverse[self._bm_model.independent_joints] ** 2)
    return self._weight * (energy + 0.05 * angle_acceleration)

  def reset(self):
    pass

  def update(self):
    pass