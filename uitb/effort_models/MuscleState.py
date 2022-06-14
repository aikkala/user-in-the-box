import numpy as np

from .base import BaseEffortModel


class MuscleState(BaseEffortModel):

  def __init__(self, bm_model, weight=1e-4, **kwargs):
    super().__init__(bm_model)
    self._weight = weight

  def cost(self, model, data):
    return self._weight * np.sum(data.act ** 2)

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass