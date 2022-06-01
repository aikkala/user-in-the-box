import numpy as np

from .base import BaseEffortModel


class Neural(BaseEffortModel):

  def __init__(self, bm_model, weight=1e-4, **kwargs):
    super().__init__(bm_model)
    self._weight = weight

  def cost(self, model, data):
    return self._weight * np.sum(data.ctrl ** 2)

  def reset(self):
    pass