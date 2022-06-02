import numpy as np

from .base import BaseEffortModel


class CumulativeFatigue(BaseEffortModel):

  # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles -- using control signals
  # instead of torque or force
  def __init__(self, bm_model, dt, **kwargs):
    super().__init__(bm_model)
    self._r = 7.5
    self._F = 0.0146
    self._R = 0.0022
    self._LD = 10
    self._LR = 10
    self._MA = None
    self._MR = None
    self._weight = 0.01
    self._dt = dt

  def cost(self, model, data):

    # Initialise MA if not yet initialised
    if self._MA is None:
      self._MA = np.zeros((model.na,))
      self._MR = np.ones((model.na,))

    # Get target load
    TL = data.act.copy()

    # Calculate C(t)
    C = np.zeros_like(self._MA)
    idxs = (self._MA < TL) & (self._MR > (TL - self._MA))
    C[idxs] = self._LD * (TL[idxs] - self._MA[idxs])
    idxs = (self._MA < TL) & (self._MR <= (TL - self._MA))
    C[idxs] = self._LD * self._MR[idxs]
    idxs = self._MA >= TL
    C[idxs] = self._LR * (TL[idxs] - self._MA[idxs])

    # Calculate rR
    rR = np.zeros_like(self._MA)
    idxs = self._MA >= TL
    rR[idxs] = self._r*self._R
    idxs = self._MA < TL
    rR[idxs] = self._R

    # Calculate MA, MR
    self._MA += (C - self._F*self._MA)*self._dt
    self._MR += (-C + rR*self._MR)*self._dt

    # Not sure if these are needed
    self._MA = np.clip(self._MA, 0, 1)
    self._MR = np.clip(self._MR, 0, 1)

    # Calculate effort
    effort = np.linalg.norm(self._MA - TL)

    return self._weight*effort

  def reset(self):
    self._MA = None
    self._MR = None

  def update(self):
    pass