from abc import ABC, abstractmethod
import mujoco
import numpy as np


class BaseEffortModel(ABC):

  def __init__(self, bm_model, **kwargs):
    self._bm_model = bm_model

  @abstractmethod
  def cost(self, model, data):
    pass

  @abstractmethod
  def reset(self, model, data):
    pass

  @abstractmethod
  def update(self, model, data):
    # If needed to e.g. reduce max force output
    pass


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

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass


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

  def reset(self, model, data):
    self._MA = None
    self._MR = None

  def update(self, model, data):
    pass


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


class Neural(BaseEffortModel):

  def __init__(self, bm_model, weight=1e-4, **kwargs):
    super().__init__(bm_model)
    self._weight = weight

  def cost(self, model, data):
    return self._weight * np.sum(data.ctrl ** 2)

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass


class Zero(BaseEffortModel):

  def cost(self, model, data):
    return 0

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass