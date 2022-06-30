from abc import ABC, abstractmethod
import numbers
import mujoco
import numpy as np
from ..base_component import BaseRewardComponent

class EffortRewardComponent(BaseRewardComponent):

  def __init__(self, model, data, bm_model, perception_modules, task, run_parameters,
               object_name="ctrl",
               condition_fct=lambda model, data: True, condition_fallback=0,  #set condition_fallback to negative value?
               norm=2, squared_norm=True,
               weight=1, **kwargs):
    super().__init__(model, data, bm_model, perception_modules, task, run_parameters, **kwargs)

    self._object_name = object_name
    #TODO: allow to pass list of joints/actuators
    self._quantity = getattr(data, self._object_name) #TODO: allow to make use of bm_model, task, etc. as well (see Distance.__init__())
    self._condition_fct = condition_fct
    self._condition_fallback = condition_fallback
    self._norm = norm
    self._squared_norm = squared_norm
    self._weight = weight

    if np.abs(self._norm) not in (0, 1, 2, np.inf):
      raise NotImplementedError("This norm is not available yet.")

    assert isinstance(self._weight, numbers.Number) and self._weight >= 0, f"'weight' should be a non-negative number but is {self._weight}."

    self._name = f"Effort_{self._object_name}"

  def reward(self):
    if self._condition_fct(self.model, self.data):
      return - self._weight * np.linalg.norm(self._quantity, ord=self._norm) ** (2 if self._squared_norm else 1)
    else:
      return self._condition_fallback

  #TODO: log history of received rewards?


class Composite(BaseRewardComponent):

  def __init__(self, model, data, bm_model, perception_modules, task, run_parameters, weight=1e-7, **kwargs):
    super().__init__(model, data, bm_model, perception_modules, task, run_parameters, **kwargs)
    self._weight = weight

  def reward(self, model, data):
    mujoco.cymj._mj_inverse(model, data) # TODO does this work with new mujoco python bindings?
    angle_acceleration = np.sum(data.qacc[self.bm_model.independent_joints] ** 2)
    energy = np.sum(data.qvel[self.bm_model.independent_joints] ** 2
                    * data.qfrc_inverse[self.bm_model.independent_joints] ** 2)
    return self._weight * (energy + 0.05 * angle_acceleration)

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass


class CumulativeFatigue(BaseRewardComponent):

  # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles -- using control signals
  # instead of torque or force
  def __init__(self, model, data, bm_model, perception_modules, task, run_parameters, dt, **kwargs):
    super().__init__(model, data, bm_model, perception_modules, task, run_parameters, **kwargs)
    self._r = 7.5
    self._F = 0.0146
    self._R = 0.0022
    self._LD = 10
    self._LR = 10
    self._MA = None
    self._MR = None
    self._weight = 0.01
    self._dt = dt

  def reward(self, model, data):

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


class MuscleState(BaseRewardComponent):

  def __init__(self, model, data, bm_model, perception_modules, task, run_parameters, weight=1e-4, **kwargs):
    super().__init__(model, data, bm_model, perception_modules, task, run_parameters, **kwargs)
    self._weight = weight

  def reward(self, model, data):
    return self._weight * np.sum(data.act ** 2)

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass


class Neural(BaseRewardComponent):

  def __init__(self, model, data, bm_model, perception_modules, task, run_parameters, weight=1e-4, **kwargs):
    super().__init__(model, data, bm_model, perception_modules, task, run_parameters, **kwargs)
    self._weight = weight

  def reward(self, model, data):
    return self._weight * np.sum(data.ctrl ** 2)

  def reset(self, model, data):
    pass

  def update(self, model, data):
    pass
