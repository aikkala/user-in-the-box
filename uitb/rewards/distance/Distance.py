from abc import ABC, abstractmethod
import numbers
import mujoco
import numpy as np
from ..base_component import BaseRewardComponent

class DistanceRewardComponent(BaseRewardComponent):

  def __init__(self, model, data, bm_model, perception_modules, task, run_parameters,
               end_effector, target=None, target_size=None,
               condition_fct=lambda model, data: True, condition_fallback=0,  #set condition_fallback to negative value?
               norm=None, squared_norm=True,
               weight=1, **kwargs):
    """

    :param model:
    :param data:
    :param bm_model:
    :param perception_modules:
    :param task:
    :param run_parameters:
    :param end_effector: list [<data-object>, <attribute-name>] or string <attribute-name> (<data-object>=geom is used then);
      if <data-object> is a MuJoCo name category (e.g., "body", "geom", site") and data.<data-object>(<attribute-name>).xpos is used,
      otherwise, <data-object> is assumed to be an attribute of self and self.<data-object>.<attribute-name> is used
    :param target: see "end_effector"
    :param target_size:
    :param condition_fct:
    :param condition_fallback:
    :param norm:
    :param squared_norm:
    :param weight:
    :param kwargs:
    """
    super().__init__(model, data, bm_model, perception_modules, task, run_parameters, **kwargs)

    if target is None:
      # set target position always to zero -> allows to use distance already stored in end_effector attribute, e.g., in task module
      self._target = ["_zero_dummy", "_zero_dummy"]
      self._target_pos = 0
    else:
      self._target = ["geom", target] if isinstance(target, str) else target
      self._target_pos = getattr(data, self._target[0])(self._target[1]).xpos if hasattr(data, self._target[0]) else getattr(getattr(self, self._target[0]), self._target[1])

    self._end_effector = ["geom", end_effector] if isinstance(end_effector, str) else end_effector
    self._end_effector_pos = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos if hasattr(data, self._end_effector[0]) else getattr(getattr(self, self._end_effector[0]), self._end_effector[1])


    if target_size is None:
      try:  #e.g., if target <data-object> is "geom" or "site"
        self._target_size = getattr(data, self._target[0])(self._target[1]).size
      except:
        self._target_size = 0
    else:
      self._target_size = target_size

    self._condition_fct = condition_fct
    self._condition_fallback = condition_fallback
    self._norm = norm
    self._squared_norm = squared_norm
    self._weight = weight

    if self._norm is not None and np.abs(self._norm) not in (0, 1, 2, np.inf):
      raise NotImplementedError("This norm is not available yet.")

    self._name = f"Distance_<{self._end_effector[0]}>{self._end_effector[1]}-<{self._target[0]}>{self._target[1]}"

    assert isinstance(self._weight, numbers.Number) and self._weight >= 0, f"'weight' should be a non-negative number but is {self._weight}."

  def reward(self):
    if self._condition_fct(self.model, self.data):
      distance = np.linalg.norm(self._end_effector_pos - self._target_pos, ord=self._norm)
      return self._weight * np.max((distance, self._target_size)) ** (2 if self._squared_norm else 1)
    else:
      return self._condition_fallback

  #TODO: log history of cost calls?