import numpy as np
from abc import ABC, abstractmethod


class BaseFunction(ABC):
  @abstractmethod
  def get(self, env, dist, info):
    pass
  @abstractmethod
  def __repr__(self):
    pass

class ExpDistanceWithHitBonus(BaseFunction):

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 2
    else:
      return np.exp(-dist * 10) / 10

  def __repr__(self):
    return "ExpDistanceWithHitBonus"

class NegativeDistanceWithHitBonus(BaseFunction):

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 5
    else:
      return -dist**2

  def __repr__(self):
    return "NegativeDistanceWithHitBonus"

class PositiveBinary(BaseFunction):

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 1
    else:
      return 0

  def __repr__(self):
    return "PositiveBinary"

class NegativeBinaryWithHitBonus(BaseFunction):

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 10
    else:
      return -1

  def __repr__(self):
    return "NegativeBinary"
