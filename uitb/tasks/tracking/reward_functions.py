from abc import ABC, abstractmethod
import numpy as np

class BaseFunction(ABC):
  @abstractmethod
  def get(self, env, dist, info):
    pass
  @abstractmethod
  def __repr__(self):
    pass


class ExpDistanceWithHitBonus(BaseFunction):

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 1
    else:
      return np.exp(-dist * 10) / 10

  def __repr__(self):
    return "ExpDistanceWithHitBonus"


class NegativeExpDistanceWithHitBonus(BaseFunction):

  def __init__(self, k=3.0, scale=1.0, bonus=1.0):
    self.k = k
    self.scale = scale
    self.bonus = bonus

  def get(self, env, dist, info):
    if info["inside_target"]:
      return self.bonus
    else:
      if callable(self.k):
        k = self.k()
      else:
        k = self.k
      return self.scale*(np.exp(-dist*k) - 1)

  def __repr__(self):
    return "NegativeExpDistanceWithHitBonus"


class NegativeDistance(BaseFunction):

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 0
    else:
      return -dist

  def __repr__(self):
    return "NegativeDistance"


class NegativeDistanceWithHitBonus(BaseFunction):

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 1
    else:
      return -dist

  def __repr__(self):
    return "NegativeDistanceWithHitBonus"
