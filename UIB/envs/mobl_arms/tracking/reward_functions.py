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

  def __init__(self, k):
    self.k = k

  def get(self, env, dist, info):
    if info["inside_target"]:
      return 1
    else:
      if callable(self.k):
        k = self.k()
      else:
        k = self.k
      return np.exp(-dist*k) - 1

  def __repr__(self):
    return "NegativeExpDistanceWithHitBonus"
