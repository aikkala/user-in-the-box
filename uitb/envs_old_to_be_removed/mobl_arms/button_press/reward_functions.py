from abc import ABC, abstractmethod
import numpy as np


class BaseFunction(ABC):
  @abstractmethod
  def get(self, env, dist, info):
    pass
  @abstractmethod
  def __repr__(self):
    pass


class NegativeExpDistanceWithHitBonus(BaseFunction):

  def __init__(self, k=10):
    self.k = k

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 8
    else:
      return (np.exp(-dist*self.k) - 1)/10

  def __repr__(self):
    return "NegativeExpDistanceWithHitBonus"