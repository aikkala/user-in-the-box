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
      return 10
    elif info["inside_target"]:
      return 0
    else:
      # Negative distance to target sphere surface squared
      return -(dist-env.target_radius)**2

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


class NegativeExpDistance(BaseFunction):

  def __init__(self, k):
    self.k = k

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 8
    elif info["inside_target"]:
      return 0
    else:
      if callable(self.k):
        k = self.k()
      else:
        k = self.k
      return (np.exp(-dist*k) - 1)/10

  def __repr__(self):
    return "ExpDistance"
