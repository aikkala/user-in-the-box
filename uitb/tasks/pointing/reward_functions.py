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

class ExpDistanceWithTimeBonus(BaseFunction):

  def get(self, env, dist, info):
    if info["target_hit"]:
      time_left = 1 - (env.steps_since_last_hit / env.max_steps_without_hit)
      max_reward = env.max_steps_without_hit*env.dt
      return 1 + time_left*max_reward
    else:
      # Get distance to surface
      dist = dist - env.target_radius
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

class TimeCost(BaseFunction):

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 8
    else:
      return -0.1

  def __repr__(self):
    return "TimeCost"


class NegativeExpDistanceWithHitBonus(BaseFunction):

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
    return "NegativeExpDistanceWithHitBonus"
