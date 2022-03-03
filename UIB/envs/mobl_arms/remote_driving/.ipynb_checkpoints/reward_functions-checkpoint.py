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

  def __init__(self, k=3):
    self.k = k

  def get(self, env, dist, info):
    if info["target_hit"]:
      return 8
    else:
      return (np.exp(-dist*self.k) - 1)/10

  def __repr__(self):
    return "NegativeExpDistanceWithHitBonus"


class RewardJoystick(BaseFunction):

  def __init__(self, bonus=0.1):
    self.bonus = bonus

  def get(self, body_at_joystick):
    if body_at_joystick:
      return self.bonus
    return 0

  def __repr__(self):
    return "RewardJoystick"