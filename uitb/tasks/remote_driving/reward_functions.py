from abc import ABC, abstractmethod
import numpy as np


class BaseFunction(ABC):
  @abstractmethod
  def get(self, ee_joystick, car_target, info, model, data):
    pass
  @abstractmethod
  def reset(self):
    pass

class NegativeExpDistance(BaseFunction):

  def __init__(self, joystick_specs=None, target_specs=None):

    # Default parameters for joystick related components
    self.joystick_specs = {"k": 3, "shift": -1, "scale": 1,
                           "bonus": 0, "bonus_active": False, "bonus_onetime": False}
    # Update parameters if needed
    if joystick_specs is not None:
      self.joystick_specs.update(joystick_specs)
    self.joystick_bonus_active = self.joystick_specs["bonus_active"]

    # Default parameters for target related components
    self.target_specs = {"k": 3, "shift": -1, "scale": 0.1,
                         "bonus": 8, "bonus_active": True, "bonus_onetime": False}
    # Update parameters if needed
    if target_specs is not None:
      self.target_specs.update(target_specs)
    self.target_bonus_active = self.target_specs["bonus_active"]

  def get(self, dist_ee_joystick, dist_car_target, info, model, data):

    cost = lambda dist, specs: (np.exp(-dist*specs["k"]) + specs["shift"])*specs["scale"]

    # Distance between end-effector and joystick
    reward = cost(dist_ee_joystick, self.joystick_specs)

    # Distance between car and target
    reward += cost(dist_car_target, self.target_specs)

    # Bonus for hitting joystick
    if self.joystick_bonus_active and info["end_effector_at_joystick"]:
      reward += self.joystick_specs["bonus"]
      if self.joystick_specs["bonus_onetime"]:
        self.joystick_bonus_active = False

    # Bonus for being inside target
    if self.target_bonus_active and info["inside_target"]:
      reward += self.target_specs["bonus"]
      if self.target_specs["bonus_onetime"]:
        self.target_bonus_active = False

    return reward

  def reset(self):
    self.joystick_bonus_active = self.joystick_specs["bonus_active"]
    self.target_bonus_active = self.target_specs["bonus_active"]


class PositiveExpDistance(BaseFunction):

  def __init__(self, joystick_specs=None, target_specs=None):

    # Default parameters for joystick related components
    self.joystick_specs = {"k": 3, "shift": 0, "scale": 0.1,
                           "bonus": 0, "bonus_active": False, "bonus_onetime": False}
    # Update parameters if needed
    if joystick_specs is not None:
      self.joystick_specs.update(joystick_specs)
    self.joystick_bonus_active = self.joystick_specs["bonus_active"]

    # Default parameters for target related components
    self.target_specs = {"k": 3, "shift": 0, "scale": 0.1,
                         "bonus": 1, "bonus_active": True, "bonus_onetime": False}
    # Update parameters if needed
    if target_specs is not None:
      self.target_specs.update(target_specs)
    self.target_bonus_active = self.target_specs["bonus_active"]

  def get(self, dist_ee_joystick, dist_car_target, info, model, data):

    cost = lambda dist, specs: (np.exp(-dist*specs["k"]) + specs["shift"])*specs["scale"]

    # Distance between end-effector and joystick
    reward = cost(dist_ee_joystick, self.joystick_specs)

    # Distance between car and target
    reward += cost(dist_car_target, self.target_specs)

    # Bonus for hitting joystick
    if self.joystick_bonus_active and info["end_effector_at_joystick"]:
      reward += self.joystick_specs["bonus"]
      if self.joystick_specs["bonus_onetime"]:
        self.joystick_bonus_active = False

    # Bonus for being inside target
    if self.target_bonus_active and info["inside_target"]:
      reward += self.target_specs["bonus"]
      if self.target_specs["bonus_onetime"]:
        self.target_bonus_active = False

    return reward

  def reset(self):
    self.joystick_bonus_active = self.joystick_specs["bonus_active"]
    self.target_bonus_active = self.target_specs["bonus_active"]
