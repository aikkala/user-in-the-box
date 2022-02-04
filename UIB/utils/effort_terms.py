import numpy as np
from abc import ABC, abstractmethod
import mujoco_py

class BaseTerm(ABC):
  @abstractmethod
  def get(self, env):
    pass
  @abstractmethod
  def __repr__(self):
    pass

class Neural(BaseTerm):
  def __init__(self, weight=1e-4):
      self.weight = weight
  def get(self, env):
    return self.weight * np.sum(env.sim.data.ctrl ** 2)
  def __repr__(self):
    return "Neural"

class Composite(BaseTerm):
  def __init__(self, weight=1e-5):
    self.weight = weight
  def get(self, env):
    mujoco_py.cymj._mj_inverse(env.sim.model, env.sim.data)
    angle_acceleration = np.sum(env.sim.data.qacc[env.independent_joints] ** 2)
    energy = np.sum(env.sim.data.qvel[env.independent_joints] ** 2 * env.sim.data.qfrc_inverse[env.independent_joints] ** 2)
    return self.weight * (energy + 0.05 * angle_acceleration)
  def __repr__(self):
    return "Composite"

class Zero(BaseTerm):
  def get(self, env):
    return 0
  def __repr__(self):
    return "Zero"

class MuscleState(BaseTerm):
  def __init__(self, weight=1e-4):
    self.weight = weight
  def get(self, env):
    return self.weight * np.sum(env.sim.data.act ** 2)