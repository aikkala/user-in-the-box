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
  def reset(self):
    pass

class Neural(BaseTerm):
  def __init__(self, weight=1e-4):
      self.weight = weight
  def get(self, env):
    return self.weight * np.sum(env.sim.data.ctrl ** 2)
  def __repr__(self):
    return "Neural"

class Composite(BaseTerm):
  def __init__(self, weight=1e-7):
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
  def __repr__(self):
    return "MuscleState"

class CumulativeFatigue(BaseTerm):
  # 3CC-r model, adapted from https://dl.acm.org/doi/pdf/10.1145/3313831.3376701 for muscles -- using control signals
  # instead of torque or force
  def __init__(self):
    self.r = 7.5
    self.F = 0.0146
    self.R = 0.0022
    self.LD = 10
    self.LR = 10
    self.MA = None
    self.MR = None
    self.weight = 0.01

  def get(self, env):

    # Initialise MA if not yet initialised
    if self.MA is None:
      self.MA = np.zeros((env.model.na,))
      self.MR = np.ones((env.model.na,))

    # Get target load
    TL = env.sim.data.act.copy()

    # Calculate C(t)
    C = np.zeros_like(self.MA)
    idxs = (self.MA < TL) & (self.MR > (TL - self.MA))
    C[idxs] = self.LD * (TL[idxs] - self.MA[idxs])
    idxs = (self.MA < TL) & (self.MR <= (TL - self.MA))
    C[idxs] = self.LD * self.MR[idxs]
    idxs = self.MA >= TL
    C[idxs] = self.LR * (TL[idxs] - self.MA[idxs])

    # Calculate rR
    rR = np.zeros_like(self.MA)
    idxs = self.MA >= TL
    rR[idxs] = self.r*self.R
    idxs = self.MA < TL
    rR[idxs] = self.R

    # Calculate MA, MR
    self.MA += (C - self.F*self.MA)*env.dt
    self.MR += (-C + rR*self.MR)*env.dt

    # Not sure if these are needed
    self.MA = np.clip(self.MA, 0, 1)
    self.MR = np.clip(self.MR, 0, 1)

    # Calculate effort
    effort = np.linalg.norm(self.MA - TL)

    return self.weight*effort

  def reset(self):
    self.MA = None
    self.MR = None

  def __repr__(self):
    return "CumulativeFatigue"