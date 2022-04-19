from UIB.perception.base import BaseModule
from UIB.utils.functions import parent_path

import numpy as np


class BasicWithEndEffectorPosition(BaseModule):

  def __init__(self, sim, bm_model, end_effector, **kwargs):
    self.end_effector = end_effector
    super().__init__(sim, bm_model)
    self.module_folder = parent_path(__file__)
    self.independent_joints = None

  @staticmethod
  def insert(task, config, **kwargs):
    pass

  def reset(self, sim, rng):
    pass

  def extractor(self):
    return None

  def get_observation_space_params(self):
    return {"low": -float('inf'), "high": float('inf'), "shape": self.observation_shape}

  def get_observation(self, sim):

    # Normalise qpos
    jnt_range = sim.model.jnt_range[self.bm_model.independent_joints]
    qpos = sim.data.qpos[self.bm_model.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5) * 2

    # Get qvel, qacc
    qvel = sim.data.qvel[self.bm_model.independent_joints].copy()
    qacc = sim.data.qacc[self.bm_model.independent_joints].copy()

    # Get end-effector position; not normalised
    fingertip_position = sim.data.get_geom_xpos(self.end_effector)

    # Normalise act
    act = (sim.data.act.copy() - 0.5) * 2

    # Proprioception features
    proprioception = np.concatenate([qpos, qvel, qacc, fingertip_position, act])

    return proprioception