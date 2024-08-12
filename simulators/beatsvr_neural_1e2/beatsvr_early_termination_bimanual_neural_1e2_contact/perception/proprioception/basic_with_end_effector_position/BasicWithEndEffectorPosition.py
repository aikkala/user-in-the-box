from ...base import BaseModule

import numpy as np


class BasicWithEndEffectorPosition(BaseModule):

  def __init__(self, model, data, bm_model, end_effector, **kwargs):
    """ Initialise a new `BasicWithEndEffectorPosition`. Represents proprioception through joint angles, velocities,
    and accelerations, and muscle activation states, and an end effector global position.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      bm_model: An instance inheriting from uitb.bm_models.base.BaseBMModel class.
      end_effector (list of lists): Each list must have two elements, with first element representing type of mujoco
        element (geom, body, site), and second element is the name of the element
      kwargs: may contain "rng" seed
    """
    super().__init__(model, data, bm_model, **kwargs)

    if not isinstance(end_effector, list):
      raise RuntimeError("end_effector must be a list of size two, or a nested list with each list of size two")

    # Simple check if the list is nested
    if isinstance(end_effector[0], str):
      # Make nested
      end_effector = [end_effector]

    # Make sure all nested lists have two elements
    if any(len(pair) != 2 for pair in end_effector):
      raise RuntimeError("end_effector must be a list of size two")

    self._end_effector = end_effector

  @staticmethod
  def insert(task, **kwargs):
    pass

  @property
  def _default_encoder(self):
    return {"module": "rl.encoders", "cls": "OneLayer", "kwargs": {"out_features": 128}}

  def get_observation(self, model, data, info=None):

    # Normalise qpos
    jnt_range = model.jnt_range[self._bm_model.independent_joints]
    qpos = data.qpos[self._bm_model.independent_qpos].copy()
    qpos = (qpos - jnt_range[:, 0]) / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5) * 2

    # Get qvel, qacc
    qvel = data.qvel[self._bm_model.independent_dofs].copy()
    qacc = data.qacc[self._bm_model.independent_dofs].copy()

    # Get end-effector position; not normalised
    ee_position = []
    for pair in self._end_effector:
      ee_position.append(getattr(data, pair[0])(pair[1]).xpos.copy())
    ee_position = np.hstack(ee_position)

    # Normalise act
    act = (data.act.copy() - 0.5) * 2

    # Smoothed average of motor actuation (only for motor actuators); normalise
    motor_act = (self._bm_model.motor_act.copy() - 0.5) * 2

    # Proprioception features
    proprioception = np.concatenate([qpos, qvel, qacc, ee_position, act, motor_act])

    return proprioception

  def _get_state(self, model, data):
    state = {}
    for pair in self._end_effector:
      state[f"{pair[1]}_xpos"] = getattr(data, pair[0])(pair[1]).xpos.copy()
      state[f"{pair[1]}_xmat"] = getattr(data, pair[0])(pair[1]).xmat.copy()
    return state