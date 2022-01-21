import gym
from gym import spaces
import numpy as np

from UIB.envs.mobl_arms.base import BaseModel

class OnePolicy(BaseModel):
  metadata = {'render.modes': ['human']}

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Reset
    observation = self.reset()

    # Set observation space
    self.observation_space = spaces.Dict({
      'proprioception': spaces.Box(low=-float('inf'), high=float('inf'), shape=observation['proprioception'].shape,
                                   dtype=np.float32),
      'visual': spaces.Box(low=-1, high=1, shape=observation['visual'].shape, dtype=np.float32),
      'ocular': spaces.Box(low=-float('inf'), high=float('inf'), shape=observation['ocular'].shape, dtype=np.float32)})

  def get_observation(self):
    # Ignore eye qpos and qvel for now
    jnt_range = self.sim.model.jnt_range[self.independent_joints]

    # Normalise qpos
    qpos = self.sim.data.qpos[self.independent_joints].copy()
    qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5)*2
    qvel = self.sim.data.qvel[self.independent_joints].copy()
    qacc = self.sim.data.qacc[self.independent_joints].copy()

    # Normalise act
    act = (self.sim.data.act.copy() - 0.5)*2

    # Estimate fingertip position, normalise to target_origin
    finger_position = self.sim.data.get_geom_xpos(self.fingertip).copy() - self.target_origin

    # Get depth array and normalise
    render = self.sim.render(width=120, height=80, camera_name='oculomotor', depth=True)
    depth = render[1]
    depth = np.flipud((depth - 0.5)*2)
    rgb = render[0]
    rgb = np.flipud((rgb/255.0 - 0.5)*2)

    return {'proprioception': np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position, act]),
            #'visual': np.transpose(np.concatenate([rgb, np.expand_dims(depth, 2)], axis=2), (2, 0, 1)),
            'visual': np.expand_dims(depth, 0),
            'ocular': np.concatenate([qpos[:2], qvel[:2], qacc[:2]])}