from gym.envs.registration import register

register(id='mobl-arms-pointing-v0', entry_point='UIB.envs.mobl_arms.pointing.PointingEnv:Proprioception')
register(id='mobl-arms-pointing-v1', entry_point='UIB.envs.mobl_arms.pointing.PointingEnv:ProprioceptionAndVisual')
register(id='mobl-arms-tracking-v1', entry_point='UIB.envs.mobl_arms.tracking.TrackingEnv:ProprioceptionAndVisual')
register(id='mobl-arms-choosing-v1', entry_point='UIB.envs.mobl_arms.choosing.ChoosingEnv:ProprioceptionAndVisual')