from gym.envs.registration import register

register(id='mobl-arms-pointing-v0', entry_point='UIB.envs.mobl_arms.pointing.PointingEnv:Proprioception')
register(id='mobl-arms-pointing-v1', entry_point='UIB.envs.mobl_arms.pointing.PointingEnv:ProprioceptionAndVisual')
register(id='mobl-arms-iso-pointing-v1', entry_point='UIB.envs.mobl_arms.iso_pointing.ISOPointingEnv:ProprioceptionAndVisual')
register(id='mobl-arms-tracking-v1', entry_point='UIB.envs.mobl_arms.tracking.TrackingEnv:ProprioceptionAndVisual')
register(id='mobl-arms-button-press-v1', entry_point='UIB.envs.mobl_arms.button_press.ButtonPressEnv:ProprioceptionAndVisual')
register(id='mobl-arms-remote-driving-v1', entry_point='UIB.envs.mobl_arms.remote_driving.RemoteDrivingEnv:ProprioceptionAndVisual')

