from gym.envs.registration import register

register(id='mobl-arms-pointing-v0', entry_point='UIB.envs_old_to_be_removed.mobl_arms.pointing.PointingEnv:Proprioception')
register(id='mobl-arms-pointing-v1', entry_point='UIB.envs_old_to_be_removed.mobl_arms.pointing.PointingEnv:ProprioceptionAndVisual')
register(id='mobl-arms-iso-pointing-v1', entry_point='UIB.envs_old_to_be_removed.mobl_arms.iso_pointing.ISOPointingEnv:ProprioceptionAndVisual')
register(id='mobl-arms-tracking-v1', entry_point='UIB.envs_old_to_be_removed.mobl_arms.tracking.TrackingEnv:ProprioceptionAndVisual')
register(id='mobl-arms-button-press-v1', entry_point='UIB.envs_old_to_be_removed.mobl_arms.button_press.ButtonPressEnv:ProprioceptionAndVisual')
register(id='mobl-arms-remote-driving-v1', entry_point='UIB.envs_old_to_be_removed.mobl_arms.remote_driving.RemoteDrivingEnv:ProprioceptionAndVisual')


register(id="simulator-v0", entry_point="UIB.simulator:Simulator")
