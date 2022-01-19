from gym.envs.registration import register

register(id='mobl-arms-muscles-v0', entry_point='UIB.envs.mobl_arms:OnePolicyNoCamera')
register(id='mobl-arms-muscles-v01', entry_point='UIB.envs.mobl_arms:TwoLevelSL')
register(id='mobl-arms-muscles-v1', entry_point='UIB.envs.mobl_arms:MuscleActuatedWithCamera')