from gym.envs.registration import register


register(id="MBRLCartpole-v0", entry_point="src.envs.cartpole:CartPoleEnv")


# register(id="MBRLReacher3D-v0", entry_point="src.envs.reacher:Reacher3DEnv")

# register(id="MBRLPusher-v0", entry_point="src.envs.pusher:PusherEnv")


register(id="MBRLHalfCheetah-v0", entry_point="src.envs.half_cheetah:HalfCheetahEnv")


# register(id="MBRLAnt-v0", entry_point="src.envs.ant:AntEnv")
