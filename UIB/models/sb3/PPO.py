import os
import gym

from stable_baselines3 import PPO as PPO_sb3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

from UIB.models.base import BaseModel
from UIB.models.sb3.callbacks import EvalCallback

class PPO(BaseModel):

  def __init__(self, config, run_folder):
    super().__init__(config)

    # Initialise parallel envs
    parallel_envs = make_vec_env(config["env_name"], n_envs=config["num_workers"], seed=0,
                                 vec_env_cls=SubprocVecEnv, env_kwargs=config["env_kwargs"],
                                 vec_env_kwargs={'start_method': config["start_method"]})

    # Initialise model
    if "resume" in config and os.path.exists(config["resume"]):
        self.model = PPO_sb3.load(config["resume"], parallel_envs, verbose=1, policy_kwargs=config["policy_kwargs"],
                                  tensorboard_log=run_folder, n_steps=config["nsteps"], batch_size=config["batch_size"],
                                  target_kl=config["target_kl"], learning_rate=config["lr"], device=config["device"])
    else:
        self.model = PPO_sb3(config["policy_type"], parallel_envs, verbose=1, policy_kwargs=config["policy_kwargs"],
                                 tensorboard_log=run_folder, n_steps=config["nsteps"], batch_size=config["batch_size"],
                                 target_kl=config["target_kl"], learning_rate=config["lr"], device=config["device"])


    # Create a checkpoint callback
    save_freq = config["save_freq"] // config["num_workers"]
    checkpoint_folder = os.path.join(run_folder, 'checkpoints')
    self.checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                  save_path=checkpoint_folder,
                                                  name_prefix='model')

    # Create an evaluation callback
    eval_env = gym.make(config["env_name"], **config["env_kwargs"])
    self.eval_callback = EveryNTimesteps(n_steps=config["total_timesteps"]//100, callback=EvalCallback(eval_env, num_eval_episodes=20))

  def learn(self, wandb_callback):
    env_callbacks = self.config["env_kwargs"].get("callbacks", [])
    self.model.learn(total_timesteps=self.config["total_timesteps"],
                     callback=[wandb_callback, self.checkpoint_callback, self.eval_callback, *env_callbacks])