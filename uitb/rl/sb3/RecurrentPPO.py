import os
import numpy as np

from sb3_contrib import RecurrentPPO as RecurrentPPO_sb3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from uitb.rl.base import BaseModel


class RecurrentPPO(BaseModel):

  def __init__(self, config, run_folder):
    super().__init__(config)

    # Initialise parallel envs_old_to_be_removed
    parallel_envs = make_vec_env(config["env_name"], n_envs=config["num_workers"], seed=0,
                                 vec_env_cls=SubprocVecEnv, env_kwargs=config["env_kwargs"],
                                 vec_env_kwargs={'start_method': config["start_method"]})

    # Initialise model
    self.model = RecurrentPPO_sb3(config["policy_type"], parallel_envs, verbose=1,
                                  policy_kwargs=config["policy_kwargs"], tensorboard_log=run_folder,
                                  n_steps=config["nsteps"], batch_size=config["batch_size"],
                                  target_kl=config["target_kl"], learning_rate=config["lr"], device=config["device"])


    save_freq = config["save_freq"] // config["num_workers"]
    checkpoint_folder = os.path.join(run_folder, 'checkpoints')
    self.checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                  save_path=checkpoint_folder,
                                                  name_prefix='model')

  def learn(self, wandb_callback):
    self.model.learn(total_timesteps=self.config["total_timesteps"],
                     callback=[wandb_callback, self.checkpoint_callback])