import torch
import numpy as np
from platform import uname
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

import wandb
from wandb.integration.sb3 import WandbCallback

from UIB.sb3_additions.schedule import linear_schedule
from UIB.utils.functions import output_path, timeout_input


if __name__=="__main__":

  config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100_000_000,
    "env_name": "UIB:mobl-arms-muscles-v0",
    "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
    "num_cpu": 10,
    "env_kwargs": {"target_radius_limit": np.array([0.05, 0.15]),
                   "action_sample_freq": 20,
                   "max_trials": 10,
                   "cost_function": "neural_effort"},
    "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                      "net_arch": [256, 256],
                      "log_std_init": 0.0},
    "lr": linear_schedule(initial_value=1e-4, min_value=1e-7, threshold=0.8),
    "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
  }

  name = timeout_input("Give a name for this run. Input empty string or wait for 30 seconds for a random name.",
                       timeout=30, default="")
  run = wandb.init(project="uib", name=name, config=config, sync_tensorboard=True, save_code=True, dir=output_path())

  # Define output directories
  model_folder = os.path.join(output_path(), config["env_name"], 'trained-models')

  # Initialise parallel envs
  parallel_envs = make_vec_env(config["env_name"], n_envs=config["num_cpu"], seed=0,
                               vec_env_cls=SubprocVecEnv, env_kwargs=config["env_kwargs"],
                               vec_env_kwargs={'start_method': config["start_method"]})

  # Initialise policy
  model = PPO(config["policy_type"], parallel_envs, verbose=1, policy_kwargs=config["policy_kwargs"],
              tensorboard_log=os.path.join(model_folder, run.name),
              n_steps=config["nsteps"], batch_size=config["batch_size"], target_kl=config["target_kl"],
              learning_rate=config["lr"])

  # Haven't figured out how to periodically save models in wandb, so let's do it in sb3 and upload the models manually
  # TODO this doesn't seem to work; do the files need to be in wandb.run.dir?
  wandb.save(os.path.join(model_folder, run.name, 'checkpoints', "model_*_steps.zip"),
             base_path=os.path.join(model_folder, run.name, 'checkpoints'))
  save_freq = config["save_freq"] // config["num_cpu"]
  checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                           save_path=os.path.join(model_folder, run.name, "checkpoints"),
                                           name_prefix='model')

  # Do the learning first with constant learning rate
  model.learn(total_timesteps=config["total_timesteps"], callback=[WandbCallback(verbose=2), checkpoint_callback])
  run.finish()