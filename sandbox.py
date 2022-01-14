import gym
import os
import torch
import numpy as np
from platform import uname
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from UIB.sb3_additions.schedule import linear_schedule
from UIB.sb3_additions.policies import MultiInputActorCriticPolicyTanhActions
from UIB.sb3_additions.feature_extractor import VisualAndProprioceptionExtractor


if __name__=="__main__":

  env_name = 'UIB:mobl-arms-muscles-v1'
  start_method = 'spawn' if 'Microsoft' in uname().release else 'forkserver'
  num_cpu = 6

  # Get project path
  project_path = pathlib.Path(__file__).parent.absolute()

  # Define output directories
  output_dir = os.path.join(project_path, 'output', env_name)
  checkpoint_dir = os.path.join(output_dir, 'checkpoint')
  log_dir = os.path.join(output_dir, 'log')

  # Leave for future kwargs
  env_kwargs = {"target_radius_limit": np.array([0.05, 0.15])}

  # Initialise parallel envs
  parallel_envs = make_vec_env(env_name, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs,
                               vec_env_kwargs={'start_method': start_method})

  # Policy parameters
  policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
                       net_arch=[256, 256],
                       log_std_init=0.0, features_extractor_class=VisualAndProprioceptionExtractor,
                       normalize_images=False)
  lr = 1e-4

  # Initialise policy
  model = PPO(MultiInputActorCriticPolicyTanhActions, parallel_envs, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=log_dir,
              n_steps=4000, batch_size=400, target_kl=5.0, #learning_rate=lr,
              learning_rate=linear_schedule(initial_value=lr, min_value=1e-7, threshold=0.8))

  # Initialise a callback for checkpoints
  save_freq = 1000000 // num_cpu
  checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_dir, name_prefix='model')

  # Initialise a callback for linearly decaying standard deviation
  #std_callback = LinearStdDecayCallback(initial_log_value=policy_kwargs['log_std_init'],
  #                                      threshold=policy_kwargs['std_decay_threshold'],
  #                                      min_value=policy_kwargs['std_decay_min'])

  # Do the learning first with constant learning rate
  model.learn(total_timesteps=50_000_000, callback=[checkpoint_callback])
