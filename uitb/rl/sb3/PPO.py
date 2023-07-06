import os
import importlib
import numpy as np
import pathlib

from stable_baselines3 import PPO as PPO_sb3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback  #, EvalCallback

from ..base import BaseRLModel
from .callbacks import EvalCallback


class PPO(BaseRLModel):

  def __init__(self, simulator, checkpoint_path=None, wandb_id=None):
    super().__init__()

    rl_config = self.load_config(simulator)
    run_parameters = simulator.run_parameters
    simulator_folder = simulator.simulator_folder

    # Get total timesteps
    self.total_timesteps = rl_config["total_timesteps"]
        
    # Initialise parallel envs
    self.n_envs = rl_config["num_workers"]
    parallel_envs = make_vec_env(simulator.__class__, n_envs=self.n_envs,
                                 seed=run_parameters.get("random_seed", None), vec_env_cls=SubprocVecEnv,
                                 env_kwargs={"simulator_folder": simulator_folder})
    
    if checkpoint_path is not None:
        # Resume training
        self.model = PPO_sb3.load(checkpoint_path, parallel_envs, verbose=1, #policy_kwargs=rl_config["policy_kwargs"], 
                                  tensorboard_log=simulator_folder, n_steps=rl_config["nsteps"],
                                  batch_size=rl_config["batch_size"], target_kl=rl_config["target_kl"],
                                  learning_rate=rl_config["lr"], device=rl_config["device"])
        self.training_resumed = True
    else:
        # Add feature and stateful information encoders to policy_kwargs
        encoders = simulator.perception.encoders
        if simulator.task.get_stateful_information_space_params()["shape"] != (0,):
          #TODO: define stateful_information (and encoder) that can be used as default, if no stateful information is provided (zero-size array do not work with sb3 currently...)
          encoders["stateful_information"] = simulator.task.stateful_information_encoder
        rl_config["policy_kwargs"]["features_extractor_kwargs"] = {"encoders": encoders}
        rl_config["policy_kwargs"]["wandb_id"] = wandb_id

        # Initialise model
        self.model = PPO_sb3(rl_config["policy_type"], parallel_envs, verbose=1, policy_kwargs=rl_config["policy_kwargs"],
                             tensorboard_log=simulator_folder, n_steps=rl_config["nsteps"],
                             batch_size=rl_config["batch_size"], target_kl=rl_config["target_kl"],
                             learning_rate=rl_config["lr"], device=rl_config["device"])
        self.training_resumed = False

        if "policy_init" in rl_config:
            params = os.path.join(pathlib.Path(__file__).parent, rl_config["policy_init"])
            self.model.policy.load_from_vector(np.load(params))
    
    # Create a checkpoint callback
    save_freq = rl_config["save_freq"] // self.n_envs
    checkpoint_folder = os.path.join(simulator_folder, 'checkpoints')
    self.checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                  save_path=checkpoint_folder,
                                                  name_prefix='model')

    # Get callbacks as a list
    self.callbacks = [*simulator.callbacks.values()]
    
    # Create an evaluation env (only used if eval_callback=True is passed to learn())
    self.eval_env = simulator.__class__(**{"simulator_folder": simulator_folder})
    
  def load_config(self, simulator):
    config = simulator.config["rl"]

    # Need to translate strings into classes
    config["policy_type"] = simulator.get_class("rl.sb3", config["policy_type"])

    if "activation_fn" in config["policy_kwargs"]:
      mods = config["policy_kwargs"]["activation_fn"].split(".")
      config["policy_kwargs"]["activation_fn"] = getattr(importlib.import_module(".".join(mods[:-1])), mods[-1])

    config["policy_kwargs"]["features_extractor_class"] = \
      simulator.get_class("rl.sb3", config["policy_kwargs"]["features_extractor_class"])

    if "lr" in config:
      if isinstance(config["lr"], dict):
        config["lr"] = simulator.get_class("rl.sb3", config["lr"]["function"])(**config["lr"]["kwargs"])

    return config

  def learn(self, wandb_callback, with_evaluation=False, eval_freq=10000, n_eval_episodes=5, eval_info_keywords=()):
    if with_evaluation:
        self.eval_env = Monitor(self.eval_env, info_keywords=eval_info_keywords)
        self.eval_freq = eval_freq // self.n_envs
        self.eval_callback = EvalCallback(self.eval_env, eval_freq=self.eval_freq, n_eval_episodes=n_eval_episodes, info_keywords=eval_info_keywords)

        self.model.learn(total_timesteps=self.total_timesteps,
                         callback=[wandb_callback, self.checkpoint_callback, self.eval_callback, *self.callbacks],
                         reset_num_timesteps=not self.training_resumed)
    else:
        self.model.learn(total_timesteps=self.total_timesteps,
                         callback=[wandb_callback, self.checkpoint_callback, *self.callbacks],
                         reset_num_timesteps=not self.training_resumed)