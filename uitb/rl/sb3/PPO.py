import os
import importlib

from stable_baselines3 import PPO as PPO_sb3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

from ..base import BaseRLModel
from .callbacks import EvalCallback

class PPO(BaseRLModel):

  def __init__(self, simulator):
    super().__init__()

    rl_config = self.load_config(simulator)
    run_parameters = simulator.run_parameters
    simulator_folder = simulator.simulator_folder

    # Get total timesteps
    self.total_timesteps = rl_config["total_timesteps"]

    # Initialise parallel envs
    parallel_envs = make_vec_env(simulator.__class__, n_envs=rl_config["num_workers"],
                                 seed=run_parameters.get("random_seed", None), vec_env_cls=SubprocVecEnv,
                                 env_kwargs={"simulator_folder": simulator_folder})

    # Add feature and stateful information encoders to policy_kwargs
    encoders = simulator.perception.encoders.copy()
    if simulator.task.get_stateful_information_space_params() is not None:
      encoders["stateful_information"] = simulator.task.stateful_information_encoder
    rl_config["policy_kwargs"]["features_extractor_kwargs"] = {"extractors": encoders}

    # Initialise model
    self.model = PPO_sb3(rl_config["policy_type"], parallel_envs, verbose=1, policy_kwargs=rl_config["policy_kwargs"],
                         tensorboard_log=simulator_folder, n_steps=rl_config["nsteps"],
                         batch_size=rl_config["batch_size"], target_kl=rl_config["target_kl"],
                         learning_rate=rl_config["lr"], device=rl_config["device"])


    # Create a checkpoint callback
    save_freq = rl_config["save_freq"] // rl_config["num_workers"]
    checkpoint_folder = os.path.join(simulator_folder, 'checkpoints')
    self.checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                  save_path=checkpoint_folder,
                                                  name_prefix='model')

    # Get callbacks as a list
    self.callbacks = [*simulator.callbacks.values()]

    # Create an evaluation callback
#    eval_env = gym.make(rl_config["env_name"], **rl_config["env_kwargs"])
#    self.eval_callback = EveryNTimesteps(n_steps=rl_config["total_timesteps"]//100, callback=EvalCallback(eval_env, num_eval_episodes=20))

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

  def learn(self, wandb_callback):
    self.model.learn(total_timesteps=self.total_timesteps,
                     callback=[wandb_callback, self.checkpoint_callback, *self.callbacks])