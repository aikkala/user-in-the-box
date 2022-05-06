import os
import gym

from stable_baselines3 import PPO as PPO_sb3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

from uitb.rl.base import BaseModel
from uitb.rl.sb3.callbacks import EvalCallback
from uitb.rl.sb3.feature_extractor import FeatureExtractor

class PPO(BaseModel):

  def __init__(self, simulator, rl_config, run_folder):
    super().__init__()

    # Get total timesteps
    self.total_timesteps = rl_config["total_timesteps"]

    # Initialise parallel envs_old_to_be_removed
    parallel_envs = make_vec_env(simulator.id, n_envs=rl_config["num_workers"], seed=0,
                                 vec_env_cls=SubprocVecEnv, env_kwargs={"run_folder": run_folder})

    # Add feature and stateful information extractors to policy_kwargs
    extractors = simulator.perception.extractors.copy()
    if simulator.task.get_stateful_information(simulator.model, simulator.data) is not None:
      extractors["stateful_information"] = simulator.task.stateful_information_extractor
    rl_config["policy_kwargs"]["features_extractor_kwargs"] = {"extractors": extractors}

    # Initialise model
    self.model = PPO_sb3(rl_config["policy_type"], parallel_envs, verbose=1, policy_kwargs=rl_config["policy_kwargs"],
                         tensorboard_log=run_folder, n_steps=rl_config["nsteps"], batch_size=rl_config["batch_size"],
                         target_kl=rl_config["target_kl"], learning_rate=rl_config["lr"], device=rl_config["device"])


    # Create a checkpoint callback
    save_freq = rl_config["save_freq"] // rl_config["num_workers"]
    checkpoint_folder = os.path.join(run_folder, 'checkpoints')
    self.checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                  save_path=checkpoint_folder,
                                                  name_prefix='model')

    # Create an evaluation callback
#    eval_env = gym.make(rl_config["env_name"], **rl_config["env_kwargs"])
#    self.eval_callback = EveryNTimesteps(n_steps=rl_config["total_timesteps"]//100, callback=EvalCallback(eval_env, num_eval_episodes=20))

  def learn(self, wandb_callback):
    #env_callbacks = self.config["env_kwargs"].get("callbacks", [])
    self.model.learn(total_timesteps=self.total_timesteps, callback=[wandb_callback, self.checkpoint_callback])