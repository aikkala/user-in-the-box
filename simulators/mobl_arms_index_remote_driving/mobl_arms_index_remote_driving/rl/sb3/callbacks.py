import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback


class LinearStdDecayCallback(BaseCallback):
    """
    Linearly decaying standard deviation

    :param initial_log_value: Log initial standard deviation value
    :param threshold: Threshold for progress remaining until decay begins
    :param min_value: Minimum value for standard deviation
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, initial_log_value, threshold, min_value, verbose=0):
      super(LinearStdDecayCallback, self).__init__(verbose)
      self.initial_value = np.exp(initial_log_value)
      self.threshold = threshold
      self.min_value = min_value

    def _on_rollout_start(self) -> None:
      progress_remaining = self.model._current_progress_remaining
      if progress_remaining > self.threshold:
        pass
      else:
        new_std = self.min_value + (progress_remaining/self.threshold) * (self.initial_value-self.min_value)
        self.model.policy.log_std.data = torch.tensor(np.log(new_std)).float()

    def _on_training_start(self) -> None:
      pass

    def _on_step(self) -> bool:
      return True

    def _on_rollout_end(self) -> None:
      pass

    def _on_training_end(self) -> None:
      pass


class LinearCurriculum(BaseCallback):
  """
  A callback to implement linear curriculum for one parameter

  :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
  """

  def __init__(self, name, start_value, end_value, end_timestep, start_timestep=0, verbose=0):
    super().__init__(verbose)
    self.name = name
    self.variable = start_value
    self.start_value = start_value
    self.end_value = end_value
    self.start_timestep = start_timestep
    self.end_timestep = end_timestep
    self.coeff = (end_value - start_value) / (end_timestep - start_timestep)

  def value(self):
    return self.variable

  def update(self, num_timesteps):
    if num_timesteps <= self.start_timestep:
      self.variable = self.start_value
    elif self.end_timestep >= num_timesteps > self.start_timestep:
      self.variable = self.start_value + self.coeff * (num_timesteps - self.start_timestep)
    else:
      self.variable = self.end_value

  def _on_training_start(self) -> None:
    pass

  def _on_rollout_start(self) -> None:
    self.training_env.env_method("callback", self.name, self.num_timesteps)

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> None:
    pass

  def _on_training_end(self) -> None:
    pass

class EvalCallback(BaseCallback):
  """
  A custom callback that derives from ``BaseCallback``.

  :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
  """

  def __init__(self, env, num_eval_episodes, verbose=0):
    super().__init__(verbose)
    self.env = env
    self.num_eval_episodes = num_eval_episodes

  def _on_training_start(self) -> None:
    pass

  def _on_rollout_start(self) -> None:

    # Run a few episodes to evaluate progress, with and without deterministic actions
    det_info = self.evaluate(deterministic=True)
    sto_info = self.evaluate(deterministic=False)

    # Log evaluations
    self.logger.record("evaluate/deterministic/ep_rew_mean", det_info[0])
    self.logger.record("evaluate/deterministic/ep_len_mean", det_info[1])
    self.logger.record("evaluate/deterministic/ep_targets_hit_mean", det_info[2])

    self.logger.record("evaluate/stochastic/ep_rew_mean", sto_info[0])
    self.logger.record("evaluate/stochastic/ep_len_mean", sto_info[1])
    self.logger.record("evaluate/stochastic/ep_targets_hit_mean", sto_info[2])

    self.logger.dump(step=self.num_timesteps)

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> None:
    pass

  def _on_training_end(self) -> None:
    pass

  def evaluate(self, deterministic):
    rewards = np.zeros((self.num_eval_episodes,))
    episode_lengths = np.zeros((self.num_eval_episodes,))
    targets_hit = np.zeros((self.num_eval_episodes,))

    for i in range(self.num_eval_episodes):
      obs = self.env.reset()
      done = False
      while not done:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        obs, r, done, info = self.env.step(action)
        rewards[i] += r
      episode_lengths[i] = self.env.steps
      targets_hit[i] = self.env.trial_idx

    return np.mean(rewards), np.mean(episode_lengths), np.mean(targets_hit)