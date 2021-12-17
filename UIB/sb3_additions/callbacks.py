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