from typing import Callable

def linear_schedule(initial_value: float, min_value: float, threshold: float = 1.0) -> Callable[[float], float]:
  """
  Linear learning rate schedule. Adapted from the example at
  https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule

  :param initial_value: Initial learning rate.
  :param min_value: Minimum learning rate.
  :param threshold: Threshold (of progress) when decay begins.
  :return: schedule that computes
    current learning rate depending on remaining progress
  """

  def func(progress_remaining: float) -> float:
    """
    Progress will decrease from 1 (beginning) to 0.

    :param progress_remaining:
    :return: current learning rate
    """
    if progress_remaining > threshold:
      return initial_value
    else:
      return min_value + (progress_remaining/threshold) * (initial_value - min_value)

  return func
