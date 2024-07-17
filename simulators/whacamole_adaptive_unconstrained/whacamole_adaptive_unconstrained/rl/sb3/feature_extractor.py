import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureExtractor(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.spaces.Dict, encoders):

    # Get the encoder models and define features_dim for parent class
    total_concat_size = 0
    extractors = dict()
    for key, encoder in encoders.items():
      extractors[key] = encoder.model
      total_concat_size += encoder.out_features

    # Initialise parent class
    super().__init__(observation_space, features_dim=total_concat_size)

    # Convert into ModuleDict
    self.extractors = nn.ModuleDict(extractors)

  def forward(self, observations) -> th.Tensor:
    encoded_tensor_list = []
    # self.extractors contain nn.Modules that do all the processing.
    for key, extractor in self.extractors.items():
      encoded_tensor_list.append(extractor(observations[key]))
    # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
    return th.cat(encoded_tensor_list, dim=1)
