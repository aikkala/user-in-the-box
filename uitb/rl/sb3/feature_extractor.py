import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureExtractor(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.spaces.Dict, extractors):
    # We do not know features-dim here before going over all the items,
    # so put something dummy for now. PyTorch requires calling
    # nn.Module.__init__ before adding modules
    super().__init__(observation_space, features_dim=1)

    fake_observation = observation_space.sample()

    # Convert None extractors into identity layers
    total_concat_size = 0
    for key, extractor in extractors.items():
      if extractor is None:
        extractors[key] = nn.Identity()
      total_concat_size += extractors[key](th.from_numpy(fake_observation[key])[None]).shape[1]

    # Convert into ModuleDict
    self.extractors = nn.ModuleDict(extractors)

    # Update the features dim manually
    self._features_dim = total_concat_size

  def forward(self, observations) -> th.Tensor:
    encoded_tensor_list = []
    # self.extractors contain nn.Modules that do all the processing.
    for key, extractor in self.extractors.items():
      encoded_tensor_list.append(extractor(observations[key]))
    # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
    return th.cat(encoded_tensor_list, dim=1)
