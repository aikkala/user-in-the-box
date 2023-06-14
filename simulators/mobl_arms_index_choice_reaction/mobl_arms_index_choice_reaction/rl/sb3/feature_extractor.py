import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureExtractor(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.spaces.Dict, extractors):

    # Sample a fake observation
    fake_observation = observation_space.sample()

    # Convert None extractors into identity layers
    total_concat_size = 0
    for key, extractor in extractors.items():
      if extractor is None:
        extractors[key] = nn.Identity()
      fake_obs_tensor = th.from_numpy(fake_observation[key])[None]
      if len(extractors[key]._modules) > 0:
        fake_obs_tensor = fake_obs_tensor.to(next(extractors[key].parameters()).device)
      total_concat_size += extractors[key](fake_obs_tensor).shape[1]

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
