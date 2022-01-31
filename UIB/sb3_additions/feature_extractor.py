import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VisualAndProprioceptionExtractor(BaseFeaturesExtractor):
  def __init__(self, observation_space: gym.spaces.Dict):
    # We do not know features-dim here before going over all the items,
    # so put something dummy for now. PyTorch requires calling
    # nn.Module.__init__ before adding modules
    super(VisualAndProprioceptionExtractor, self).__init__(observation_space, features_dim=1)

    extractors = {}

    total_concat_size = 0
    # We need to know size of the output of this extractor,
    # so go over all the spaces and compute output feature sizes
    for key, subspace in observation_space.spaces.items():
      if key == "visual":
        fake_observation = th.as_tensor(observation_space["visual"].sample()[None])
        # Run through a simple CNN
        cnn = nn.Sequential(
          nn.Conv2d(in_channels=fake_observation.shape[1], out_channels=16, kernel_size=5, padding=0, stride=2),
          nn.LeakyReLU(),
          nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=0, stride=2),
          nn.LeakyReLU(),
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0, stride=1),
          nn.LeakyReLU(),
          nn.Flatten())

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = cnn(fake_observation).shape[1]

        extractors[key] = nn.Sequential(
          cnn,
          nn.Linear(in_features=n_flatten, out_features=256),
          nn.LeakyReLU())
        #total_concat_size += subspace.shape[1] * subspace.shape[2] * 1
        total_concat_size += 256
      elif key == "proprioception":
        # Run through a simple MLP
        extractors[key] = nn.Sequential(
          nn.Linear(subspace.shape[0], 128),
          nn.LeakyReLU())
        total_concat_size += 128
      elif key == "ocular":
        # Do nothing with this for now
        pass

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
