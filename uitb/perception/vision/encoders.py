from torch import nn
import torch

def small_cnn(observation_shape, out_features):
  cnn = nn.Sequential(
    nn.Conv2d(in_channels=observation_shape[0], out_channels=8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
    nn.LeakyReLU(),
    nn.Flatten())

  # Compute shape by doing one forward pass
  with torch.no_grad():
    n_flatten = cnn(torch.zeros(observation_shape)[None]).shape[1]

  return nn.Sequential(
    cnn,
    nn.Linear(in_features=n_flatten, out_features=out_features),
    nn.LeakyReLU())