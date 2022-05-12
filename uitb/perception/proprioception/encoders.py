from torch import nn

def one_layer(observation_shape, out_features):
  return nn.Sequential(
    nn.Linear(observation_shape[0], out_features),
    nn.LeakyReLU())