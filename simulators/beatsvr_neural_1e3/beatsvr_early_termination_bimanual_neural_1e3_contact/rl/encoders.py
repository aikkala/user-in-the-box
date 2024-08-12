from torch import nn
import torch
from typing import final


class BaseEncoder:
  """ Defines an encoder. Note that in stable baselines training we only use the model definitions given here. We don't
  e.g. after training set the encoder parameters into these objects, but instead use the ones saved/loaded by stable
  baselines. In other words, these encoders are not used after during/after training, only to initialise the encoders
  for stable baselines. """

  def __init__(self, observation_shape, **kwargs):
    self._observation_shape = observation_shape

    # Define a PyTorch model (e.g. using torch.nn.Sequential)
    self._model = None

    # We assume all encoders output a vector with self._out_features elements in it
    self.out_features = None

  @final
  @property
  def model(self):
    return self._model

class Identity(BaseEncoder):
  """ Define an identity encoder. Used when no encoder has been defined. Can only be used for one-dimensional
  observations. """

  def __init__(self, observation_shape):
    super().__init__(observation_shape)
    if len(observation_shape) > 1:
      raise RuntimeError("You must not use the Identity encoder for higher dimensional observations. Use an encoder"
                         "that maps the high dimensional observations into one dimensional vectors.")
    self._model = torch.nn.Identity()
    self.out_features = observation_shape[0]

class SmallCNN(BaseEncoder):

  def __init__(self, observation_shape, out_features):
    super().__init__(observation_shape)
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

    self._model = nn.Sequential(
      cnn,
      nn.Linear(in_features=n_flatten, out_features=out_features),
      nn.LeakyReLU())
    self.out_features = out_features


class OneLayer(BaseEncoder):

  def __init__(self, observation_shape, out_features):
    super().__init__(observation_shape)
    self.out_features = out_features
    self._model = nn.Sequential(
      nn.Linear(self._observation_shape[0], out_features),
      nn.LeakyReLU())
