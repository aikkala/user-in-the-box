import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np


class SimpleSequentialCNN(nn.Module):

  def __init__(self, height, width, proprioception_size, seq_max_len):
    super().__init__()

    self.height = height
    self.width = width
    self.proprioception_size = proprioception_size
    self.seq_max_len = seq_max_len
    self.nchannels = 1

    self.cnn_base = nn.Sequential(
      nn.Conv2d(in_channels=self.nchannels, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      nn.BatchNorm2d(8),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(),
      nn.Flatten()
    )

    # Figure out output size
    out = self.cnn_base(torch.zeros(1, self.nchannels, height, width))
    self.hidden_size = 256

    #self.encoder_out = nn.Linear(in_features=np.prod(out.shape[-1]), out_features=self.hidden_size)
    self.cnn_base = nn.Sequential(*self.cnn_base,
                                  nn.Linear(in_features=np.prod(out.shape[-1]), out_features=self.hidden_size),
                                  nn.LeakyReLU())

    self.encoder = nn.Linear(in_features=self.hidden_size+self.proprioception_size, out_features=self.hidden_size)

    # Finally the LSTM part
    self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
    self.output = nn.Sequential(nn.Linear(in_features=self.hidden_size, out_features=int(self.hidden_size/2)),
                                nn.LeakyReLU(),
                                nn.Linear(in_features=int(self.hidden_size/2), out_features=4))

    # Hidden state of LSTM for inference
    self.lstm_hidden = torch.zeros(1, 1, self.hidden_size)
    self.lstm_cell = torch.zeros(1, 1, self.hidden_size)


  def calculate_loss(self, seqs):

    # Do predictions
    predicted, targets_reshaped, mask = self.forward(seqs)
    print()
    print(predicted[0, 0])
    print(predicted[0, int(sum(mask[0]) - 1)])
    print(targets_reshaped[0, 0])

    # Estimate loss
    masked = (predicted - targets_reshaped) * mask.unsqueeze(2)
    loss = torch.mean(torch.sum(masked ** 2, dim=[1, 2]) / mask.sum(dim=1))

    return loss, predicted, targets_reshaped

  def forward(self, input):

    batch_size = len(input)

    # Parse input
    encoded = torch.zeros(batch_size, self.seq_max_len, self.hidden_size)
    targets = torch.zeros(batch_size, self.seq_max_len, 4)
    mask = torch.zeros(batch_size, self.seq_max_len, dtype=torch.float)
    for idx, (img, prop, tgt) in enumerate(input):
      encoded[idx, :img.shape[0]] = self.encoder(torch.concat([self.cnn_base(torch.as_tensor(img)), torch.as_tensor(prop)], dim=1))
      targets[idx, :tgt.shape[0]] = torch.as_tensor(tgt)
      mask[idx, :tgt.shape[0]] = 1

    lstm_out, (hidden, cell) = self.lstm(encoded)
    lstm_out = lstm_out.reshape(batch_size*self.seq_max_len, self.hidden_size)
    estimate = self.output(lstm_out)

    # estimate for radius needs to be positive
    #radius = F.softplus(estimate[:, 3])
    #estimate = torch.cat([estimate[:, :3], radius.unsqueeze(1)], dim=1)

    # Reshape again
    estimate = estimate.reshape(batch_size, self.seq_max_len, 4)

    return estimate, targets, mask

  def estimate(self, img, prop):

    with torch.no_grad():

      # Encode
      encoded = self.encoder(torch.concat([self.cnn_base(torch.as_tensor(img.copy()).unsqueeze(0).float()),
                                           torch.as_tensor(prop).unsqueeze(0).float()], dim=1))

      # Run through lstm
      lstm_out, (h, c) = self.lstm(encoded.unsqueeze(0), (self.lstm_hidden, self.lstm_cell))
      self.lstm_hidden = h
      self.lstm_cell = c

      # Get the output
      estimate = self.output(lstm_out)

    return estimate

  def initialise(self):
    self.lstm_hidden = torch.zeros(1, 1, self.hidden_size)
    self.lstm_cell = torch.zeros(1, 1, self.hidden_size)


class SimpleCNN(nn.Module):

  def __init__(self, height, width, proprioception_size):
    super().__init__()

    self.height = height
    self.width = width
    self.proprioception_size = proprioception_size

    self.current_estimate = None
    self.alpha = 0.1
    self.nchannels = 2

    self.cnn_base = nn.Sequential(
      nn.Conv2d(in_channels=self.nchannels, out_channels=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      nn.BatchNorm2d(8),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      nn.BatchNorm2d(16),
      nn.LeakyReLU(),
      #nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      #nn.BatchNorm2d(32),
      #nn.LeakyReLU(),
      #nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      #nn.BatchNorm2d(64),
      #nn.LeakyReLU(),
      #nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      #nn.BatchNorm2d(128),
      #nn.LeakyReLU(),
      #nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      #nn.BatchNorm2d(256),
      #nn.LeakyReLU(),
      nn.Flatten()
    )

    # Figure out output size
    out = self.cnn_base(torch.zeros(1, self.nchannels, height, width))
    self.hidden_size = 256

    self.cnn_base = nn.Sequential(*self.cnn_base,
                                  nn.Linear(in_features=np.prod(out.shape[-1]), out_features=self.hidden_size),
                                  nn.LeakyReLU())

    self.encoder = nn.Linear(in_features=self.hidden_size+self.proprioception_size, out_features=self.hidden_size)

    # Finally the output
    self.output = nn.Sequential(#nn.Linear(in_features=self.hidden_size, out_features=int(self.hidden_size/2)),
                                #nn.LeakyReLU(),
                                nn.Linear(in_features=int(self.hidden_size), out_features=3))


  def calculate_loss(self, seqs):

    # Do predictions
    predicted, targets_reshaped = self.forward(seqs)
    print()
    print(predicted[0])
    print(targets_reshaped[0])

    # Estimate loss
    loss_fn = nn.HuberLoss()
    loss = loss_fn(predicted, targets_reshaped)
    #loss = torch.mean(torch.sum(torch.abs(predicted - targets_reshaped), dim=1))

    return loss, predicted, targets_reshaped

  def forward(self, input):

    # Parse input
    encoded = []
    targets = []
    for idx, (img, prop, tgt) in enumerate(input):
      encoded.append(self.encoder(torch.concat([self.cnn_base(torch.as_tensor(img[::10])), torch.as_tensor(prop[::10])], dim=1)))
      targets.append(torch.as_tensor(tgt[::10]))

    estimate = self.output(torch.cat(encoded))

    return estimate, torch.cat(targets)

  def initialise(self):
    self.current_estimate = None

  def estimate(self, img, prop):

    with torch.no_grad():

      # Encode
      encoded = self.encoder(torch.concat([self.cnn_base(torch.as_tensor(img.copy()).unsqueeze(0).float()),
                                           torch.as_tensor(prop).unsqueeze(0).float()], dim=1))

      # Get the output
      estimate = self.output(encoded)

    if self.current_estimate is None:
      self.current_estimate = estimate.numpy().copy()
    else:
      self.current_estimate = (1-self.alpha)*self.current_estimate + self.alpha*estimate.numpy()

    return self.current_estimate.copy()


class VGG11(nn.Module):

  def __init__(self, height, width):
    super().__init__()

    # Grab a pretrained VGG11 model
    vgg = models.vgg11_bn(pretrained=False)

    # Extract only the base of the model
    base, adaptive_filter = list(vgg.children())[0:2]

    # Switch the first CNN to 1 channel version
    all_but_first_cnn = base[1:]
    first_cnn = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))

    # Create the new base
    new_base = nn.Sequential(first_cnn, *all_but_first_cnn, adaptive_filter)

    # Figure out output size
    out = new_base(torch.zeros(1, 1, height, width))

    # Construct a model for predicting target coordinates and radius (i.e. 4 outputs)
    self.net = nn.Sequential(*new_base, nn.Flatten(),
                             nn.Linear(in_features=np.prod(out.shape[1:]), out_features=4))

  def forward(self, input):
    estimate = self.net(input)
    # estimate for radius needs to be positive
    radius = F.softplus(estimate[:, 3])
    return torch.cat([estimate[:, :3], radius.unsqueeze(1)], dim=1)


class VGG11Transfer(nn.Module):

    def __init__(self, height, width):
      super().__init__()

      # Grab a pretrained VGG11 model
      vgg = models.vgg11_bn(pretrained=True)

      # Extract only the base of the model
      self.features = nn.Sequential(*(list(vgg.children())[0:2]))

      # Figure out self.features output size
      out = self.features(torch.zeros(1, 3, height, width))

      # Construct a model for predicting target coordinates and radius (i.e. 4 outputs)
      self.net = nn.Sequential(self.features, nn.Flatten(),
                               nn.Linear(in_features=np.prod(out.shape[1:]), out_features=4))

      # Perhaps freeze couple of the first layers?
      for param in self.features.parameters():
        param.requires_grad = False


    def forward(self, input):
      estimate = self.net(input)
      # estimate for radius needs to be positive
      radius = F.softplus(estimate[:, 3])
      return torch.cat([estimate[:, :3], radius.unsqueeze(1)], dim=1)
