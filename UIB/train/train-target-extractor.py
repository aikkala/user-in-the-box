import gym
from stable_baselines3 import PPO
import os
import numpy as np
import torch
import pathlib
import matplotlib.pyplot as pp

from UIB.archs.regressor import *


if __name__ == "__main__":

  # Get path of this file
  project_path = pathlib.Path(__file__).parent.absolute()

  # Load policy
  model_file = os.path.join(project_path, "../../output/current-best-v0/checkpoint/model_100000000_steps.zip")
  print(f'Loading model: {model_file}')
  model = PPO.load(model_file)

  # Image height and width
  width, height = 120, 80  # env.metadata["imagesize"]

  # One epoch is a data collection phase + training phase
  num_epochs = 10000
  num_episodes = 50

  # Initialise gym
  env_kwargs = {"target_radius_limit": np.array([0.05, 0.15])}
  env = gym.make("UIB:mobl-arms-muscles-v0", **env_kwargs)

  # Initialise a regressor network
#  net = SimpleSequentialCNN(height=height, width=width, env=env,
#                            seq_max_len=env.max_steps_without_hit+1)
  net = SimpleCNN(height=height, width=width, proprioception_size=env.grab_proprioception().size)

  # Initialise an optimizer
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

  # Start the training procedure
  train_error = []
  for epoch in range(num_epochs):

    images_tmp = []
    proprioception_tmp = []
    targets_tmp = []

    # First collect data
    for episode in range(num_episodes):

      # Reset environment
      obs = env.reset()
      done = False

      episode_images = [env.grab_image(height=height, width=width)]
      episode_proprioception = [env.grab_proprioception()]
      episode_targets = [env.grab_target()]

      # Loop until episode ends
      while not done:

        # Get actions from policy; could use random actions here as well?
        if np.random.rand() < 0.2:
          action, _states = model.predict(obs, deterministic=False)
        else:
          action = env.action_space.sample()

        # Grab image and target before stepping
        episode_images.append(env.grab_image(height=height, width=width))
        episode_proprioception.append(env.grab_proprioception())
        episode_targets.append(env.grab_target())

        # Take a step
        obs, r, done, info = env.step(action)

        if info["target_hit"] or done:
          # Save current sequence
          images_tmp.append(np.array(episode_images, dtype=np.float32))
          proprioception_tmp.append(np.array(episode_proprioception, dtype=np.float32))
          targets_tmp.append(np.array(episode_targets, dtype=np.float32))
          episode_images = []
          episode_proprioception = []
          episode_targets = []

    # Need to initialise empty arrays with dtype=object to force ragged arrays
    images = np.empty(len(images_tmp), dtype=object)
    proprioception = np.empty(len(proprioception_tmp), dtype=object)
    targets = np.empty(len(targets_tmp), dtype=object)

    images[:] = images_tmp
    proprioception[:] = proprioception_tmp
    targets[:] = targets_tmp

    # Start training
    for iteration in range(1):

      # Grab random samples
      #idxs = np.random.choice(images.shape[0], batch_size)
      idxs = np.arange(images.shape[0])
      train_images = images[idxs]
      train_proprioception = proprioception[idxs]
      train_targets = targets[idxs]

      # Choose varying frame skip
      frame_skip = np.random.randint(1, 2)

      # Get sequences
      seqs = [(img[::frame_skip], prop[::frame_skip], tgt[::frame_skip]) for img, prop, tgt in
              zip(train_images, train_proprioception, train_targets)]

      # Calculate loss
      loss, predicted, targets_reshaped = net.calculate_loss(seqs)

      # Calculate grad and backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f'epoch {epoch}: Average MSE over sequence: {loss.item()}')
      train_error.append(np.log(loss.item()))

    del images, proprioception, targets

    if (epoch % 10) == 0:
      # Save the network
      model_file = os.path.join(project_path, '../archs/regressor')
      torch.save(net.state_dict(), model_file)

      train_error_file = os.path.join(project_path, '../archs/train_error')
      pp.plot(train_error)
      pp.savefig(train_error_file)
      pp.close()
