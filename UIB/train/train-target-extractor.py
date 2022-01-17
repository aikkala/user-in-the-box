import gym
from stable_baselines3 import PPO
import os
import numpy as np
import torch

from UIB.archs.regressor import *


def grab_image(env):
  rendered = env.sim.render(height=height, width=width, camera_name='oculomotor', depth=True)
#  rgb = ((rendered[0] / 255.0) - 0.5) * 2
  depth = (rendered[1] - 0.5) * 2
  return np.expand_dims(np.flipud(depth), 0)

def grab_proprioception(env):
  # Ignore eye qpos and qvel for now
  jnt_range = env.sim.model.jnt_range[env.independent_joints]

  qpos = env.sim.data.qpos[env.independent_joints].copy()
  qpos = qpos - jnt_range[:, 0] / (jnt_range[:, 1] - jnt_range[:, 0])
  qpos = (qpos - 0.5) * 2
  qvel = env.sim.data.qvel[env.independent_joints].copy()
  qacc = env.sim.data.qacc[env.independent_joints].copy()

  finger_position = env.sim.data.get_geom_xpos(env.fingertip)
  return np.concatenate([qpos[2:], qvel[2:], qacc[2:], finger_position])

def grab_target(env):
  return np.concatenate([env.target_position + env.target_origin, np.array([env.target_radius])])

if __name__ == "__main__":

  # Load policy
  model_file = "/home/aleksi/Workspace/user-in-the-box/output/current-best-v0/checkpoint/model_100000000_steps.zip"
  print(f'Loading model: {model_file}')
  model = PPO.load(model_file)

  # Image height and width
  width, height = 120, 80  # env.metadata["imagesize"]

  # One epoch is a data collection phase + training phase
  num_epochs = 10000
  num_episodes = 50

  # Initialise gym
  env = gym.make("UIB:mobl-arms-muscles-v0")

  # Initialise a regressor network
  net = SimpleSequentialCNN(height=height, width=width, proprioception_size=grab_proprioception(env).size,
                            seq_max_len=env.max_steps_without_hit+1)

  # Initialise an optimizer
  optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

  # Start the training procedure
  train_error = []
  for epoch in range(num_epochs):

    images = []
    proprioception = []
    targets = []

    # First collect data
    for episode in range(num_episodes):

      # Reset environment
      obs = env.reset()
      done = False

      episode_images = [grab_image(env)]
      episode_proprioception = [grab_proprioception(env)]
      episode_targets = [grab_target(env)]

      # Loop until episode ends
      while not done:

        # Get actions from policy; could use random actions here as well?
        action, _states = model.predict(obs, deterministic=False)

        # Grab image and target before stepping
        episode_images.append(grab_image(env))
        episode_proprioception.append(grab_proprioception(env))
        episode_targets.append(grab_target(env))

        # Take a step
        obs, r, done, info = env.step(action)

        if info["target_hit"] or done:
          images.append(np.array(episode_images, dtype=np.float32))
          proprioception.append(np.array(episode_proprioception, dtype=np.float32))
          targets.append(np.array(episode_targets, dtype=np.float32))
          episode_images = []
          episode_proprioception = []
          episode_targets = []

    images = np.array(images, dtype=object)
    proprioception = np.array(proprioception, dtype=object)
    targets = np.array(targets, dtype=object)

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

      # Do the prediction
      predicted, targets_filled, mask = net(seqs)
      print()
      print(predicted[0, 0])
      print(targets_filled[0, 0])

      # Estimate loss
      masked = (predicted - targets_filled) * mask.unsqueeze(2)
      loss = torch.mean(torch.sum(masked**2, dim=[1,2]) / mask.sum(dim=1))

      # Calculate grad and backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print('Average MSE over sequence:', loss.item())
      train_error.append(loss.item())

    del images, proprioception, targets

  # Save the network
  torch.save(net.state_dict(), 'regressor')

  import matplotlib.pyplot as pp
  pp.plot(train_error)
  pp.savefig('train_error')