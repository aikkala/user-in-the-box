import gym
import os
import sys
import numpy as np
from stable_baselines3 import PPO
import re

from UIB.utils.logger import EvaluationLogger


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(env):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  # Grab images
  width, height = env.metadata["imagesize"]
  img = np.flipud(env.sim.render(height=height, width=width, camera_name='for_testing'))
  ocular_img = np.flipud(env.sim.render(height=height//4, width=width//4, camera_name='oculomotor'))

  # Embed ocular image into free image
  i = height - height//4
  j = width - width//4
  img[i:, j:] = ocular_img

  return img

if __name__=="__main__":

  # TODO input parameter parsing using argparse

  # Move into an input parameter?
  render = False

  # Get env name and checkpoint dir
  env_name = sys.argv[1]
  checkpoint_dir = sys.argv[2]

  # Load latest model if filename not given
  if len(sys.argv) == 4:
    model_file = sys.argv[3]
  else:
    files = natural_sort(os.listdir(checkpoint_dir))
    model_file = files[-1]

  # Define number of episodes -- could be input param
  num_episodes = 1000

  # Load previous policy
  print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}')
  model = PPO.load(os.path.join(checkpoint_dir, model_file))

  # Initialise environment
  #env_kwargs = {"target_radius_limit": np.array([0.1, 0.1])}
  env_kwargs = {}
  env = gym.make(env_name, **env_kwargs)

  # Initialise log
  logger = EvaluationLogger(num_episodes)

  # Visualise evaluations
  episode_lengths = []
  rewards = []
  imgs = []
  for episode_idx in range(num_episodes):

    # Reset environment
    obs = env.reset()
    done = False
    reward = 0
    logger.log(episode_idx, {**env.get_state(), "termination": False, "target_hit": False})

    if render:
      imgs.append(grab_pip_image(env))

    # Loop until episode ends
    while not done:

      # Get actions from policy
      action, _states = model.predict(obs, deterministic=False)

      # Take a step
      obs, r, done, info = env.step(action)
      reward += r
      logger.log(episode_idx, {**env.get_state(), **info})

      if render:
        # Visualise muscle activation
        env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl[2:] * 0.7
        imgs.append(grab_pip_image(env))

    print(f"Episode {episode_idx}: length {env.steps}, reward {reward}. ")
    episode_lengths.append(env.steps)
    rewards.append(reward)

  print(f'Average episode length and reward over {num_episodes} episodes: '
        f'length {np.mean(episode_lengths)}, reward {np.mean(rewards)}')

  # Finally, write the video
  if render:
    env.write_video(imgs, 'test.mp4')

  # Output log
  logger.save('log')