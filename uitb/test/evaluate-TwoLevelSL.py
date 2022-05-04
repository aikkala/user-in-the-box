import gym
import os
import sys
import numpy as np
from stable_baselines3 import PPO
import re
import argparse

from UIB.utils.logger import EvaluationLogger
from UIB.envs_old_to_be_removed.mobl_arms import TwoLevelSL


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

  parser = argparse.ArgumentParser(description='Evaluate a policy.')
  parser.add_argument('--num_episodes', type=int, default=10,
                      help='how many episodes are evaluated (default: 10)')
  parser.add_argument('--record', action='store_true', help='enable recording')
  parser.add_argument('--out_file', type=str, default='evaluate-TwoLevelSL.mp4',
                      help='output file for recording if recording is enabled (default: ./evaluate-TwoLevelSL.mp4)')
  parser.add_argument('--logging', action='store_true', help='enable logging')
  parser.add_argument('--log_file', default='log-TwoLevelSL',
                      help='output file for log if logging is enabled (default: ./log-TwoLevelSL)')
  args = parser.parse_args()

  # Initialise TwoLevelSL
  env_kwargs = {"target_radius_limit": np.array([0.05, 0.15]), "action_sample_freq": 100}
  env = TwoLevelSL(**env_kwargs)

  if args.logging:
    # Initialise log
    logger = EvaluationLogger(args.num_episodes)

  # Visualise evaluations
  episode_lengths = []
  rewards = []
  imgs = []
  for episode_idx in range(args.num_episodes):

    # Reset environment
    obs = env.reset()
    done = False
    reward = 0

    if args.logging:
      logger.log(episode_idx, {**env.get_state(), "termination": False, "target_hit": False})

    if args.record:
      imgs.append(grab_pip_image(env))

    # Loop until episode ends
    while not done:

      # Get actions from policy
      action  = env.get_action(obs)

      # Take a step
      obs, r, done, info = env.step(action)
      reward += r

      if args.logging:
        logger.log(episode_idx, {**env.get_state(), **info})

      if args.record:
        # Visualise muscle activation
        env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl[2:] * 0.7
        imgs.append(grab_pip_image(env))

    print(f"Episode {episode_idx}: length {env.steps}, reward {reward}. ")
    episode_lengths.append(env.steps)
    rewards.append(reward)

  print(f'Average episode length and reward over {args.num_episodes} episodes: '
        f'length {np.mean(episode_lengths)}, reward {np.mean(rewards)}')


  if args.logging:
    # Output log
    logger.save(args.log_file)
    print(f'A log has been saved to file {args.log_file}.pickle')

  if args.record:
    # Write the video
    env.write_video(imgs, args.out_file)
    print(f'A recording has been saved to file {args.out_file}')