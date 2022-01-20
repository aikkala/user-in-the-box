import gym
import os
import sys
import numpy as np
from stable_baselines3 import PPO
import re
import argparse

from UIB.utils.logger import StateLogger, ActionLogger


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

  parser = argparse.ArgumentParser(description='Evaluate a policy.')
  parser.add_argument('env_name', type=str,
                      help='name of the environment')
  parser.add_argument('checkpoint_dir', type=str,
                      help='directory where checkpoints are')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='filename of a specific checkpoint (default: None, latest checkpoint is used)')
  parser.add_argument('--num_episodes', type=int, default=10,
                      help='how many episodes are evaluated (default: 10)')
  parser.add_argument('--record', action='store_true', help='enable recording')
  parser.add_argument('--out_file', type=str, default='evaluate.mp4',
                      help='output file for recording if recording is enabled (default: ./evaluate.mp4)')
  parser.add_argument('--logging', action='store_true', help='enable logging')
  parser.add_argument('--state_log_file', default='state_log',
                      help='output file for state log if logging is enabled (default: ./state_log)')
  parser.add_argument('--action_log_file', default='action_log',
                      help='output file for action log if logging is enabled (default: ./action_log)')
  args = parser.parse_args()

  # Get env name and checkpoint dir
  env_name = args.env_name
  checkpoint_dir = args.checkpoint_dir

  # Load latest model if filename not given
  if args.checkpoint is not None:
    model_file = args.checkpoint
  else:
    files = natural_sort(os.listdir(checkpoint_dir))
    model_file = files[-1]

  # Load policy
  print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}')
  model = PPO.load(os.path.join(checkpoint_dir, model_file))

  # Initialise environment
  env_kwargs = {"target_radius_limit": np.array([0.05, 0.15]), "action_sample_freq": 100}
  #env_kwargs = {}
  env = gym.make(env_name, **env_kwargs)

  if args.logging:

    # Initialise log
    state_logger = StateLogger(args.num_episodes)

    # Actions are logged separately to make things easier
    action_logger = ActionLogger(args.num_episodes)

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
      state = env.get_state()
      state_logger.log(episode_idx, {**state, "termination": False, "target_hit": False})

    if args.record:
      imgs.append(grab_pip_image(env))

    # Loop until episode ends
    while not done:

      # Get actions from policy
      action, _states = model.predict(obs, deterministic=True)

      # Take a step
      obs, r, done, info = env.step(action)
      reward += r

      if args.logging:
        action_logger.log(episode_idx, {"step": state["step"], "timestep": state["timestep"], "action": action.copy(),
                                        "ctrl": env.sim.data.ctrl.copy()})
        state = env.get_state()
        state_logger.log(episode_idx, {**state, **info})

      if args.record:
        # Visualise muscle activation
        env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl[2:] * 0.7
        imgs.append(grab_pip_image(env))

    print(f"Episode {episode_idx}: length {env.steps*env.dt} seconds ({env.steps} steps), reward {reward}. ")
    episode_lengths.append(env.steps)
    rewards.append(reward)

  print(f'Average episode length and reward over {args.num_episodes} episodes: '
        f'length {np.mean(episode_lengths)*env.dt} seconds ({np.mean(episode_lengths)} steps), reward {np.mean(rewards)}')

  if args.logging:
    # Output log
    state_logger.save(args.state_log_file)
    action_logger.save(args.action_log_file)
    print(f'Log files have been saved files {args.state_log_file}.pickle and {args.action_log_file}.pickle')

  if args.record:
    # Write the video
    env.write_video(imgs, args.out_file)
    print(f'A recording has been saved to file {args.out_file}')