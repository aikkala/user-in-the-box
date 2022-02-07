import gym
import os
import sys
import numpy as np
from stable_baselines3 import PPO
import re
import argparse
import scipy.ndimage

from UIB.utils.logger import StateLogger, ActionLogger
from UIB.utils.functions import output_path


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(env):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  # Define image size
  width, height = env.metadata["imagesize"]

  # Visualise target plane
  env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0.1

  # Grab images
  img = np.flipud(env.sim.render(height=height, width=width, camera_name='for_testing'))
  ocular_img = np.flipud(env.sim.render(height=env.height, width=env.width, camera_name='oculomotor'))

  # Disable target plane
  env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0.0

  # Resample
  resample_factor = 3
  resample_height = env.height*resample_factor
  resample_width = env.width*resample_factor
  resampled_img = np.zeros((resample_height, resample_width, 3), dtype=np.uint8)
  for channel in range(3):
    resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)

  # Embed ocular image into free image
  i = height - resample_height
  j = width - resample_width
  img[i:, j:] = resampled_img

  return img

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate a policy.')
  parser.add_argument('env_name', type=str,
                      help='name of the environment')
  parser.add_argument('checkpoint_dir', type=str,
                      help='directory where checkpoints are')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='filename of a specific checkpoint (default: None, latest checkpoint is used)')
  parser.add_argument('--config_file', type=str, default=None,
                      help='config file used for training the model (default: None)')
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


  # If config file is given load that
  if args.config_file is not None:
    config = np.load(args.config_file)

    # Define output directories
    env_name = config["env_name"]
    model_dir = os.path.join(output_path(), config["env_name"], 'trained-models', config["name"])
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    evaluate_dir = os.path.join(output_path(), config["env_name"], 'evaluate', config["name"])

  else:

    # Get env name and checkpoint dir
    env_name = args.env_name
    checkpoint_dir = args.checkpoint_dir
    evaluate_dir = os.path.join(checkpoint_dir, '..', 'evaluate')

  # Make sure output dir exists
  os.makedirs(evaluate_dir, exist_ok=True)

  # Load latest model if filename not given
  if args.checkpoint is not None:
    model_file = args.checkpoint
  else:
    files = natural_sort(os.listdir(checkpoint_dir))
    model_file = files[-1]

  # Load policy
  print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}')
  model = PPO.load(os.path.join(checkpoint_dir, model_file))

  env_kwargs = {}
  if args.config_file is not None:
    env_kwargs = config["env_kwargs"]
  else:
    # Load env_kwargs if they have been saved
    env_kwargs_file = os.path.join(checkpoint_dir, "env_kwargs.npy")
    if os.path.exists(env_kwargs_file):
      print(f"Loading env_kwargs: {env_kwargs_file}")
      env_kwargs = np.load(env_kwargs_file, allow_pickle=True).item()

  # Override kwargs
  env_kwargs["action_sample_freq"] = 100

  print(f"env_kwargs are: {env_kwargs}")

  # Initialise environment
  env = gym.make(env_name, **env_kwargs)

  if args.logging:

    # Initialise log
    state_logger = StateLogger(args.num_episodes, keys=env.get_state().keys())

    # Actions are logged separately to make things easier
    action_logger = ActionLogger(args.num_episodes)

  # Visualise evaluations
  episode_lengths = []
  num_trials = []
  rewards = []
  imgs = []
  for episode_idx in range(args.num_episodes):

    # Reset environment
    obs = env.reset()
    done = False
    reward = 0

    if args.logging:
      state = env.get_state()
      state_logger.log(episode_idx, state)

    if args.record:
      imgs.append(grab_pip_image(env))

    # Loop until episode ends
    while not done:

      # Get actions from policy
      action, _states = model.predict(obs, deterministic=False)

      # Take a step
      obs, r, done, info = env.step(action)
      reward += r

      if args.logging:
        action_logger.log(episode_idx, {"step": state["step"], "timestep": state["timestep"], "action": action.copy(),
                                        "ctrl": env.sim.data.ctrl.copy(), "reward": r})
        state = env.get_state()
        state.update(info)
        state_logger.log(episode_idx, state)

      if args.record and not done:
        # Visualise muscle activation
        env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl * 0.7
        imgs.append(grab_pip_image(env))

    print(f"Episode {episode_idx}: {env.trial_idx} targets hit, length {env.steps*env.dt} seconds ({env.steps} steps), reward {reward}. ")
    episode_lengths.append(env.steps)
    rewards.append(reward)
    num_trials.append(env.trial_idx)

  print(f'Averages over {args.num_episodes} episodes: '
        f'targets hit {np.mean(num_trials)}, length {np.mean(episode_lengths)*env.dt} seconds ({np.mean(episode_lengths)} steps), reward {np.mean(rewards)}')

  if args.logging:
    # Output log
    state_logger.save(os.path.join(evaluate_dir, args.state_log_file))
    action_logger.save(os.path.join(evaluate_dir, args.action_log_file))
    print(f'Log files have been saved files {os.path.join(evaluate_dir, args.state_log_file)}.pickle and '
          f'{os.path.join(evaluate_dir, args.action_log_file)}.pickle')

  if args.record:
    # Write the video
    env.write_video(imgs, os.path.join(evaluate_dir, args.out_file))
    print(f'A recording has been saved to file {os.path.join(evaluate_dir, args.out_file)}')