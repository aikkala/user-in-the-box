import gym
import os
import sys
import numpy as np
from stable_baselines3 import PPO
import re
import argparse
import scipy.ndimage
import pickle
from pathlib import Path
from tqdm import tqdm

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

  if hasattr(env, 'target_plane_geom_idx'):
    # Visualise target plane
    env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0.1

  # Grab images
  img = np.flipud(env.sim.render(height=height, width=width, camera_name='for_testing'))
  ocular_img = np.flipud(env.sim.render(height=env.ocular_image_height, width=env.ocular_image_width,
                                        camera_name='oculomotor'))

  if hasattr(env, 'target_plane_geom_idx'):
    # Disable target plane
    env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0.0

  # Resample
  resample_factor = 3
  resample_height = env.ocular_image_height*resample_factor
  resample_width = env.ocular_image_width*resample_factor
  resampled_img = np.zeros((resample_height, resample_width, 3), dtype=np.uint8)
  for channel in range(3):
    resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)

  # Embed ocular image into free image
  i = height - resample_height
  j = width - resample_width
  img[i:, j:] = resampled_img

  return img

def generate_target_position(env):
  env.spawn_target()
  return env.target_position.copy()


def run_trial(target_position, radius, episode_idx):

  # Set target position and radius
  env.set_target_position(target_position)
  env.set_target_radius(radius)

  if args.logging:
    state = env.get_state()
    state_logger.log(episode_idx, state)

  if args.record:
    imgs.append(grab_pip_image(env))

  obs = env.get_observation()

  # Loop until trial ends
  while True:

    # Get actions from policy
    action, _states = model.predict(obs, deterministic=False)

    # Take a step
    obs, r, done, info = env.step(action)

    # This is a hack for logging and visualisation purposes; otherwise a new randomly sampled target will appear
    # in the last frame of the video, or in one line in the log
    if info["target_hit"]:
      env.set_target_position(target_position)
      env.set_target_radius(radius)

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

    if info["target_hit"]:
      return True
    elif done:
      return False


if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate a policy.')
  parser.add_argument('config_file', type=str,
                      help='config file used for training the model')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='filename of a specific checkpoint (default: None, latest checkpoint is used)')
  parser.add_argument('--num_movements', type=int, default=10,
                      help='how many different movements are evaluated (default: 10)')
  parser.add_argument('--num_episodes', type=int, default=5,
                      help='how many episodes are evaluated per movement (default: 5)')
  parser.add_argument('--record', action='store_true', help='enable recording')
  parser.add_argument('--out_file', type=str, default='evaluate.mp4',
                      help='output file for recording if recording is enabled (default: evaluate.mp4)')
  parser.add_argument('--logging', action='store_true', help='enable logging')
  parser.add_argument('--state_log_file', default='state_log',
                      help='output file for state log if logging is enabled (default: state_log)')
  parser.add_argument('--action_log_file', default='action_log',
                      help='output file for action log if logging is enabled (default: action_log)')
  args = parser.parse_args()


  # If config file is given load that
  with open(args.config_file, 'rb') as file:
    config = pickle.load(file)

  # Define output directories
  env_name = config["env_name"]
  run_folder = Path(args.config_file).parent.absolute()
  checkpoint_dir = os.path.join(run_folder, 'checkpoints')
  evaluate_dir = os.path.join(output_path(), config["env_name"], config["name"], 'repeated-movements')

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

  # Load env_kwargs
  env_kwargs = config["env_kwargs"]

  # Override kwargs
  env_kwargs["action_sample_freq"] = 100
  env_kwargs["max_trials"] = 2
  env_kwargs["evaluate"] = False

  print(f"env_kwargs are: {env_kwargs}")

  # Initialise environment
  env = gym.make(env_name, **env_kwargs)

  # Define radii that are used
  num_radius = 5
  radius = np.linspace(*env.target_radius_limit, num_radius)

  for movement_idx in tqdm(range(args.num_movements)):

    # Generate locations for first and second target
    pos1 = generate_target_position(env)
    pos2 = generate_target_position(env)

    # Create a folder
    movement_dir = os.path.join(evaluate_dir, f'movement_{str(movement_idx).zfill(len(str(args.num_movements)))}')
    os.makedirs(movement_dir, exist_ok=True)

    for radius_idx, r in enumerate(radius):

      # Create a folder
      radius_dir = os.path.join(movement_dir, f'radius_{str(radius_idx).zfill(len(str(num_radius)))}')
      os.makedirs(radius_dir, exist_ok=True)

      if args.logging:
        # Initialise log
        state_logger = StateLogger(args.num_episodes, keys=env.get_state().keys())

        # Actions are logged separately to make things easier
        action_logger = ActionLogger(args.num_episodes)

      if args.record:
        imgs = []

      # Repeat this movement for this radius for a few times
      for episode_idx in range(args.num_episodes):

        env.reset()

        if run_trial(pos1, r, episode_idx):
          if run_trial(pos2, r, episode_idx):

            if args.logging:
              # Output log
              state_logger.save(os.path.join(radius_dir, args.state_log_file))
              action_logger.save(os.path.join(radius_dir, args.action_log_file))
              #print(f'Log files have been saved files {os.path.join(radius_dir, args.state_log_file)}.pickle and '
              #      f'{os.path.join(radius_dir, args.action_log_file)}.pickle')

            if args.record:
              # Write the video
              env.write_video(imgs, os.path.join(radius_dir, args.out_file))
              #print(f'A recording has been saved to file {os.path.join(radius_dir, args.out_file)}')