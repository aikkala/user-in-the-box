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
from collections import defaultdict
import matplotlib.pyplot as pp
from mujoco_py.modder import TextureModder

from UIB.utils.logger import StateLogger, ActionLogger
from UIB.utils.functions import output_path

# Use publication mode (see the function 'set_publication_mode' for explanation of what it does)
use_publication_mode = False


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(env, modder):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  if use_publication_mode:
    set_publication_mode(env, modder, True, skybox=True)

  # Define image size
  width, height = env.metadata["imagesize"]

  if hasattr(env, 'target_plane_geom_idx'):
    # Visualise target plane
    env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0.2

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

  if use_publication_mode:
    set_publication_mode(env, modder, False, skybox=True)

  return img

def set_publication_mode(env, modder, value, skybox=True):
  # This function only improves the visual aspects of the recorded episodes, like adds a light blue color skybox
  # (instead of default black), and changes the floor color. This hack makes the code run slower, and there's no point
  # in using it except for producing visually more pleasing videos.

  # Do nothing for remote driving env
  if env.spec._env_name == "mobl-arms-remote-driving":
    return

  if value:

    # Visualise muscle activation
    env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl * 0.7

    # Change floor color
    #env.model.geom_rgba[env.model._geom_name2id["floor"]][3] = 0
    env.model.geom_rgba[env.model._geom_name2id["floor"]] = [1, 1, 1, 0.2]
    env.model.geom_matid[env.model._geom_name2id["floor"]] = 1

    if skybox:
      # Set skybox to light blue
      skybox = modder.get_texture('skybox')
      rgb = np.zeros((skybox.height, skybox.width, 3))
      rgb[:, :] = np.array([204, 230, 255])
      modder.set_rgb('skybox', rgb)

    # Activate additional light
    #env.model.light_active[env.model._light_name2id["additional"]] = 1

  if not value:

    # Set back to default tendon color, might affect policy
    env.model.tendon_rgba[:, 0] = 0.95

    if skybox:
      # Set skybox back to black
      modder = TextureModder(env.sim)
      skybox = modder.get_texture('skybox')
      arr = np.zeros((skybox.height, skybox.width, 3))*0
      modder.set_rgb('skybox', arr)

    # Change floor color back
    #env.model.geom_rgba[env.model._geom_name2id["floor"]][3] = 1
    env.model.geom_matid[env.model._geom_name2id["floor"]] = 0
    env.model.geom_rgba[env.model._geom_name2id["floor"]] = [0.8, 0.6, 0.4, 1.0]

    # Disable additional light
    #env.model.light_active[env.model._light_name2id["additional"]] = 0

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate a policy.')
  parser.add_argument('config_file', type=str,
                      help='config file used for training the model')
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


  # If config file is given load that
  with open(args.config_file, 'rb') as file:
    config = pickle.load(file)

  # Define output directories
  env_name = config["env_name"]
  run_folder = Path(args.config_file).parent.absolute()
  checkpoint_dir = os.path.join(run_folder, 'checkpoints')
  evaluate_dir = os.path.join(output_path(), config["env_name"], config["name"])

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
  env_kwargs["freq_curriculum"] = lambda : 1.0
  env_kwargs["evaluate"] = True
  env_kwargs["target_radius_limit"] = np.array([0.05, 0.1])
  #env_kwargs["target_halfsize_limit"] = np.array([0.3, 0.3])

  # Use deterministic actions?
  deterministic = False

  print(f"env_kwargs are: {env_kwargs}")

  # Initialise environment
  env = gym.make(env_name, **env_kwargs)

  if args.logging:

    # Initialise log
    state_logger = StateLogger(args.num_episodes, keys=env.get_state().keys())

    # Actions are logged separately to make things easier
    action_logger = ActionLogger(args.num_episodes)

  # Visualise evaluations
  statistics = defaultdict(list)
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
      modder = TextureModder(env.sim)
      #set_publication_mode(env, modder, True, skybox=True)
      imgs.append(grab_pip_image(env, modder))

    # Loop until episode ends
    while not done:

      # Get actions from policy
      action, _states = model.predict(obs, deterministic=deterministic)

      # Take a step
      obs, r, done, info = env.step(action)
      reward += r
      print(env.steps)

      if args.logging:
        action_logger.log(episode_idx, {"step": state["step"], "timestep": state["timestep"], "action": action.copy(),
                                        "ctrl": env.sim.data.ctrl.copy(), "reward": r})
        state = env.get_state()
        state.update(info)
        state_logger.log(episode_idx, state)

      if args.record and not done:
        imgs.append(grab_pip_image(env, modder))

    print(f"Episode {episode_idx}: {env.get_episode_statistics_str()}")

    episode_statistics = env.get_episode_statistics()
    for key in episode_statistics:
      statistics[key].append(episode_statistics[key])

  print(f'Averages over {args.num_episodes} episodes (std in parenthesis):',
        ', '.join(['{}: {:.2f} ({:.2f})'.format(k, np.mean(v), np.std(v)) for k, v in statistics.items()]))

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