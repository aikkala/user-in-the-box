import os
import numpy as np
from stable_baselines3 import PPO
import re
import argparse
import scipy.ndimage
from collections import defaultdict
import matplotlib.pyplot as pp

from uitb.utils.logger import StateLogger, ActionLogger
from uitb.simulator import Simulator


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(env):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  # Grab images
  img = env.viewer.render()

  ocular_img = None
  for module in env.perception.perception_modules:
    if module.modality == "vision":
      # TODO would be better to have a class function that returns "human-viewable" rendering of the observation;
      #  e.g. in case the vision model has two cameras, or returns a combination of rgb + depth images etc.
      ocular_img, _ = module.render()

  if ocular_img is not None:

    # Resample
    resample_factor = 2
    resample_height = ocular_img.shape[0]*resample_factor
    resample_width = ocular_img.shape[1]*resample_factor
    resampled_img = np.zeros((resample_height, resample_width, 3), dtype=np.uint8)
    for channel in range(3):
      resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)

    # Embed ocular image into free image
    i = env.viewer.resolution[1] - resample_height
    j = env.viewer.resolution[0] - resample_width
    img[i:, j:] = resampled_img

  return img


if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate a policy.')
  parser.add_argument('run_folder', type=str,
                      help='the simulation folder')
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

  # Define directories
  checkpoint_dir = os.path.join(args.run_folder, 'checkpoints')
  evaluate_dir = os.path.join(args.run_folder, 'evaluate')

  # Make sure output dir exists
  os.makedirs(evaluate_dir, exist_ok=True)

  # Override run parameters
  run_params = dict()
  run_params["action_sample_freq"] = 100
  run_params["evaluate"] = True

  # Use deterministic actions?
  deterministic = False

  # Initialise environment
  env = Simulator.get(args.run_folder, run_parameters=run_params)

  print(f"run parameters are: {env.run_parameters}")

  # Load latest model if filename not given
  if args.checkpoint is not None:
    model_file = args.checkpoint
  else:
    files = natural_sort(os.listdir(checkpoint_dir))
    model_file = files[-1]

  # Load policy TODO should create a load method for uitb.rl.BaseRLModel
  print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}')
  model = PPO.load(os.path.join(checkpoint_dir, model_file))

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
      imgs.append(grab_pip_image(env))

    # Loop until episode ends
    while not done:

      # Get actions from policy
      action, _states = model.predict(obs, deterministic=deterministic)

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
        imgs.append(grab_pip_image(env))

    #print(f"Episode {episode_idx}: {env.get_episode_statistics_str()}")

    #episode_statistics = env.get_episode_statistics()
    #for key in episode_statistics:
    #  statistics[key].append(episode_statistics[key])

  #print(f'Averages over {args.num_episodes} episodes (std in parenthesis):',
  #      ', '.join(['{}: {:.2f} ({:.2f})'.format(k, np.mean(v), np.std(v)) for k, v in statistics.items()]))

  if args.logging:
    # Output log
    state_logger.save(os.path.join(evaluate_dir, args.state_log_file))
    action_logger.save(os.path.join(evaluate_dir, args.action_log_file))
    print(f'Log files have been saved files {os.path.join(evaluate_dir, args.state_log_file)}.pickle and '
          f'{os.path.join(evaluate_dir, args.action_log_file)}.pickle')

  if args.record:
    # Write the video
    env.viewer.write_video(imgs, os.path.join(evaluate_dir, args.out_file))
    print(f'A recording has been saved to file {os.path.join(evaluate_dir, args.out_file)}')