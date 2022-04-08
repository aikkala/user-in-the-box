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

from scipy.spatial.transform import Rotation

from UIB.utils.logger import StateLogger, ActionLogger
from UIB.utils.functions import output_path


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(env, ocular_img_type="inset", dynamic_camera=False):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  # Define camera to use for agent perception
  eye_camera_name = 'oculomotor'

  # Define camera to use for visualization
  main_camera_name = ('dynamicview' if dynamic_camera else 'backview') if 'driving' in env.spec.id else 'for_testing'  #for_testing #front #rightview #oculomotor #backview

  # Define image size
  width, height = env.metadata["imagesize"]

  # Visualise target plane
  if hasattr(env, "target_plane_geom_idx"):
    env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0.1 #0 #.25 #0.1

  # Dynamic camera mode
  if dynamic_camera:
    if not hasattr(env, "camera_angle"):
      env.camera_angle = -np.pi/2
      env.camera_angle_step = 0
      env.camera_pos = np.array([0, 1.75, 0])
      env.camera_angle_vertical = -np.pi/2
    env.camera_angle -= (0.02) - 0.02*(min((1, env.camera_angle_step/400)))  #last term: dynamic speed change
    env.camera_angle_vertical += 0.005 if 200 <= env.camera_angle_step < 300 else 0  #dynamically adjust vertical camera angle
    env.camera_pos[1] -= 0.004 if 100 <= env.camera_angle_step < 300 else 0  #dynamically adjust camera position relative to torso
    env.camera_pos[2] -= 0.001 if 200 <= env.camera_angle_step < 300 else 0  #dynamically adjust camera position relative to torso
    env.camera_angle_step += 1
    env.sim.model.body_quat[env.sim.model._body_name2id[main_camera_name]][:] = Rotation.from_euler("xyz", (1.57, env.camera_angle_vertical, env.camera_angle)).as_quat()[[3, 0, 1, 2]]
    env.sim.model.cam_pos[env.sim.model._camera_name2id[main_camera_name]][:] = env.camera_pos

  # Grab images
  img = np.flipud(env.sim.render(height=height, width=width, camera_name=main_camera_name))
  ocular_img = np.flipud(env.sim.render(height=env.ocular_image_height, width=env.ocular_image_width, camera_name=eye_camera_name))

  # Disable target plane
  if hasattr(env, "target_plane_geom_idx"):
    env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0.0

  if ocular_img_type == "inset":
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
  elif ocular_img_type == "full":
    # Show only ocular image
    img = ocular_img
    env.metadata["imagesize"] = (env.ocular_image_width, env.ocular_image_height)

  return img

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
  env_kwargs["action_sample_freq"] = 20
  env_kwargs["freq_curriculum"] = lambda : 1  #increase target speed by replacing 0.5 by 1
  env_kwargs["max_frequency"] = 0.5
  if "iso" in env_name:  #TODO: create new param to select between random and ISO targets
    env_kwargs["evaluate"] = True

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
  targets_hit = []
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
      action, _states = model.predict(obs, deterministic=True)

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
        if not ("driving" in env_name or "button" in env_name):  #TODO: add env attribute for visual input type
          env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl * 0.7
        imgs.append(grab_pip_image(env))

    #print(f"Episode {episode_idx}: targets hit {env.targets_hit}, length {env.steps*env.dt} seconds ({env.steps} steps), reward {reward}. ")
    print(f"Episode {episode_idx}: length {env.steps * env.dt} seconds ({env.steps} steps), reward {reward}. ")
    episode_lengths.append(env.steps)
    rewards.append(reward)
    #targets_hit.append(env.targets_hit)

  print(f'Averages over {args.num_episodes} episodes: '#targets hit {np.mean(targets_hit)}, '
        f'length {np.mean(episode_lengths)*env.dt} seconds ({np.mean(episode_lengths)} steps), reward {np.mean(rewards)}')

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