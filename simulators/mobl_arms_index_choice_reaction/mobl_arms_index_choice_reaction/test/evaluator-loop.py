import os
import numpy as np
from stable_baselines3 import PPO
import re
import argparse
import scipy.ndimage
from collections import defaultdict
import matplotlib.pyplot as pp
import mujoco


from uitb.utils.logger import StateLogger, ActionLogger
from uitb.simulator import Simulator


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


### DEPRECATED, use simulator.render() instead
# def grab_pip_image(simulator):
#   # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'
#
#   # Grab images
#   img, _ = simulator._GUI_camera.render()
#
#   ocular_img = None
#   for module in simulator.perception.perception_modules:
#     if module.modality == "vision":
#       # TODO would be better to have a class function that returns "human-viewable" rendering of the observation;
#       #  e.g. in case the vision model has two cameras, or returns a combination of rgb + depth images etc.
#       ocular_img, _ = module._camera.render()
#
#   if ocular_img is not None:
#
#     # Resample
#     resample_factor = 2
#     resample_height = ocular_img.shape[0]*resample_factor
#     resample_width = ocular_img.shape[1]*resample_factor
#     resampled_img = np.zeros((resample_height, resample_width, 3), dtype=np.uint8)
#     for channel in range(3):
#       resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)
#
#     # Embed ocular image into free image
#     i = simulator._GUI_camera.height - resample_height
#     j = simulator._GUI_camera.width - resample_width
#     img[i:, j:] = resampled_img
#
#   return img


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate a policy.')
    parser.add_argument('simulator_folder', type=str,
                        help='the simulation folder')
    parser.add_argument('--action_sample_freq', type=float, default=20,
                        help='action sample frequency (how many times per second actions are sampled from policy, default: 20)')
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
    checkpoint_dir = os.path.join(args.simulator_folder, 'checkpoints')
    evaluate_dir = os.path.join(args.simulator_folder, 'evaluate')

    # Make sure output dir exists
    os.makedirs(evaluate_dir, exist_ok=True)

    # Override run parameters
    run_params = dict()
    run_params["action_sample_freq"] = args.action_sample_freq
    run_params["evaluate"] = True

    # Use deterministic actions?
    deterministic = False

    # Initialise simulator
    simulator = Simulator.get(args.simulator_folder, render_mode="rgb_array", run_parameters=run_params)

    print(f"run parameters are: {simulator.run_parameters}\n")

    # Load latest model if filename not given
    if args.checkpoint is not None:
        model_file = args.checkpoint
    else:
        files = natural_sort(os.listdir(checkpoint_dir))
        model_file = files[-1]

    # Load policy TODO should create a load method for uitb.rl.BaseRLModel
    print(f'heloLoading model: {os.path.join(checkpoint_dir, model_file)}\n')
    model = PPO.load(os.path.join(checkpoint_dir, model_file))

    # Set callbacks to match the value used for this training point (if the simulator had any)
    simulator.update_callbacks(model.num_timesteps)

    if args.logging:
        # Initialise log
        state_logger = StateLogger(args.num_episodes, keys=simulator.get_state().keys())

        # Actions are logged separately to make things easier
        action_logger = ActionLogger(args.num_episodes)

    if args.record:
        simulator._GUI_camera.write_video_set_path(os.path.join(evaluate_dir, args.out_file))

    # Visualise evaluations
    statistics = defaultdict(list)
    for episode_idx in range(args.num_episodes):

        print(f"Run episode {episode_idx + 1}/{args.num_episodes}.")

        # Reset environment
        obs, info = simulator.reset()
        terminated = False
        truncated = False
        reward = 0

        if args.logging:
            state = simulator.get_state()
            state_logger.log(episode_idx, state)

        if args.record:
            simulator._GUI_camera.write_video_add_frame(simulator.render())

        # Loop until episode ends
        while not terminated and not truncated:

            # Get actions from policy
            action, _states = model.predict(obs, deterministic=deterministic)

            
            simulator.task._target_qpos = action
            simulator._steps = 0
            obs = simulator.get_observation()
            acc_reward = 0
    
            while simulator._steps <= simulator._max_steps:
        
                llc_action, _states = simulator.llc_model.predict(simulator.get_llcobservation(action), deterministic=True)
                # Set control for the bm model
                simulator.bm_model.set_ctrl(simulator._model, simulator._data, llc_action)

                # Set control for perception modules (e.g. eye movements)
                simulator.perception.set_ctrl(simulator._model, simulator._data, action[simulator.bm_model.nu:])

                # Advance the simulation
                mujoco.mj_step(simulator._model, simulator._data, nstep=int(simulator._run_parameters["frame_skip"]))

                # Update bm model (e.g. update constraints); updates also effort model
                simulator.bm_model.update(simulator._model, simulator._data)

                # Update perception modules
                simulator.perception.update(simulator._model, simulator._data)
   
                dist = np.abs(action - simulator._get_qpos(simulator._model, simulator._data))

                # Update environment        
                r, terminated, truncated, info = simulator.task.update(simulator._model, simulator._data)
        
                # Add an effort cost to reward
                r -= simulator.bm_model.get_effort_cost(simulator._model, simulator._data)
                
                #acc_reward += reward
                # Get observation
                obs = simulator.get_observation()

                # Add frame to stack
                if simulator._render_mode == "rgb_array_list":
                  simulator._render_stack.append(simulator._GUI_rendering())
                elif simulator._render_mode == "human":
                  simulator._GUI_rendering_pygame()
                
                if truncated or terminated:
                    break
        
                if info["new_button_generated"] or info["target_hit"]:
                    break
            
                simulator._steps += 1
                if np.all(dist < simulator._target_radius):
                    break
   
            
                # Take a step
                #obs, r, terminated, truncated, info = simulator.step(action)
                reward += r

                if args.logging:
                    action_logger.log(episode_idx,
                                  {"steps": state["steps"], "timestep": state["timestep"], "action": action.copy(),
                                   "reward": r})
                    state = simulator.get_state()
                    state.update(info)
                    state_logger.log(episode_idx, state)

                if args.record and not terminated and not truncated:
                    simulator._GUI_camera.write_video_add_frame(simulator.render())

        # print(f"Episode {episode_idx}: {simulator.get_episode_statistics_str()}")

        # episode_statistics = simulator.get_episode_statistics()
        # for key in episode_statistics:
        #  statistics[key].append(episode_statistics[key])

    # print(f'Averages over {args.num_episodes} episodes (std in parenthesis):',
    #      ', '.join(['{}: {:.2f} ({:.2f})'.format(k, np.mean(v), np.std(v)) for k, v in statistics.items()]))

    if args.logging:
        # Output log
        state_logger.save(os.path.join(evaluate_dir, args.state_log_file))
        action_logger.save(os.path.join(evaluate_dir, args.action_log_file))
        print(f'Log files have been saved files {os.path.join(evaluate_dir, args.state_log_file)}.pickle and '
              f'{os.path.join(evaluate_dir, args.action_log_file)}.pickle')

    if args.record:
        # Write the video
        simulator._GUI_camera.write_video_close()
        print(f'A recording has been saved to file {os.path.join(evaluate_dir, args.out_file)}')
