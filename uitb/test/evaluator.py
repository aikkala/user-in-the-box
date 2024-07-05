import os
import numpy as np
from stable_baselines3 import PPO
import re
import argparse
import scipy.ndimage
from collections import defaultdict
import matplotlib.pyplot as pp
import cv2

from uitb.utils.logger import StateLogger, ActionLogger
from uitb.simulator import Simulator

#from uitb.bm_models.effort_models import CumulativeFatigue3CCr, ConsumedEndurance


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
    parser.add_argument('--uncloned', dest="cloned", action='store_false', help='use source code instead of files from cloned simulator module')
    parser.add_argument('--app_condition', type=str, default=None,
                        help="can be used to override the 'condition' argument passed to a Unity app")
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

    run_params["unity_record_gameplay"] = args.record  #False
    run_params["unity_logging"] = True
    run_params["unity_output_folder"] = evaluate_dir
    if args.app_condition is not None:
        run_params["app_args"] = ['-condition', args.app_condition]
    # run_params["unity_random_seed"] = 123

    # Embed visual observations into main mp4 or store as separate mp4 files
    render_mode_perception = "separate" if run_params["unity_record_gameplay"] else "embed"

    # Use deterministic actions?
    deterministic = False

    # Initialise simulator
    simulator = Simulator.get(args.simulator_folder, render_mode="rgb_array_list", render_mode_perception=render_mode_perception, run_parameters=run_params, use_cloned=args.cloned)

    # ## Change effort model #TODO: delete
    # simulator.bm_model._effort_model = CumulativeFatigue3CCr(simulator.bm_model, dt=simulator._run_parameters["dt"])

    print(f"run parameters are: {simulator.run_parameters}")

    # Load latest model if filename not given
    _policy_loaded = False
    if args.checkpoint is not None:
        model_file = args.checkpoint
        _policy_loaded = True
    else:
        try:
            files = natural_sort(os.listdir(checkpoint_dir))
            model_file = files[-1]
            _policy_loaded = True
        except (FileNotFoundError, IndexError):
            print("No checkpoint found. Will continue evaluation with randomly sampled controls.")

    if _policy_loaded:
        # Load policy TODO should create a load method for uitb.rl.BaseRLModel
        print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}\n')
        model = PPO.load(os.path.join(checkpoint_dir, model_file))

        # Set callbacks to match the value used for this training point (if the simulator had any)
        simulator.update_callbacks(model.num_timesteps)

    if args.logging:
        # Initialise log
        state_logger = StateLogger(args.num_episodes, keys=simulator.get_state().keys())

        # Actions are logged separately to make things easier
        action_logger = ActionLogger(args.num_episodes)

    # Visualise evaluations
    # statistics = defaultdict(list)
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

        # Loop until episode ends
        while not terminated and not truncated:
            # #print(f"Episode {episode_idx}: {simulator.get_episode_statistics_str()}")
            # print(reward)

            if _policy_loaded:
                # Get actions from policy
                action, _internal_policy_state = model.predict(obs, deterministic=deterministic)
            else:
                # choose random action from action space
                action = simulator.action_space.sample()

            # Take a step
            obs, r, terminated, truncated, info = simulator.step(action)
            reward += r

            if args.logging:
                action_logger.log(episode_idx,
                                  {"steps": state["steps"], "timestep": state["timestep"], "action": action.copy(),
                                   "reward": r})
                state = simulator.get_state()
                state.update(info)
                state_logger.log(episode_idx, state)

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
        simulator._GUI_camera.write_video_set_path(os.path.join(evaluate_dir, args.out_file))

        # Write the video
        # simulator._camera.write_video(imgs, os.path.join(evaluate_dir, args.out_file))
        _imgs = simulator.render()
        for _img in _imgs:
            simulator._GUI_camera.write_video_add_frame(_img)

        simulator._GUI_camera.write_video_close()
        print(f'A recording has been saved to file {os.path.join(evaluate_dir, args.out_file)}')

        # Write additional videos for each perception module camera (only if simulator._render_mode_perception == "separate")
        if simulator._render_mode_perception == "separate":
            _perception_imgs = simulator.get_render_stack_perception()
            for _module_name, _imgs in _perception_imgs.items():
                _out_file = os.path.splitext(args.out_file)[0] + f"_{_module_name.replace('/', '-')}" + os.path.splitext(args.out_file)[1]

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(os.path.join(evaluate_dir, _out_file), fourcc, simulator._GUI_camera._fps, (_imgs[0].shape[1], _imgs[0].shape[0]))
                # out.open(_out_file)
                for img in _imgs:
                    out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                out.release()
                print(f'A recording has been saved to file {os.path.join(evaluate_dir, _out_file)}')

    simulator.close()
