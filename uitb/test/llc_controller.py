import os
import numpy as np
np.set_printoptions(precision=2, suppress=True)
from stable_baselines3 import PPO
import re
import argparse
import scipy.ndimage
from collections import defaultdict
import matplotlib.pyplot as pp
import matplotlib
matplotlib.use('TkAgg')
import cv2
from time import sleep
import queue
from pynput.keyboard import Listener, Key

from uitb.utils.logger import StateLogger, ActionLogger
from uitb.simulator import Simulator


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(simulator):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  # Visualise muscle actuation
  simulator._model.tendon_rgba[:, 0] = 0.3 + simulator._data.ctrl * 0.7

  # Grab images
  img, _ = simulator._camera.render()

  ocular_img = None
  for module in simulator.perception.perception_modules:
    if module.modality == "vision":
      # TODO would be better to have a class function that returns "human-viewable" rendering of the observation;
      #  e.g. in case the vision model has two cameras, or returns a combination of rgb + depth images etc.
      ocular_img, _ = module._camera.render()

  # Set back to default
  simulator._model.tendon_rgba[:, 0] = 0.95

  if ocular_img is not None:

    # Resample
    resample_factor = 2
    resample_height = ocular_img.shape[0]*resample_factor
    resample_width = ocular_img.shape[1]*resample_factor
    resampled_img = np.zeros((resample_height, resample_width, 3), dtype=np.uint8)
    for channel in range(3):
      resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)

    # Embed ocular image into free image
    i = simulator._camera.height - resample_height
    j = simulator._camera.width - resample_width
    img[i:, j:] = resampled_img

  return img

# def read_keyboard(queue):
#   # Wait for input from user
#   input =

def on_press(key):
  try:

    # Exit with esc
    if key == Key.esc:
      return False

    # Set to initial position with space
    elif key == Key.space:
      q.put(("set", initial_position.copy()))

    # First joint
    elif key.char == "1":
      q.put(("add", np.array([0.5, 0, 0, 0, 0])))
    elif key.char == "q":
      q.put(("add", np.array([0.1, 0, 0, 0, 0])))
    elif key.char == "a":
      q.put(("add", np.array([-0.1, 0, 0, 0, 0])))
    elif key.char == "z":
      q.put(("add", np.array([-0.5, 0, 0, 0, 0])))

    # Second joint
    elif key.char == "2":
      q.put(("add", np.array([0, 0.5, 0, 0, 0])))
    elif key.char == "w":
      q.put(("add", np.array([0, 0.1, 0, 0, 0])))
    elif key.char == "s":
      q.put(("add", np.array([0, -0.1, 0, 0, 0])))
    elif key.char == "x":
      q.put(("add", np.array([0, -0.5, 0, 0, 0])))

    # Third joint
    elif key.char == "3":
      q.put(("add", np.array([0, 0, 0.5, 0, 0])))
    elif key.char == "e":
      q.put(("add", np.array([0, 0, 0.1, 0, 0])))
    elif key.char == "d":
      q.put(("add", np.array([0, 0, -0.1, 0, 0])))
    elif key.char == "c":
      q.put(("add", np.array([0, 0, -0.5, 0, 0])))

    # Fourth joint
    elif key.char == "4":
      q.put(("add", np.array([0, 0, 0, 0.5, 0])))
    elif key.char == "r":
      q.put(("add", np.array([0, 0, 0, 0.1, 0])))
    elif key.char == "f":
      q.put(("add", np.array([0, 0, 0, -0.1, 0])))
    elif key.char == "v":
      q.put(("add", np.array([0, 0, 0, -0.5, 0])))

    # Fifth joint
    elif key.char == "5":
      q.put(("add", np.array([0, 0, 0, 0, 0.5])))
    elif key.char == "t":
      q.put(("add", np.array([0, 0, 0, 0, 0.1])))
    elif key.char == "g":
      q.put(("add", np.array([0, 0, 0, 0, -0.1])))
    elif key.char == "b":
      q.put(("add", np.array([0, 0, 0, 0, -0.5])))

    else:
      return

  except AttributeError:
    pass

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Evaluate a policy.')
  parser.add_argument('simulator_folder', type=str,
                      help='the simulation folder')
  parser.add_argument('--action_sample_freq', type=float, default=100,
                      help='action sample frequency (how many times per second actions are sampled from policy, default: 20)')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='filename of a specific checkpoint (default: None, latest checkpoint is used)')
  parser.add_argument('--logging', action='store_true', help='enable logging')
  parser.add_argument('--state_log_file', default='controller_state_log',
                      help='output file for state log if logging is enabled (default: ./controller_state_log)')
  parser.add_argument('--action_log_file', default='controller_action_log',
                      help='output file for action log if logging is enabled (default: ./controller_action_log)')
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
  deterministic = True

  # Initialise simulator
  simulator = Simulator.get(args.simulator_folder, run_parameters=run_params)

  print(f"run parameters are: {simulator.run_parameters}\n")

  # Load latest model if filename not given
  if args.checkpoint is not None:
    model_file = args.checkpoint
  else:
    files = natural_sort(os.listdir(checkpoint_dir))
    model_file = files[-1]

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
  statistics = defaultdict(list)
  imgs = []

  # Reset environment
  simulator.reset()
  simulator.task._target_qpos[:] = simulator.task._qpos.copy()
  initial_position = simulator.task._qpos.copy()
  obs = simulator.get_observation()


  if args.logging:
    state = simulator.get_state()
    state_logger.log(0, state)

  # Initialise a queue so we can share data between threads
  q = queue.Queue()

  # Start a keyboard press reader in a thread
  # keyboard_reader = threading.Thread(target=read_keyboard, args=(q,), daemon=True)
  # keyboard_reader.start()
  # listener = Listener(on_press=lambda event: on_press(event, q))
  listener = Listener(on_press=on_press)
  listener.start()

  # Run simulation here in main thread
  while True:

    # If listener has exited (user pressed 'esc'), break
    if not listener.is_alive():
      break

    # Read queue
    if not q.empty():
      action = q.get()
      if action[0] == "add":
        simulator.task._target_qpos = np.clip(simulator.task._target_qpos + action[1], -1, 1)
      elif action[0] == "set":
        simulator.task._target_qpos = action[1]
      obs = simulator.get_observation()
      print(simulator.task._target_qpos)

    # Get actions from policy
    action, _states = model.predict(obs, deterministic=deterministic)

    # Take a step
    obs, r, done, info = simulator.step(action)

    if args.logging:
      action_logger.log(0, {"steps": state["steps"], "timestep": state["timestep"], "action": action.copy(),
                            "reward": r})
      state = simulator.get_state()
      state.update(info)
      state_logger.log(0, state)

    # Do some rendering
    img = grab_pip_image(simulator)
    qpos = simulator.task._target_qpos
    cv2.putText(img, f'{qpos[0]:.2f} elevation angle', (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f'{qpos[1]:.2f} shoulder elevation', (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f'{qpos[2]:.2f} shoulder rotation', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f'{qpos[3]:.2f} elbow flexion', (700, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f'{qpos[4]:.2f} pronation/supination', (700, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("simulation", np.flip(img, axis=2)/255)
    cv2.waitKey(1)
    # sleep(0.01)

  # Run keyboard press reader in another thread


  if args.logging:
    # Output log
    state_logger.save(os.path.join(evaluate_dir, args.state_log_file))
    action_logger.save(os.path.join(evaluate_dir, args.action_log_file))
    print(f'Log files have been saved files {os.path.join(evaluate_dir, args.state_log_file)}.pickle and '
          f'{os.path.join(evaluate_dir, args.action_log_file)}.pickle')
