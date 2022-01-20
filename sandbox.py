import gym
import os
import torch
import numpy as np
from platform import uname

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from utils import opensim_file
import mujoco_py

from UIB.sb3_additions.schedule import linear_schedule

def generate_random_trajectories(env, num_trajectories=1_000, trajectory_length_seconds=10, render_mode="human"):

  def add_noise(actions, rate, scale):
    return np.random.normal(loc=rate*actions, scale=scale)

  # Vary noise scale and noise rate
  std_limits = [2, 4]
  timescale_limits = [0.5, 4]

  # Ignore first states cause they all start from the default pose
  ignore_first_seconds = 5
  assert trajectory_length_seconds > ignore_first_seconds, \
    f"Trajectory must be longer than {ignore_first_seconds} seconds"
  ignore_first = int(ignore_first_seconds/env.dt)

  # Trajectory length in steps
  trajectory_length = int(trajectory_length_seconds/env.dt)

  # Collect states
  states = np.zeros((num_trajectories, trajectory_length, len(env.independent_joints)))

  for trajectory_idx in range(num_trajectories):

    env.reset()
    env.render(mode=render_mode)

    # Sample noise statistics, different for each episode
    rate = np.exp(-env.dt / np.random.uniform(*timescale_limits))
    scale = np.random.uniform(*std_limits) * np.sqrt(1-rate*rate)

    # Start with zero actions
    actions = np.zeros((env.action_space.shape[0]))

  #   env.sim.data.qpos[:] = np.array([ 0.,        0.,          0.,         -0.06199092,  0.02625643, -0.02625643,
  # 0.06199092, -0.0125519,   0.10144003,  0.0455966,  -0.0455966,  -0.10144003,
  # 0.0125519,  -0.40572562,  0.25616023,  0.40572562, -0.11530936,  1.70942086,
  # 0.37816576])
  #   input(env.sim.data.qpos)


    for step_idx in range(trajectory_length):

      # Add noise to actions
      actions = add_noise(actions, rate, scale)

      # Step
      # actions = np.array([0.,         0.,         0.60831097, 0.51026837, 0.02,       0.19714249,
      #  0.16943364, 0.06905723, 0.02,       0.04002639, 0.02,       0.02,
      #  0.02,       0.02,       0.02,       0.02,       0.02,       0.02,
      #  0.02,       0.02,       0.02,       0.02,       0.06319038, 0.02,
      #  0.03585573, 0.02504265, 0.10648935, 0.04966655])
      #np.array([0.,         0.,         0.02,       0.06319038, 0.02,       0.03585573, 0.02504265, 0.10648935, 0.04966655])

      env.step(actions)
      # input(env.sim.data.qpos[:])
      env.render(mode=render_mode)

      states[trajectory_idx, step_idx] = env.sim.data.qpos[env.independent_joints].copy()

  return states[:, ignore_first:]

if __name__=="__main__":

  env_name = 'UIB:mobl-arms-muscles-v0'
  train = True
  render_mode = "human"  #"human", "rgb-array"
  start_method = 'spawn' if 'Microsoft' in uname().release else 'forkserver'
  generate_experience = True
  experience_file = 'experience.npy'
  num_cpu = 7
  output_dir = os.path.join('output', env_name)
  checkpoint_dir = os.path.join(output_dir, 'checkpoint')
  log_dir = os.path.join(output_dir, 'log')

  # Leave for future kwargs
  env_kwargs = {"xml_file": "models/mobl_arms_muscles_original.xml"}

  # Initialise environment
  env = gym.make(env_name, **env_kwargs)
  env.step(env.action_space.sample())

  if generate_experience:
   experience = generate_random_trajectories(env, num_trajectories=1000, trajectory_length_seconds=10, render_mode=render_mode)
   np.save(experience_file, experience)
  else:
   try:
    experience = np.load(experience_file)
   except FileNotFoundError:
     pass

  # Do the training
  if train:

    # Initialise parallel envs
    parallel_envs = make_vec_env(env_name, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs,
                                 vec_env_kwargs={'start_method': start_method})

    # Policy parameters
    policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
                         net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                         log_std_init=0.0)
    lr = 3e-4

    # Initialise policy
    model = PPO('MlpPolicy', parallel_envs, verbose=1, policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir, learning_rate=lr)

    # Initialise a callback for checkpoints
    save_freq = 5000000 // num_cpu
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_dir, name_prefix='model')

    # Do the learning first with constant learning rate
    model.learn(total_timesteps=20_000_000, callback=[checkpoint_callback])

    # Then some more learning with a decaying learning rate
    model.learn(total_timesteps=10_000_000, callback=[checkpoint_callback], learning_rate=linear_schedule(lr),
                reset_num_timesteps=False)

  else:

    # Load previous policy
    model = PPO.load(os.path.join(checkpoint_dir, 'model_2999997_steps'))


  # Initialise environment
  env = gym.make(env_name, **env_kwargs)

  # Visualise evaluations, perhaps save a video as well
  while not train:

    obs = env.reset()
    env.render(mode=render_mode)
    done = False
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, rewards, done, info = env.step(action)
      env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl[2:] * 0.7
      env.render(mode=render_mode)
