import gym
import os
import sys
import numpy as np
from stable_baselines3 import PPO


if __name__=="__main__":

  render = False

  # Get env name and checkpoint dir
  env_name = sys.argv[1]
  checkpoint_dir = sys.argv[2]

  # Load latest model if filename not given
  if len(sys.argv) == 4:
    model_file = sys.argv[3]
  else:
    files = np.sort(os.listdir(checkpoint_dir))
    model_file = files[0]

  # Load previous policy
  print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}')
  model = PPO.load(os.path.join(checkpoint_dir, model_file))

  # Initialise environment
  env = gym.make(env_name)

  # Visualise evaluations
  episode_lengths = []
  rewards = []
  num_episodes = 200
  for i in range(num_episodes):
    obs = env.reset()
    if render:
      env.render()
    done = False
    episode_length = 0
    reward = 0
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, r, done, info = env.step(action)
      env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl[2:] * 0.7
      if render:
        env.render()
      episode_length += 1
      reward += r
    print(f"Episode {i}: length {episode_length}, reward {reward}. ")
    episode_lengths.append(episode_length)
    rewards.append(reward)

  print(f'Average episode length and reward over {num_episodes} episodes: '
        f'length {np.mean(episode_lengths)}, reward {np.mean(rewards)}')