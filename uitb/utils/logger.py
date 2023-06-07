import pickle


class BaseLogger:

  def __init__(self, num_episodes, keys):
    self.num_episodes = num_episodes
    self.keys = keys
    self.data = {self.strify(num): {key: [] for key in self.keys} for num in range(num_episodes)}

  def strify(self, num):
    return f"episode_{str(num).zfill(len(str(self.num_episodes)))}"

  def log(self, episode, d):
    # We use lists so we should add None to those keys that aren't included in d but are present in self.data
    for key in self.keys:
      if key in d:
        value = d[key]
      else:
        value = None
      self.data[self.strify(episode)][key].append(value)

  def save(self, filename):
    with open(f"{filename}.pickle", 'wb') as handle:
      pickle.dump(self.data, handle)

class StateLogger(BaseLogger):
  def __init__(self, num_episodes, keys):
    super().__init__(num_episodes=num_episodes, keys=keys)

class ActionLogger(BaseLogger):
  def __init__(self, num_episodes):
    super().__init__(num_episodes=num_episodes, keys=["step", "timestep", "action", "ctrl", "reward"])