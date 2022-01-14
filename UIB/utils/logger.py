import pickle


class EvaluationLogger:


  def __init__(self, num_episodes):
    self.num_episodes = num_episodes
    self.keys = ["step", "timestep", "qpos", "qvel", "qacc", "act", "fingertip_xpos", "fingertip_xmat",
                 "fingertip_xvelp", "fingertip_xvelr", "termination", "target_hit", "target_position", "target_radius"]

#    self.data = {self.strify(num):
#                   {"step": [], "timestep": [], "qpos": [], "qvel": [], "qacc": [], "act": [],
#                    "fingertip_xpos": [], "fingertip_xmat": [],
#                    "fingertip_xvelp": [], "fingertip_xvelr": [],
#                    "termination": [], "target_hit": []} for num in range(num_episodes)}

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
