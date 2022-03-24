import os
import pickle
import re

from UIB.utils.functions import output_path, timeout_input
from UIB.train.configs import *

import wandb
from wandb.integration.sb3 import WandbCallback

def natural_sort(l):
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
  return sorted(l, key=alphanum_key)

if __name__=="__main__":

  # Load a config
  config = mobl_arms_tracking_v1

  # Get name for this run from config
  name = config.get("name", None)

  # Resume training at latest model
  resume = False

  if name is None:
    # Ask user to name this run
    name = timeout_input("Give a name for this run. Input empty string or wait for 30 seconds for a random name.",
                         timeout=30, default="")
    config["name"] = name

  # Initialise wandb
  run = wandb.init(project="uib", name=name, config=config, sync_tensorboard=True, save_code=True, resume=resume, dir=output_path())

  # Define output directories
  run_folder = os.path.join(output_path(), config["env_name"], run.name)
  os.makedirs(run_folder, exist_ok=True)

  # Load latest model
  if resume:
    checkpoint_dir = os.path.join(run_folder, 'checkpoints')
    files = natural_sort(os.listdir(checkpoint_dir))
    model_file = files[-1]
    config["resume"] = os.path.join(checkpoint_dir, model_file)
    # # Load policy
    # input(f'Loading model: {os.path.join(checkpoint_dir, model_file)}')
    # model.model = PPO_sb3.load()

  # Initialise model
  model = config["model"](config, run_folder=run_folder)

  # Save config (except those that can't be pickled)
  config["lr"] = None
  with open(os.path.join(run_folder, 'config.pickle'), 'wb') as file:
    pickle.dump(config, file)

  # Haven't figured out how to periodically save models in wandb; this is currently done inside the model class
  # TODO this doesn't seem to work; do the files need to be in wandb.run.dir?
  #wandb.save(os.path.join(model_folder, run.name, 'checkpoints', "model_*_steps.zip"),
  #           base_path=os.path.join(model_folder, run.name, 'checkpoints'))

  # Start the training
  model.learn(WandbCallback(verbose=2))
  run.finish()