import os

from uitb.simulator import Simulator
from uitb.utils.functions import output_path, timeout_input
from uitb.train.configs import *

import wandb
from wandb.integration.sb3 import WandbCallback

if __name__=="__main__":

  # Load a config
  config = pointing

  # Build the simulator
  Simulator.build(config)

  # Initialise
  simulator = Simulator(config["run_folder"])

  # Get name for this run from config
  name = config.get("name", None)

  if name is None:
    # Ask user to name this run
    name = timeout_input("Give a name for this run. Input empty string or wait for 30 seconds for a random name.",
                         timeout=30, default="")
    config["name"] = name

  # Initialise wandb
  run = wandb.init(project="uitb", name=name, sync_tensorboard=True, save_code=True, dir=output_path())

  # Initialise RL model
  rl_model = config["rl"]["algorithm"](simulator, config["rl"], config["run_folder"])

  # Haven't figured out how to periodically save rl in wandb; this is currently done inside the rl_model class
  # TODO this doesn't seem to work; do the files need to be in wandb.run.dir?
  #wandb.save(os.path.join(model_folder, run.name, 'checkpoints', "model_*_steps.zip"),
  #           base_path=os.path.join(model_folder, run.name, 'checkpoints'))

  # Start the training
  rl_model.learn(WandbCallback(verbose=2))
  run.finish()