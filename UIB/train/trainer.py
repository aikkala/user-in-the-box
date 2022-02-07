import os

from UIB.utils.functions import output_path, timeout_input
from UIB.train.configs import *

import wandb
from wandb.integration.sb3 import WandbCallback


if __name__=="__main__":

  # Load a config
  config = mobl_arms_pointing_v1

  # Get name for this run from config
  name = config.get("name", None)

  if name is None:
    # Ask user to name this run
    name = timeout_input("Give a name for this run. Input empty string or wait for 30 seconds for a random name.",
                         timeout=30, default="")
    config["name"] = name

  # Initialise wandb
  run = wandb.init(project="uib", name=name, config=config, sync_tensorboard=True, save_code=True, dir=output_path())

  # Define output directories
  model_folder = os.path.join(output_path(), config["env_name"], 'trained-models')

  # Initialise model
  run_folder = os.path.join(model_folder, run.name)
  model = config["model"](config, run_folder=run_folder)

  # Save config
  np.save(os.path.join(run_folder, 'config.npy'), config)

  # Haven't figured out how to periodically save models in wandb; this is currently done inside the model class
  # TODO this doesn't seem to work; do the files need to be in wandb.run.dir?
  #wandb.save(os.path.join(model_folder, run.name, 'checkpoints', "model_*_steps.zip"),
  #           base_path=os.path.join(model_folder, run.name, 'checkpoints'))

  # Start the training
  model.learn(WandbCallback(verbose=2))
  run.finish()