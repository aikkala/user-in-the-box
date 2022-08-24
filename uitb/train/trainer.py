import sys

import wandb
from wandb.integration.sb3 import WandbCallback

from uitb.simulator import Simulator
from uitb.utils.functions import output_path, timeout_input


if __name__=="__main__":

  # Get config file path
  assert len(sys.argv) == 2, "You should input only one argument, the path to a config file"
  config_file_path = sys.argv[1]

  # Build the simulator
  simulator_folder = Simulator.build(config_file_path)

  # Initialise
  simulator = Simulator.get(simulator_folder)

  # Get the config
  config = simulator.config

  # Get simulator name
  name = config.get("simulator_name", None)

  if name is None:
    # Ask user to name this run
    name = timeout_input("Give a name for this run. Input empty string or wait for 30 seconds for a random name.",
                         timeout=30, default="")
    config["simulator_name"] = name.replace("-", "_")

  # Initialise wandb
  run = wandb.init(project="uitb", name=name, config=config, sync_tensorboard=True, save_code=True, dir=output_path())

  # Initialise RL model
  rl_cls = simulator.get_class("rl", config["rl"]["algorithm"])
  rl_model = rl_cls(simulator)

  # Haven't figured out how to periodically save rl in wandb; this is currently done inside the rl_model class
  # TODO this doesn't seem to work; do the files need to be in wandb.run.dir?
  #wandb.save(os.path.join(model_folder, run.name, 'checkpoints', "model_*_steps.zip"),
  #           base_path=os.path.join(model_folder, run.name, 'checkpoints'))

  # Start the training
  rl_model.learn(WandbCallback(verbose=2))
  run.finish()