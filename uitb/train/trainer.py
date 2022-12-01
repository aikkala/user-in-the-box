import sys
import os, logging

import wandb
from wandb.integration.sb3 import WandbCallback

from uitb.simulator import Simulator
from uitb.utils.functions import output_path, timeout_input

from stable_baselines3.common.save_util import load_from_zip_file

import argparse

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Train an agent.')
  parser.add_argument('config_file_path', type=str,
                      help='path to the config file')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='filename of a specific checkpoint to resume training at '
                           '(default: None, start training from the scratch)')
  parser.add_argument('--resume', action='store_true', help='resume at latest checkpoint')
  parser.add_argument('--eval', action='store_true', help='run and store evaluations at a specific frequency defined by rl_model')
  args = parser.parse_args()

  # Get config file path
  config_file_path = args.config_file_path

  # Build the simulator
  simulator_folder = Simulator.build(config_file_path)

  # Initialise
  simulator = Simulator.get(simulator_folder)

  # Get the config
  config = simulator.config

  # Get simulator name
  name = config.get("simulator_name", None)

  # Restore wandb run_id from checkpoint if available
  checkpoint = args.checkpoint
  resume_training = args.resume or (checkpoint is not None)
  if resume_training:
    if checkpoint is not None:
      checkpoint_path = os.path.join(simulator._simulator_folder, 'checkpoints', checkpoint)
    else:
      checkpoint_dir = os.path.join(simulator._simulator_folder, 'checkpoints')
      # # find largest checkpoint
      # checkpoint_path = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))], key=lambda f: int(f.split("_steps")[0].split("_")[-1]))[-1]
      # find latest checkpoint
      checkpoint_path = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))], key=os.path.getctime)[-1]
    try:
      _data, _, _ = load_from_zip_file(checkpoint_path)
      wandb_id = _data["policy_kwargs"]["wandb_id"]
      print(f"Resume wandb run {wandb_id} starting at checkpoint {checkpoint_path}.")
    except Exception:
      logging.warning("Cannot reliably identify wandb run id. Will resume training, but with new wandb instance and with step counter reset to zero.")
      wandb_id = None
  else:
    checkpoint_path = None
    wandb_id = None

  if name is None:
    # Ask user to name this run
    name = timeout_input("Give a name for this run. Input empty string or wait for 30 seconds for a random name.",
                         timeout=30, default="")
    config["simulator_name"] = name.replace("-", "_")

  # Initialise wandb
  if wandb_id is None:
    wandb_id = wandb.util.generate_id()
  run = wandb.init(id=wandb_id, resume="allow", project="uitb", name=name, config=config, sync_tensorboard=True, save_code=True, dir=output_path())

  # Initialise RL model
  rl_cls = simulator.get_class("rl", config["rl"]["algorithm"])
  rl_model = rl_cls(simulator, checkpoint_path=checkpoint_path, wandb_id=wandb_id)

  # Haven't figured out how to periodically save rl in wandb; this is currently done inside the rl_model class
  # TODO this doesn't seem to work; do the files need to be in wandb.run.dir?
  #wandb.save(os.path.join(model_folder, run.name, 'checkpoints', "model_*_steps.zip"),
  #           base_path=os.path.join(model_folder, run.name, 'checkpoints'))

  # Start the training
  rl_model.learn(WandbCallback(verbose=2), eval_callback=args.eval)
  run.finish()