import torch
import numpy as np
from platform import uname

from uitb.rl.sb3.schedule import linear_schedule
from uitb.rl.sb3.policies import MultiInputActorCriticPolicyTanhActions
from uitb.rl.sb3.PPO import PPO
from uitb.rl.sb3.callbacks import LinearCurriculum

from uitb.utils import effort_terms

from uitb.bm_models import MoblArmsIndex
from uitb.tasks import RemoteDriving, Pointing
from uitb.perception.proprioception import BasicWithEndEffectorPosition
from uitb.perception.vision import FixedEye
from uitb.rl.sb3.feature_extractor import FeatureExtractor

pointing = \
  {"name": "mobl-arms-index-pointing-with-shoulder-variant-before-step",
   "rl": {
     "algorithm": PPO,
     "policy_type": MultiInputActorCriticPolicyTanhActions,
     "policy_kwargs": {
       "activation_fn": torch.nn.LeakyReLU,
       "net_arch": [256, 256, 256],
       "log_std_init": 0.0,
       "features_extractor_class": FeatureExtractor,
       "normalize_images": False
     },
     "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
     "total_timesteps": 100_000_000, "device": "cuda", "num_workers": 10,
     "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
   },
   "simulation": {
     "bm_model": MoblArmsIndex,
     "perception_modules": {
       FixedEye: dict(resolution=[120, 80], channels=[3], pos="0 0 1.2", quat="0.583833 0.399104 -0.399421 -0.583368"),
       BasicWithEndEffectorPosition: dict(end_effector="hand_2distph")},
     "task": Pointing,
     "task_kwargs": dict(end_effector="hand_2distph", shoulder="humphant"),
     "run_parameters": {"action_sample_freq": 20, "use_cloned_files": False}
   },
   }

tracking = \
  {"name": "mobl-arms-index-tracking",
   "run_parameters": {"action_sample_freq": 20},
   "rl": {
     "algorithm": PPO,
     "policy_type": MultiInputActorCriticPolicyTanhActions,
     "policy_kwargs": {
       "activation_fn": torch.nn.LeakyReLU,
       "net_arch": [256, 256, 256],
       "log_std_init": 0.0,
       "features_extractor_class": FeatureExtractor,
       "normalize_images": False
     },
     "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
     "total_timesteps": 100_000_000, "device": "cuda", "num_workers": 10,
     "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
   },
   "simulation": {
     "bm_model": MoblArmsIndex,
     "perception_modules": {
       BasicWithEndEffectorPosition: dict(end_effector="hand_2distph"),
       FixedEye: dict(resolution=[120, 80], channels=[3], buffer=0.1, pos="0 0 1.2",
                      quat="0.583833 0.399104 -0.399421 -0.583368")},
     "task": Pointing,
     "task_kwargs": dict(end_effector="hand_2distph", shoulder="humphant")}
   }

remote_driving = \
  {"name": "mobl-arms-index-remote-driving",
   "run_parameters": {"action_sample_freq": 20},
   "rl": {
     "algorithm": PPO,
     "policy_type": MultiInputActorCriticPolicyTanhActions,
     "policy_kwargs": {
       "activation_fn": torch.nn.LeakyReLU,
       "net_arch": [256, 256, 256],
       "log_std_init": 0.0,
       "features_extractor_class": FeatureExtractor,
       "normalize_images": False
     },
     "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
     "total_timesteps": 100_000_000, "device": "cuda", "num_workers": 10,
     "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
   },
   "simulation": {
     "bm_model": MoblArmsIndex,
     "perception_modules": {
       "BasicWithEndEffectorPosition": (BasicWithEndEffectorPosition, dict(end_effector="hand_2distph")),
       "FixedEye": (FixedEye, dict(resolution=[120, 80], pos="0 0 1.2", quat="0.583833 0.399104 -0.399421 -0.583368"))},
     "task": RemoteDriving,
     "task_kwargs": dict(end_effector="hand_2distph", episode_length_seconds=10)}
   }
