import torch
import numpy as np
from platform import uname

from UIB.models.sb3.schedule import linear_schedule
from UIB.models.sb3.policies import MultiInputActorCriticPolicyTanhActions
from UIB.models.sb3.feature_extractor import VisualAndProprioceptionExtractor
from UIB.models.sb3.PPO import PPO
from UIB.models.sb3.recurrent_policies import RecurrentMultiInputActorCriticPolicyTanhActions
from UIB.models.sb3.RecurrentPPO import RecurrentPPO
from UIB.models.sb3.callbacks import LinearCurriculum

from UIB.utils import effort_terms
import UIB.envs.mobl_arms.pointing.reward_functions as pointing_rewards


mobl_arms_pointing_v0 = {
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-pointing-v0",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cpu",
  "env_kwargs": {"target_radius_limit": np.array([0.05, 0.15]),
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.Zero(),
                 "reward_function": pointing_rewards.ExpDistanceWithHitBonus()},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

reward_fn_curriculum = LinearCurriculum("reward_fn_curriculum", start_value=3, end_value=200, end_timestep=50_000_000)
mobl_arms_pointing_v1 = {
  "name": "testing-curriculum",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-pointing-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"target_radius_limit": np.array([0.05, 0.15]),
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.Composite(),
                 "reward_function": pointing_rewards.NegativeExpDistance(k=reward_fn_curriculum.value),
                 "callbacks": [reward_fn_curriculum]},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

mobl_arms_tracking_v1 = {
  "name": "test-tracking-v1",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-tracking-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"target_radius": 0.05,
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.Zero()},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}