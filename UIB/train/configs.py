import torch
import numpy as np
from platform import uname

from UIB.models.sb3.schedule import linear_schedule
from UIB.models.sb3.policies import MultiInputActorCriticPolicyTanhActions, ActorCriticPolicyTanhActions
from UIB.models.sb3.feature_extractor import VisualAndProprioceptionExtractor
from UIB.models.sb3.PPO import PPO
from UIB.models.sb3.callbacks import LinearCurriculum

from UIB.utils import effort_terms
import UIB.envs.mobl_arms.pointing.reward_functions as pointing_rewards
import UIB.envs.mobl_arms.iso_pointing.reward_functions as iso_pointing_rewards
import UIB.envs.mobl_arms.tracking.reward_functions as tracking_rewards
import UIB.envs.mobl_arms.button_press.reward_functions as button_press_rewards
import UIB.envs.mobl_arms.remote_driving.reward_functions as driving_rewards

mobl_arms_pointing_v0 = {
  "name": "pointing-v0-old-model-new-muscles-3CC-r",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-pointing-v0",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cpu",
  "env_kwargs": {"target_radius_limit": np.array([0.05, 0.15]),
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.CumulativeFatigue(),
                 "reward_function": pointing_rewards.NegativeExpDistanceWithHitBonus(k=10)},
  "policy_type": ActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

mobl_arms_pointing_v1 = {
  "name": "pointing-v1-patch-v1",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-pointing-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"target_radius_limit": np.array([0.05, 0.15]),
                 "action_sample_freq": 20,
                 "shoulder_variant": "patch-v1",
                 "effort_term": effort_terms.Neural(),
                 "reward_function": pointing_rewards.NegativeExpDistanceWithHitBonus(k=10),
                 "callbacks": []},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

target_speed_curriculum = LinearCurriculum("target_speed_curriculum", start_value=0, end_value=1,
                                           end_timestep=60_000_000, start_timestep=40_000_000)
mobl_arms_tracking_v1 = {
  "name": "tracking-v1-patch-v1",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-tracking-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"target_radius": 0.05,
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.Neural(),
                 "reward_function": tracking_rewards.NegativeDistance(),
                 "freq_curriculum": target_speed_curriculum.value,
                 "episode_length_seconds": 10,
                 "callbacks": [target_speed_curriculum]},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}


mobl_arms_button_press_v1 = {
  "name": "test-button-press",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-button-press-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"action_sample_freq": 20,
                 "effort_term": effort_terms.Neural(),
                 "reward_function": button_press_rewards.NegativeExpDistanceWithHitBonus(),
                 "callbacks": []},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

mobl_arms_iso_pointing_v1 = {
  "name": "iso-pointing-U1-patch-v1-dwell",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-iso-pointing-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"user": "U1",
                 "evaluate": True,
                 "shoulder_variant": "patch-v1",
                 "target_radius_limit": np.array([0.05, 0.05]),
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.Neural(),
                 "reward_function": iso_pointing_rewards.NegativeExpDistanceWithHitBonus(k=10),
                 "callbacks": []},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

mobl_arms_remote_driving_v1 = {
  "name": "driving-updated-patch-v1",
  "model": PPO,
  "total_timesteps": 200_000_000,
  "env_name": "UIB:mobl-arms-remote-driving-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"direction": "horizontal",
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.Neural(),
                 "episode_length_seconds_extratime": 6,
                 "episode_length_seconds": 4,
                 # "car_velocity_threshold": 0.0,
                 # "reward_function_joystick": driving_rewards.NegativeExpDistance(shift=-1, scale=1),
                 # "reward_function_target": driving_rewards.NegativeExpDistance(shift=-1, scale=0.1),
                 # "reward_function_joystick_bonus": driving_rewards.RewardBonus(bonus=0.8, onetime=True),
                 # "reward_function_target_bonus": driving_rewards.RewardBonus(bonus=8, onetime=False)
                 "shoulder_variant": "patch-v1",
                 "gamepad_scale_factor": 3
                 },
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

mobl_arms_remote_driving_v2 = {
  "name": "driving-updated-patch-v2",
  "model": PPO,
  "total_timesteps": 200_000_000,
  "env_name": "UIB:mobl-arms-remote-driving-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"direction": "horizontal",
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.Neural(),
                 "episode_length_seconds_extratime": 6,
                 "episode_length_seconds": 4,
                 # "car_velocity_threshold": 0.0,
                 # "reward_function_joystick": driving_rewards.NegativeExpDistance(shift=-1, scale=1),
                 # "reward_function_target": driving_rewards.NegativeExpDistance(shift=-1, scale=0.1),
                 # "reward_function_joystick_bonus": driving_rewards.RewardBonus(bonus=0.8, onetime=True),
                 # "reward_function_target_bonus": driving_rewards.RewardBonus(bonus=8, onetime=False)
                 "shoulder_variant": "patch-v2"
                 },
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

#### Below are the configs used to train the models for the UIST paper ####

mobl_arms_pointing_uist = {
  "name": "pointing-v1-patch-v1",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-pointing-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"target_radius_limit": np.array([0.05, 0.15]),
                 "action_sample_freq": 20,
                 "shoulder_variant": "patch-v1",
                 "effort_term": effort_terms.Neural(),
                 "reward_function": pointing_rewards.NegativeExpDistanceWithHitBonus(k=10),
                 "callbacks": []},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

mobl_arms_iso_pointing_uist = {
  "name": "iso-pointing-general-patch-v1-dwell",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-iso-pointing-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"user": "general",
                 "evaluate": True,
                 "shoulder_variant": "patch-v1",
                 "target_radius_limit": np.array([0.05, 0.05]),
                 "action_sample_freq": 20,
                 "effort_term": effort_terms.Neural(),
                 "reward_function": iso_pointing_rewards.NegativeExpDistanceWithHitBonus(k=10),
                 "callbacks": []},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

target_speed_curriculum_uist = LinearCurriculum("target_speed_curriculum_uist", start_value=0, end_value=1,
                                                end_timestep=40_000_000, start_timestep=15_000_000)
mobl_arms_tracking_uist = {
  "name": "tracking-v1-patch-v1",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-tracking-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"target_radius": 0.05,
                 "action_sample_freq": 20,
                 "shoulder_variant": "patch-v1",
                 "effort_term": effort_terms.Neural(),
                 "reward_function": tracking_rewards.NegativeDistance(),
                 "freq_curriculum": target_speed_curriculum_uist.value,
                 "episode_length_seconds": 10,
                 "callbacks": [target_speed_curriculum_uist]},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}

mobl_arms_button_press_uist = {
  "name": "button-press-v1-patch-v1-smaller-buttons",
  "model": PPO,
  "total_timesteps": 100_000_000,
  "env_name": "UIB:mobl-arms-button-press-v1",
  "start_method": 'spawn' if 'Microsoft' in uname().release else 'forkserver',
  "num_workers": 10,
  "device": "cuda",
  "env_kwargs": {"action_sample_freq": 20,
                 "shoulder_variant": "patch-v1",
                 "effort_term": effort_terms.Neural(),
                 "reward_function": button_press_rewards.NegativeExpDistanceWithHitBonus()},
  "policy_type": MultiInputActorCriticPolicyTanhActions,
  "policy_kwargs": {"activation_fn": torch.nn.LeakyReLU,
                    "net_arch": [256, 256],
                    "log_std_init": 0.0,
                    "features_extractor_class": VisualAndProprioceptionExtractor,
                    "normalize_images": False},
  "lr": linear_schedule(initial_value=5e-5, min_value=1e-7, threshold=0.8),
  "nsteps": 4000, "batch_size": 500, "target_kl": 1.0, "save_freq": 5000000
}
