import gym
from gym.wrappers.time_limit import TimeLimit
import os
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from utils import opensim_file

def generate_random_trajectories(env, num_trajectories=1_000, trajectory_length_seconds=10, render_mode="human"):

  def add_noise(actions, rate, scale):
    return np.random.normal(loc=rate*actions, scale=scale)

  # Vary noise scale and noise rate
  std_limits = [2, 4]
  timescale_limits = [0.5, 4]

  # Ignore first states cause they all start from the default pose
  ignore_first_seconds = 5
  assert trajectory_length_seconds > ignore_first_seconds, \
    f"Trajectory must be longer than {ignore_first_seconds} seconds"
  ignore_first = int(ignore_first_seconds/env.dt)

  # Trajectory length in steps
  trajectory_length = int(trajectory_length_seconds/env.dt)

  # Collect states
  states = np.zeros((num_trajectories, trajectory_length, len(env.independent_joints)))

  for trajectory_idx in range(num_trajectories):

    env.reset()
    env.render(mode=render_mode)

    # Sample noise statistics, different for each episode
    rate = np.exp(-env.dt / np.random.uniform(*timescale_limits))
    scale = np.random.uniform(*std_limits) * np.sqrt(1-rate*rate)

    # Start with zero actions
    actions = np.zeros((env.action_space.shape[0]))


    for step_idx in range(trajectory_length):

      # Add noise to actions
      actions = add_noise(actions, rate, scale)

      # Step
      env.step(actions)
      env.render(mode=render_mode)

      states[trajectory_idx, step_idx] = env.sim.data.qpos[env.independent_joints].copy()

  return states[:, ignore_first:]


# def read_opensim_file(filepath):
#
#   # Read filepath and parse it
#   with open(filepath) as f:
#     text = f.read()
#   opensim_xml = xmltodict.parse(text)
#
#   vOpenSim = opensim_xml["OpenSimDocument"]["@Version"]
#
#   if vOpenSim.startswith("3"):
#     opensim_xml_bodies = opensim_xml["OpenSimDocument"]["Model"]["BodySet"]["objects"]
#
#     bodies_scale = {}
#     bodies_mass = {}
#     bodies_mass_center = {}
#     bodies_inertia = {}
#     bodies_position = {}
#
#     # TODO: check whether thorax.stl is scaled correctly
#     for obj in opensim_xml_bodies["Body"]:
#       body_name = obj["@name"]
#       if "VisibleObject" in obj:
#         # Get scaling of VisibleObject
#         visible_object_scale = np.array(obj["VisibleObject"]["scale_factors"].split(), dtype=float)
#         bodies_scale.update({body_name: visible_object_scale})
#
#       bodies_mass.update({body_name: float(obj["mass"])})
#       bodies_mass_center.update({body_name: np.array(obj["mass_center"].split(), dtype=float)})
#       # bodies_inertia.update({body_name: np.array([obj[x] for x in
#       #                     ["inertia_xx", "inertia_yy", "inertia_zz",
#       #                      "inertia_xy", "inertia_xz", "inertia_yz"]], dtype=float)})
#
#       joint = obj["Joint"]
#       if body_name != "ground":
#         if joint is None or len(joint) == 0:
#           print('ERROR! Body {} passed.'.format(body_name))
#           pass
#         else:
#           assert len(joint) == 1, 'TODO Multiple joints for one body'
#           joint = joint[list(joint)[0]]
#           bodies_position.update({body_name: np.array(joint["location_in_parent"].split(), dtype=float)})
#
#
#   elif vOpenSim.startswith("4"):
#
#     opensim_xml_bodies = opensim_xml["OpenSimDocument"]["Model"]["BodySet"]["objects"]
#     opensim_xml_joints = opensim_xml["OpenSimDocument"]["Model"]["JointSet"]["objects"]
#
#     bodies_scale = {}
#     bodies_scale_old = {}  # used to scale joints not available in OpenSim correctly, independent of scale in original MuJoCo file
#     bodies_mass = {}
#     bodies_mass_center = {}
#     bodies_inertia = {}
#     bodies_position = {}
#
#     for obj in opensim_xml_bodies["Body"]:
#       body_name = obj["@name"]
#       if "attached_geometry" in obj:
#         # Get scaling of VisibleObject
#         if obj["attached_geometry"] is None:
#           bodies_scale.update({body_name: np.array([1, 1, 1])})
#         else:
#           try:
#             visible_object_scale = np.array(obj["attached_geometry"]["Mesh"]["scale_factors"].split(),
#                                             dtype=float)
#           except TypeError:
#             visible_object_scale = np.array(obj["attached_geometry"]["Mesh"][0]["scale_factors"].split(),
#                                             dtype=float)
#           bodies_scale.update({body_name: visible_object_scale})
#       else:
#         print(body_name)
#         raise KeyError
#
#       bodies_mass.update({body_name: float(obj["mass"])})
#       bodies_mass_center.update({body_name: np.array(obj["mass_center"].split(), dtype=float)})
#       input('test required')
#
#       bodies_inertia.update({body_name: strinterval_to_nparray(obj["inertia"])})
#
#     for joint in opensim_xml_joints["CustomJoint"]:
#       body_offsetname = joint["socket_child_frame"]
#       offset_frame_to_use = [i for i in range(len(joint["frames"]["PhysicalOffsetFrame"])) if
#                              joint["frames"]["PhysicalOffsetFrame"][i]["@name"] != body_offsetname]
#       assert len(offset_frame_to_use) == 1
#       # body_name = joint["frames"]["PhysicalOffsetFrame"][offset_frame_to_use[0]]["socket_parent"].split('/')[-1]
#       body_name = body_offsetname.split('_offset')[0]
#       translation = joint["frames"]["PhysicalOffsetFrame"][offset_frame_to_use[0]]["translation"]
#       if body_name != "ground":
#         if joint is None or len(joint) == 0:
#           print('ERROR! Body {} passed.'.format(body_name))
#           pass
#         else:
#           # assert len(joint) == 1, 'TODO Multiple joints for one body'
#           # joint = joint[list(joint)[0]]
#           bodies_position.update({body_name: np.array(translation.split(), dtype=float)})
#
#     # set missing body position to zero vector:
#     for body_name in [body for body in bodies_mass.keys() if body not in bodies_position.keys()]:
#       bodies_position.update({body_name: np.array([0, 0, 0])})
#
#     # PATCH: add entry for "thorax" body if removed from OpenSim file
#     if 'thorax' not in bodies_scale:
#       bodies_scale['thorax'] = bodies_scale['clavicle']
#
#   else:
#     raise NotImplementedError
#
if __name__=="__main__":

  env_name = 'UIB:mobl-arms-muscles-v0'
  train = True
  render_mode = "human"  #"human", "rgb-array"
  start_method = 'forkserver'  # forkserver in linux, spawn in windows/wsl
  generate_experience = False
  experience_file = 'experience.npy'
  num_cpu = 7
  output_dir = os.path.join('output', env_name)
  checkpoint_dir = os.path.join(output_dir, 'checkpoint')
  log_dir = os.path.join(output_dir, 'log')

  # Leave for future kwargs
  env_kwargs = {}

  # Initialise environment
  env = gym.make(env_name, **env_kwargs)
  env.step(env.action_space.sample())

  osim_file = opensim_file("UIB/envs/mobl_arms/models/MoBL_ARMS_model_for_mujoco.osim")

  optimal_fiber_length = osim_file.optimal_fiber_length

  # import sys
  # orig_stdout = sys.stdout
  # f = open('MoBL_ARMS_analysis.txt', 'w')
  # sys.stdout = f

  # modify scale ratios of optimal fiber length
  env.sim.model.actuator_gainprm[:, :2] = [[0.5, 2]] * env.sim.model.nu

  #print("Scale ratios of optimal fiber length: [0.75, 1.05] (MuJoCo default)")
  print("Scale ratios of optimal fiber length: [0.5, 2]")

  for actuator_id in [i for i in range(env.sim.model.nu) if env.sim.model.actuator_trntype[i] == 3]:  #only consider tendon actuators
    LO = (env.sim.model.actuator_lengthrange[actuator_id][0] - env.sim.model.actuator_lengthrange[actuator_id][1]) / (
              env.sim.model.actuator_gainprm[actuator_id][0] - env.sim.model.actuator_gainprm[actuator_id][1])
    LT = env.sim.model.actuator_lengthrange[actuator_id][0] - env.sim.model.actuator_gainprm[actuator_id][0] * LO
    #LT = env.sim.model.actuator_lengthrange[actuator_id][1] - env.sim.model.actuator_gainprm[actuator_id][1] * LO

    actuator_name = env.sim.model.actuator_id2name(actuator_id)
    tendon_id = env.sim.model.actuator_trnid[actuator_id][0]
    print(f"{actuator_name}:\n\tactuator_length (MuJoCo): {env.sim.data.actuator_length[actuator_id]}\n\ttendon length LT (MuJoCo): {LT}\n\tmuscle length LM (MuJoCo): {env.sim.data.actuator_length[actuator_id] - LT}")
    #print(f"{actuator_name}:\n\tactuator_length (MuJoCo): {env.sim.data.actuator_length[actuator_id]}\n\ttendon length (MuJoCo): {env.sim.data.ten_length[tendon_id]}\n\tmuscle length (MuJoCo): {env.sim.data.actuator_length[actuator_id] - env.sim.data.ten_length[tendon_id]}")
    #print(f"{actuator_name}:actuator_length: \n\t{env.sim.data.actuator_length}ten_length: \n\t{env.sim.model.tendon_length0}")

    print(f"\n\tlength range (MuJoCo): {env.sim.model.actuator_lengthrange[actuator_id]}\n\toptimal fiber length LO (MuJoCo): {LO}")
    if actuator_name in optimal_fiber_length:
      print(f"\toptimal fiber length (OpenSim): {optimal_fiber_length[actuator_name]}")

  # sys.stdout = orig_stdout
  # f.close()
  input('DEBUG END.')

  if generate_experience:
   experience = generate_random_trajectories(env, num_trajectories=1000, trajectory_length_seconds=10, render_mode=render_mode)
   np.save(experience_file, experience)
  else:
   try:
    experience = np.load(experience_file)
   except FileNotFoundError:
     pass

  # Do the training
  if train:

    # Initialise parallel envs
    parallel_envs = make_vec_env(env_name, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs,
                                 vec_env_kwargs={'start_method': start_method})

    # Policy parameters
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                         net_arch=[dict(pi=[128, 128], vf=[128, 128])],
                         log_std_init=0.0)

    # Initialise policy
    model = PPO('MlpPolicy', parallel_envs, verbose=1, policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir)

    # Initialise a callback for checkpoints
    save_freq = 1000000 // num_cpu
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_dir, name_prefix='model')

    # Do the learning
    model.learn(total_timesteps=20_000_000, callback=[checkpoint_callback])

  else:

    # Load previous policy
    model = PPO.load(os.path.join(checkpoint_dir, 'model_6999888_steps'))

  # Visualise evaluations, perhaps save a video as well
  while True:
    obs = env.reset()
    env.render(mode=render_mode)
    done = False
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, rewards, done, info = env.step(action)
      env.render(mode=render_mode)