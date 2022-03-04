import gym
import os
import numpy as np
import re
import argparse
import UIB



def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(env):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  # Grab images
  width, height = env.metadata["imagesize"]
  img = np.flipud(env.sim.render(height=height, width=width, camera_name='for_testing'))
  ocular_img = np.flipud(env.sim.render(height=height//4, width=width//4, camera_name='oculomotor'))

  # Embed ocular image into free image
  i = height - height//4
  j = width - width//4
  img[i:, j:] = ocular_img

  return img

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Forward simulation with some fixed policy.')
  parser.add_argument('env_name', type=str,
                      help='name of the environment')
  # parser.add_argument('checkpoint_dir', type=str,
  #                     help='directory where checkpoints are')
  # parser.add_argument('--checkpoint', type=str, default=None,
  #                     help='filename of a specific checkpoint (default: None, latest checkpoint is used)')
  # parser.add_argument('--num_episodes', type=int, default=10,
  #                     help='how many episodes are evaluated (default: 10)')
  parser.add_argument('--record', action='store_true', help='enable recording')
  parser.add_argument('--out_file', type=str, default='evaluate.mp4',
                      help='output file for recording if recording is enabled (default: ./evaluate.mp4)')
  parser.add_argument('--logging', action='store_true', help='enable logging')
  parser.add_argument('--log_file', default='log',
                      help='output file for log if logging is enabled (default: ./log)')
  args = parser.parse_args()

  # Get env name
  env_name = args.env_name

  # # Load policy
  # print(f'Loading model: {os.path.join(checkpoint_dir, model_file)}')

  # Initialise environment
  env_kwargs = {"target_radius_limit": np.array([0.01, 0.05]), "action_sample_freq": 100, "render_observations": True}
  #env_kwargs = {}
  env = gym.make(env_name, **env_kwargs)

  imgs = []

  apply_car_dynamics = True

  # Reset environment
  obs = env.reset()
  done = False
  reward = 0

  if args.record:
    imgs.append(grab_pip_image(env))

  # Loop until episode ends
  while not done:

    # Define actions
    #action, _states = model.predict(obs, deterministic=False)
    action = np.zeros((env.sim.model.na, ))
    #action = np.random.uniform(0, 1, size=env.sim.model.na)

    # # TESTING - WARNING: Requires commenting "self.update_car_dynamics()" in step()!
    # throttle = -50*(20*np.pi/180)  #* np.random.uniform(0, 1)
    # env.sim.data.qfrc_applied[env.model._joint_name2id[env.engine_back_joint]] = throttle
    # # env.sim.data.qfrc_applied[env.model._joint_name2id["steering_gear:rot-y"]] = throttle
    # env.sim.data.qfrc_applied[env.model._joint_name2id[env.engine_front_joint]] = throttle
    # apply_car_dynamics = False
    # ## ALTERNATIVE: directly apply torque to slide joint:
    # # env.sim.data.qfrc_applied[env.model._joint_name2id["car"]] = -throttle

    # Take a step
    obs, r, done, info = env.step(action, apply_car_dynamics=apply_car_dynamics)
    #input((env.sim.data.qpos[env.model._joint_name2id["car"]]))
    #input((env.sim.data.qpos[env.model._joint_name2id[env.joystick_joint]]))
    # for i in range(env.sim.data.ncon):
    #   print((env.sim.data.contact[i].dim, env.model.geom_id2name(env.sim.data.contact[i].geom1),
    #           env.model.geom_id2name(env.sim.data.contact[i].geom2), env.sim.data.contact[i].dist,
    #           env.sim.data.contact[i].pos))
    #input(env.sim.data.qfrc_constraint[env.model._joint_name2id[env.joystick_joint]])
    #input((env.sim.data.get_body_xpos("car"), env.sim.data.get_body_xpos("target")))
    reward += r

    if args.record:
      # Visualise muscle activation
      env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl[:] * 0.7
      imgs.append(grab_pip_image(env))

  print(f"Episode {0}: length {env.steps*env.dt} seconds ({env.steps} steps), reward {reward}. ")

  # print(f'Average episode length and reward over {args.num_episodes} episodes: '
  #       f'length {np.mean(episode_lengths)*env.dt} seconds ({np.mean(episode_lengths)} steps), reward {np.mean(rewards)}')


  if args.record:
    # Write the video
    env.write_video(imgs, args.out_file)
    print(f'A recording has been saved to file {args.out_file}')