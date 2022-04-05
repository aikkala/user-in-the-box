import gym
import os
import numpy as np
import re
import argparse
import UIB
from PIL import Image, ImageFont, ImageDraw
import matplotlib
import pickle

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def grab_pip_image(env, ocular_img_type="none", show_reward=False, stage_reward=None, acc_reward=None, show_qpos=False):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  # OPTIONAL: Visualise target plane
  if hasattr(env, "target_plane_geom_idx"):
    env.model.geom_rgba[env.target_plane_geom_idx][-1] = 0#.1

  # Grab images
  width, height = env.metadata["imagesize"]
  img = np.flipud(env.sim.render(height=height, width=width, camera_name='backview' if 'driving' in env.spec.id else 'front')) #for_testing #front #rightview #oculomotor
  ocular_img = np.flipud(env.sim.render(height=env.ocular_image_height, width=env.ocular_image_width, camera_name='oculomotor'))

  # Embed ocular image into free image
  if ocular_img_type == "inset":
    i = height - height//4
    j = width - width//4
    img[i:, j:] = ocular_img
  elif ocular_img_type == "full":
    # Show only ocular image
    img = ocular_img
    env.metadata["imagesize"] = (env.ocular_image_width, env.ocular_image_height)

  # Display current and accumulated reward
  if show_reward:
    assert stage_reward is not None
    assert acc_reward is not None
    img = np.array(add_text_to_image(Image.fromarray(img),
                        f"{stage_reward:.2g} / {acc_reward:.2}",
                        pos=(10, 10), color=(99, 207, 163), fontsize=48))

  # Display joint angles
  if show_qpos:
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
      'hot_0.8-0.3', matplotlib.cm.get_cmap('hot')(0.7 + (0.2 - 0.7) * (np.geomspace(0.1, 1, 256) - 0.1) / 0.9))
    for idx, joint_id in enumerate(env.independent_joints):
      jnt_range = env.sim.model.jnt_range[joint_id, :]
      norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5 * (jnt_range[1] - jnt_range[0]))
      qpos_deviation_from_mean = np.abs(env.sim.data.qpos[joint_id] - np.mean(jnt_range))
      img = np.array(
        add_text_to_image(Image.fromarray(img),
                          f"{env.sim.model.joint_id2name(joint_id)}: {env.sim.data.qpos[joint_id]:.2g}",
                          pos=(10, 70 + 30 * idx), color=custom_cmap(norm(qpos_deviation_from_mean), bytes=True),
                          fontsize=24))  # color=(235, 155, 52)

  return img

def add_text_to_image(image, text, font="dejavu/DejaVuSans.ttf", pos=(400, 300), color=(255, 0, 0), fontsize=120):
    draw = ImageDraw.Draw(image)
    draw.text(pos, text, fill=color, font=ImageFont.truetype(font, fontsize))
    #draw.text(pos, text, fill=color, font=ImageFont.truetype("/usr/share/fonts/truetype/" + font, fontsize))
    # draw.text(pos, text, fill=color)
    return image

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
  env_kwargs = {"direction": "horizontal", "target_radius_limit": np.array([0.01, 0.05]),
                "action_sample_freq": 20, "render_observations": True,
                "shoulder_variant": "patch-v1", "gamepad_scale_factor": 1,
                #"user": "U1"
                }

  #env_kwargs = {}
  if "pointing" in env_name:
    env_kwargs["max_trials"] = 1
  env = gym.make(env_name, **env_kwargs)

  imgs = []

  apply_car_dynamics = True

  # Reset environment
  obs = env.reset()
  done = False
  reward = 0

  if args.record:
    imgs.append(grab_pip_image(env))

  # ##############################################################
  # # TESTING - Joint ranges
  # joint_to_test = env.sim.model.joint_name2id("shoulder_rot")
  # nsteps_to_test = 100
  #
  # env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')] = 0  # 1.06  # 0.74
  # env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')] = np.pi/6  # 1.06  # 0.74
  # env.sim.data.qpos[env.sim.model.joint_name2id('elbow_flexion')] = 1.57  # 1.06  # 0.74
  # ## patch-v2:
  # # env.sim.model.jnt_range[env.sim.model.joint_name2id('shoulder_rot'), :] = np.array(
  # #   [-np.pi / 2, np.pi / 9]) - 2 * np.min((env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')], np.pi - env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')])) / np.pi * env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')]
  #
  # eq_shoulder1_r2 = ([idx for idx, i in enumerate(env.sim.model.eq_data) if
  #                     env.sim.model.eq_type[idx] == 2 and (id_1 := env.sim.model.eq_obj1id[idx]) > 0 and (
  #                       id_2 := env.sim.model.eq_obj2id[idx]) and {env.sim.model.joint_id2name(id_1),
  #                                                                  env.sim.model.joint_id2name(id_2)} == {
  #                       "shoulder1_r2",
  #                       "elv_angle"}])
  # assert len(eq_shoulder1_r2) == 1
  # env.sim.model.eq_data[eq_shoulder1_r2[0], 1] = -((np.pi - 2 * env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')]) / np.pi)
  #
  # # ensure constraints
  # ## adjust virtual joints according to active constraints:
  # for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
  #         env.sim.model.eq_obj1id[
  #           (env.sim.model.eq_type == 2) & (env.sim.model.eq_active == 1)],
  #         env.sim.model.eq_obj2id[
  #           (env.sim.model.eq_type == 2) & (env.sim.model.eq_active == 1)],
  #         env.sim.model.eq_data[(env.sim.model.eq_type == 2) &
  #                                    (env.sim.model.eq_active == 1), 4::-1]):
  #   env.sim.data.qpos[virtual_joint_id] = np.polyval(poly_coefs, env.sim.data.qpos[physical_joint_id])
  #
  # range_to_test = np.linspace(env.sim.model.jnt_range[joint_to_test, 0],
  #                             env.sim.model.jnt_range[joint_to_test, 1], nsteps_to_test)
  # print('RANGE: {}'.format(env.sim.model.jnt_range[joint_to_test, :]))
  # for i in range(nsteps_to_test):
  #   env.sim.data.qpos[joint_to_test] = range_to_test[i]
  #
  #   # EXPLICITLY ENFORCE EQUALITY CONSTRAINT:
  #   ##env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')] = -env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')]
  #   env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')] = -((np.pi - 2*env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')])/np.pi) * env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')]
  #   #env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')] = -np.cos(env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')]) * env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')]
  #   #input((env.sim.model.jnt_range[env.sim.model.joint_name2id('shoulder_rot'), :], env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')]+env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_rot')], env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_rot')]))
  #
  #   # # UPDATE EQUALITY CONSTRAINT (needs to be (manually) satisfied in initial state!):
  #   env.sim.model.eq_data[eq_shoulder1_r2[0], 1] = -((np.pi - 2 * env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')]) / np.pi)
  #   # #env.sim.model.eq_solimp[eq_shoulder1_r2[0], :3] = [0.9999, 0.9999, 100]
  #   # #input((env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')], env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')]))
  #
  #   # if joint_to_test == env.sim.model.joint_name2id(
  #   #         'elv_angle'):  # include joint equality constraint between "elv_angle" and "shoulder1_r2" and positive angle of "shoulder_elv"
  #   #   env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')] = 3.14 #1.06  # 0.74
  #   #   env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_rot')] = -1.5  # 0.74
  #   #   env.sim.data.qpos[env.sim.model.joint_name2id('elbow_flexion')] = 1.57  # 0.74
  #   #   env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')] = -range_to_test[
  #   #     i]  # i.e., "qpos[eval_env.sim.model.joint_name2id('shoulder1_r2')] = -qpos[eval_env.sim.model.joint_name2id('elv_angle')]"!!!
  #   # elif joint_to_test == env.sim.model.joint_name2id('shoulder_rot'):
  #   #   env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')] = 2.3  # 0.74
  #   #   env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')] = 1.57  # 0.74
  #   env.sim.forward()
  #   if args.record:
  #     # Visualise muscle activation
  #     env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl[:] * 0.7
  #     imgs.append(grab_pip_image(env))
  # if args.record:
  #   # Write the video
  #   env.write_video(imgs, args.out_file)
  #   print(f'A recording has been saved to file {args.out_file}')
  # raise SystemExit(0)
  # ##############################################################

  # # ensure constraints
  # ## adjust virtual joints according to active constraints:
  # for (virtual_joint_id, physical_joint_id, poly_coefs) in zip(
  #         env.sim.model.eq_obj1id[
  #           (env.sim.model.eq_type == 2) & (env.sim.model.eq_active == 1)],
  #         env.sim.model.eq_obj2id[
  #           (env.sim.model.eq_type == 2) & (env.sim.model.eq_active == 1)],
  #         env.sim.model.eq_data[(env.sim.model.eq_type == 2) &
  #                                    (env.sim.model.eq_active == 1), 4::-1]):
  #   env.sim.data.qpos[virtual_joint_id] = np.polyval(poly_coefs, env.sim.data.qpos[physical_joint_id])

  # # RUN FORWARD SIM. WITH STORED ACTION SEQ. - load stored action
  # stored_initstate = np.load("test_initqposqvel.npy")
  # assert stored_initstate.shape == (2, len(env.independent_joints)), "Wrong number of DOFs"
  # env.sim.data.qpos[env.independent_joints] = stored_initstate[0]
  # env.sim.data.qvel[env.independent_joints] = stored_initstate[1]
  # stored_action = np.load("test_action.npy")
  # assert stored_action.shape[1] == env.sim.model.na, "Wrong number of controls."
  # new_qpos = []
  # new_qvel = []

  step_idx = 0

  # ##############################################################
  # # Replay episode from log
  episode_id = 'episode_00'
  # with open("/home/florian/user-in-the-box/output/UIB:mobl-arms-button-press-v1/button-press-v1-patch-v1-smaller-buttons/state_log.pickle",
  #         'rb') as f:
  #     state_log = pickle.load(f)
  # with open(
  #         "/home/florian/user-in-the-box/output/UIB:mobl-arms-button-press-v1/button-press-v1-patch-v1-smaller-buttons/action_log.pickle",
  #         'rb') as f:
  #     action_log = pickle.load(f)
  with open(
          "/home/florian/user-in-the-box/output/UIB:mobl-arms-tracking-v1/tracking-v1-patch-v1/state_log_0.5.pickle",
          'rb') as f:
    state_log = pickle.load(f)
  with open(
          "/home/florian/user-in-the-box/output/UIB:mobl-arms-tracking-v1/tracking-v1-patch-v1/action_log_0.5.pickle",
          'rb') as f:
    action_log = pickle.load(f)
  qpos_replay = np.squeeze(state_log[episode_id]['qpos'])
  qvel_replay = np.squeeze(state_log[episode_id]['qvel'])
  ctrl_replay = np.squeeze(action_log[episode_id]['ctrl'])
  #car_pos_replay = np.squeeze(state_log[episode_id]['car_xpos'])
  target_pos_replay = np.squeeze(state_log[episode_id]['target_position'])

  assert len(ctrl_replay) == len(qpos_replay) - 1

  eq_shoulder1_r2 = ([idx for idx, i in enumerate(env.sim.model.eq_data) if
                      env.sim.model.eq_type[idx] == 2 and (id_1 := env.sim.model.eq_obj1id[idx]) > 0 and (
                        id_2 := env.sim.model.eq_obj2id[idx]) and {env.sim.model.joint_id2name(id_1),
                                                                   env.sim.model.joint_id2name(id_2)} == {
                        "shoulder1_r2",
                        "elv_angle"}])
  assert len(eq_shoulder1_r2) == 1
  env.sim.model.eq_data[eq_shoulder1_r2[0], 1] = -((np.pi - 2 * env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')]) / np.pi)

  # # Reset car joint, since position of body car is updated directly below
  # env.sim.data.qpos[env.sim.model.joint_name2id('car')] = 0

  #for qpos_current, qvel_current, ctrl_current, car_pos_current, target_pos_current in zip(qpos_replay, qvel_replay, ctrl_replay, car_pos_replay, target_pos_replay):
  #for qpos_current, qvel_current, ctrl_current in zip(qpos_replay, qvel_replay, ctrl_replay):
  for qpos_current, qvel_current, ctrl_current, target_pos_current in zip(qpos_replay, qvel_replay, ctrl_replay, target_pos_replay):
    # Set joint angles
    env.sim.data.qpos[env.independent_joints] = qpos_current
    env.sim.data.qvel[env.independent_joints] = qvel_current

    # # Set car and target position
    # ## env.sim.data.qpos[env.sim.model._joint_name2id[env.car_joint]] = 2.75
    # ## env.set_target_position(np.array([0, -0.5, 0]))
    env.set_target_position(target_pos_current - env.target_origin)
    # env.set_car_position(car_pos_current)
    # #input((env.sim.data.body_xpos[env.sim.model.body_name2id("target")], env.target_origin, env.target_position, env.model.body_pos[env.model._body_name2id["target"]], target_pos_current))

    # EXPLICITLY ENFORCE EQUALITY CONSTRAINT:
    ##env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')] = -env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')]
    env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')] = -((np.pi - 2*env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')])/np.pi) * env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')]
    #env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')] = -np.cos(env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')]) * env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')]
    #input((env.sim.model.jnt_range[env.sim.model.joint_name2id('shoulder_rot'), :], env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')]+env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_rot')], env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_rot')]))

    # # UPDATE EQUALITY CONSTRAINT (needs to be (manually) satisfied in initial state!):
    env.sim.model.eq_data[eq_shoulder1_r2[0], 1] = -((np.pi - 2 * env.sim.data.qpos[env.sim.model.joint_name2id('shoulder_elv')]) / np.pi)
    # #env.sim.model.eq_solimp[eq_shoulder1_r2[0], :3] = [0.9999, 0.9999, 100]
    # #input((env.sim.data.qpos[env.sim.model.joint_name2id('elv_angle')], env.sim.data.qpos[env.sim.model.joint_name2id('shoulder1_r2')]))

    env.sim.forward()
    assert all(env.sim.data.qpos[env.independent_joints] == qpos_current)
    assert all(env.sim.data.qvel[env.independent_joints] == qvel_current)
    if args.record:
      # Visualise muscle activation
      env.model.tendon_rgba[:, 0] = 0.3 + ctrl_current[:] * 0.7
      imgs.append(grab_pip_image(env, ocular_img_type="none", show_qpos=False))
  if args.record:
    # Write the video
    env.write_video(imgs, args.out_file)
    print(f'A recording has been saved to file {args.out_file}')
  raise SystemExit(0)
  ##############################################################


  # Loop until episode ends
  while not done:
    step_idx += 1

    # Define actions
    #action, _states = model.predict(obs, deterministic=False)
    action = np.zeros((env.sim.model.na, ))
    #action = np.random.uniform(0, 1, size=env.sim.model.na)

    # ## Car driving task - enforce joystick contact:
    # if step_idx >= 10:
    #   env.sim.model.geom_rgba[env.sim.model.geom_name2id("hand_2distph")] = [0, 1, 0, 1]
    #
    #   fingertip_to_joystick_constraint = [idx for idx in range(env.sim.model.neq) if env.sim.model.eq_type[idx] == 4 and {env.sim.model.geom_id2name(env.sim.model.eq_obj1id[idx]), env.sim.model.geom_id2name(env.sim.model.eq_obj2id[idx])} == {"hand_2distph", "thumb-stick-1-virtual"}]
    #   assert len(fingertip_to_joystick_constraint) == 1
    #   env.sim.model.eq_active[fingertip_to_joystick_constraint[0]] = True
    #   print((step_idx, info["fingertip_at_joystick"], env.dist_fingertip_to_joystick))
    # if step_idx == 200:
    #   break
    # if step_idx == 500:
    #   env.sim.data.qpos[env.sim.model.joint_name2id("shoulder_elv")] = np.pi/2

    # if step_idx == 10:
    #   break

    # # RUN FORWARD SIM. WITH STORED ACTION SEQ. - execute action
    # if step_idx == len(stored_action):
    #     break
    # action = stored_action[step_idx]
    # step_idx += 1

    # # TESTING - WARNING: Requires commenting "self.update_car_dynamics()" in step()!
    # throttle = -50*(20*np.pi/180)  #* np.random.uniform(0, 1)
    # env.sim.data.qfrc_applied[env.model._joint_name2id[env.engine_back_joint]] = throttle
    # # env.sim.data.qfrc_applied[env.model._joint_name2id["steering_gear:rot-y"]] = throttle
    # env.sim.data.qfrc_applied[env.model._joint_name2id[env.engine_front_joint]] = throttle
    # apply_car_dynamics = False
    # # ## ALTERNATIVE: directly apply torque to slide joint:
    # # env.sim.data.qfrc_applied[env.model._joint_name2id["car"]] = -throttle
    #print([env.sim.model.joint_id2name(efc_id) for idx, efc_id in enumerate(env.sim.data.efc_id[:]) if env.sim.data.efc_type[idx] == 3])

    # Take a step
    if "remote-driving" in env_name:
      obs, r, done, info = env.step(action, apply_car_dynamics=apply_car_dynamics)
    else:
      obs, r, done, info = env.step(action)
    #input((env.sim.data.qpos[env.model._joint_name2id["car"]]))
    #input((env.sim.data.qpos[env.model._joint_name2id[env.joystick_joint]]))
    # for i in range(env.sim.data.ncon):
    #   print((env.sim.data.contact[i].dim, env.model.geom_id2name(env.sim.data.contact[i].geom1),
    #           env.model.geom_id2name(env.sim.data.contact[i].geom2), env.sim.data.contact[i].dist,
    #           env.sim.data.contact[i].pos))
    #input(env.sim.data.qfrc_constraint[env.model._joint_name2id[env.joystick_joint]])
    #input((env.sim.data.get_body_xpos("car"), env.sim.data.get_body_xpos("target")))
    reward += r

    # input((action, env.sim.data.ctrl[:], env.sim.data.act[:]))

    # # TESTING - Visualize a few task conditions:
    # if env.steps % 20 == 0:
    #   # Spawn a new car location
    #   env.spawn_car()
    #
    #   # Spawn a new target location (depending on current car location)
    #   env.spawn_target()

    # # RUN FORWARD SIM. WITH STORED ACTION SEQ. - append achieved postures
    # new_qpos.append(env.sim.data.qpos[env.independent_joints])
    # new_qvel.append(env.sim.data.qvel[env.independent_joints])

    if args.record:
      # Visualise muscle activation
      env.model.tendon_rgba[:, 0] = 0.3 + env.sim.data.ctrl[:] * 0.7
      imgs.append(grab_pip_image(env, show_reward=True, stage_reward=r, acc_reward=reward))

  print(f"Episode {0}: length {env.steps*env.dt} seconds ({env.steps} steps), reward {reward}. ")

  # print(f'Average episode length and reward over {args.num_episodes} episodes: '
  #       f'length {np.mean(episode_lengths)*env.dt} seconds ({np.mean(episode_lengths)} steps), reward {np.mean(rewards)}')

  # # RUN FORWARD SIM. WITH STORED ACTION SEQ. - store achieved postures
  # np.save("new_qpos.npy", np.array(new_qpos))
  # np.save("new_qvel.npy", np.array(new_qvel))

  if args.record:
    # Write the video
    env.write_video(imgs, args.out_file)
    print(f'A recording has been saved to file {args.out_file}')