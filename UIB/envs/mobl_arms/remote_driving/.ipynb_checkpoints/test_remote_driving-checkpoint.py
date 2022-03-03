from UIB.envs.mobl_arms.remote_driving.DrivingEnv import DrivingEnv
import mujoco_py

#xml_file = "/home/markus/mujoco/user-in-the-box/UIB/envs/mobl_arms/remote_driving/models/test_model.xml"

#model = mujoco_py.load_model_from_path(xml_file)

env = DrivingEnv()
env.step(0)