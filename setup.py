from setuptools import setup

setup(
   name='uitb',
   version='0.2.0',
   author='Aleksi Ikkala',
   author_email='aleksi.ikkala@gmail.com',
   packages=['uitb'],
   url='https://github.com/aikkala/user-in-the-box',
   license='LICENSE',
   description='Modeling and simulating HCI interaction tasks in MuJoCo',
   long_description=open('README.md').read(),
   python_requires='>=3.7',
   install_requires=[
       "gym", #>=0.26.0",
       "mujoco>=2.2.0",
       "stable_baselines3>=1.4.0", "torch",
       "wandb", "tensorboard",
       "numpy", "matplotlib", "scipy",
       "opencv-python",
       "ruamel.yaml",
   ],
)
