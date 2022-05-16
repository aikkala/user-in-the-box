from setuptools import setup

setup(name='uitb',
      version='0.1.1',
      author="Aleksi Ikkala",
      author_email="aleksi.ikkala@aalto.fi",
      packages=["uitb"],
      python_requires='>3.5',
      install_requires=['gym', 'mujoco==2.1.5', 'stable-baselines3', 'tensorboard', 'torch', 'ruamel.yaml']
)
