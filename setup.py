from setuptools import setup

setup(name='UIB',
      version='0.0.2',
      author="Aleksi Ikkala",
      author_email="aleksi.ikkala@aalto.fi",
      packages=["UIB"],
      # this branch should be used for evaluation only
      install_requires=['gym', 'matplotlib', 'pandas', 'scipy', 'numpy', 'xmltodict',
                        'uitb_tools @ git+https://github.com/fl0fischer/uitb-tools.git#egg=uitb_tools-0.1.0'],
      extra_requires={'training': ['mujoco_py', 'stable-baselines3', 'tensorboard', 'torch']}
)
