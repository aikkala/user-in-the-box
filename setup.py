from setuptools import setup

setup(name='UIB',
      version='0.0.1',
      author="Aleksi Ikkala",
      author_email="aleksi.ikkala@aalto.fi",
      packages=["UIB"],
      install_requires=['gym', 'mujoco_py']
)
