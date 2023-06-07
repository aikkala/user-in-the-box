from scipy.spatial.transform import Rotation
import pathlib
import os
from datetime import datetime
import sys
import select
import numpy as np
import re

from ruamel.yaml import YAML

from .transformations import transformation_matrix


def parent_path(file):
  return pathlib.Path(file).parent.absolute()

def project_path():
  return pathlib.Path(__file__).parent.parent.absolute()

def output_path():
  return os.path.join(project_path().parent.absolute(), "simulators")

def strtime():
  return datetime.utcfromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%dT%H-%M-%SZ')

def timeout_input(prompt, timeout=30, default=""):
  print(prompt, end='\n>> ', flush=True)
  inputs, outputs, errors = select.select([sys.stdin], [], [], timeout)
  print()
  return sys.stdin.readline().strip() if inputs else default

# Numerically stable sigmoid
def sigmoid(x):
  return np.exp(-np.logaddexp(0, -x))

def is_suitable_package_name(name):
  match = re.match("^[a-z0-9_]*$", name)
  return match is not None and name[0].isalpha()

def parse_yaml(yaml_file):
  yaml = YAML()
  with open(yaml_file, 'r') as stream:
    parsed = yaml.load(stream)
  return parsed

def write_yaml(data, file):
  yaml = YAML()
  with open(file, "w") as stream:
    yaml.dump(data, stream)

def img_history(imgs, k=0.9):

  # Make sure intensities are properly normalised
  N = len(imgs)

  img = np.zeros_like(imgs[0], dtype=np.float)
  norm = 0

  for i in range(N):
    coeff = np.exp(-((N-1)-i)*k)
    img += coeff * imgs[i]
    norm += coeff

  return img / (255*norm)

def initialise_pos_and_quat(model, data, aux_body, relpose, body):
  """ Initialise pos and quat of body according to the relpose wrt to aux_body"""
  T1 = transformation_matrix(pos=data.body(aux_body).xpos, quat=data.body(aux_body).xquat)
  T2 = transformation_matrix(pos=relpose[:3], quat=relpose[3:])
  T = np.matmul(T1, np.linalg.inv(T2))
  model.body(body).pos = T[:3, 3]
  model.body(body).quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)
