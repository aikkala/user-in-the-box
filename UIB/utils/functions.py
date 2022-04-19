import pathlib
import os
from datetime import datetime
import sys
import select
import numpy as np
from distutils.dir_util import copy_tree


def parent_path(file):
  return pathlib.Path(file).parent.absolute()

def project_path():
  return pathlib.Path(__file__).parent.parent.absolute()

def output_path():
  return os.path.join(project_path().parent.absolute(), "output")

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