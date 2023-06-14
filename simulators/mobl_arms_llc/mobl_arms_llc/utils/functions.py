import pathlib
import os
from datetime import datetime
import sys
import select
import numpy as np
import re
from ruamel.yaml import YAML
import importlib


def parent_path(file):
  return pathlib.Path(file).parent.absolute()

def project_path():
  return pathlib.Path(__file__).parent.parent.absolute()

def output_path():
  try:
    from .__simulatorsdir__ import SIMULATORS_DIR
  except (ModuleNotFoundError, ImportError):
    SIMULATORS_DIR = os.path.join(project_path().parent.absolute(), "simulators")
  return SIMULATORS_DIR

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

def importer(import_info):
  """ Imports a class or a function from given module and returns it.

  Args:
    import_info: A dict, must contain keyword 'module'. If neither keywords 'cls' nor 'function' are defined, the module
      is returned. Otherwise, if 'cls' or 'function' is defined, the importer tries to import and return either an
      object of the given class, or a function from the given module
  """

  # Make sure "module" is defined
  if "module" not in import_info:
    raise RuntimeError(f"The import info {import_info} is missing keyword 'module'")

  # Import the module
  module = importlib.import_module(f"{__package__.split('.')[0]}.{import_info['module']}")

  # Check whether "cls" or "function" is defined
  if not any(k in import_info for k in {"cls", "function"}):
    return module
  else:
    imp = import_info["cls"] if "cls" in import_info else import_info["function"]
    instance = getattr(module, imp)
    return instance