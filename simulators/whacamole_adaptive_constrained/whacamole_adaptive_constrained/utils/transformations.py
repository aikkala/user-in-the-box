import numpy as np
from scipy.spatial.transform import Rotation


def transformation_matrix(pos, quat=None, matrix=None, scalar_first=True):
  """ A helper function to create transformation matrices from position vectors and rotation matrices or quaternions"""

  # Make sure only 'quat' or 'matrix' is defined
  if (quat is None and matrix is None) or (quat is not None and matrix is not None):
    raise RuntimeError("You need to define either 'quat' or 'matrix', both cannot be None or defined")

  # Transform quaternion into a rotation matrix
  if quat is not None:

    # Make sure Rotation.from_quat gets a quaternion in scalar-last format
    if scalar_first:
      quat = np.roll(quat, -1)

    # Create the rotation matrix
    matrix = Rotation.from_quat(quat).as_matrix()

  # Create the matrix
  T = np.eye(4)
  T[:3, :3] = matrix
  T[:3, 3] = pos
  return T