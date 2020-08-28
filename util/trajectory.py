import os
import numpy as np


class CameraPose:

  def __init__(self, meta, mat):
    self.metadata = meta
    self.pose = mat

  def __str__(self):
    return 'metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
        "pose : " + "\n" + np.array_str(self.pose)


def read_trajectory(filename, dim=4):
  traj = []
  print('-'*20)
  print(filename)
  print('-'*20)
  assert os.path.exists(filename)
  with open(filename, 'r') as f:
    metastr = f.readline()
    while metastr:
      metadata = list(map(int, metastr.split()))
      mat = np.zeros(shape=(dim, dim))
      for i in range(dim):
        matstr = f.readline()
        mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
      traj.append(CameraPose(metadata, mat))
      metastr = f.readline()
    print(traj)
    return traj


def write_trajectory(traj, filename, dim=4):
  print('----------------write trajectory----------------')
  print(filename)
  with open(filename, 'w') as f:
    print(traj)
    for x in traj:
      p = x.pose.tolist()
      print('|'*20)
      f.write(' '.join(map(str, x.metadata)) + '\n')
      f.write('\n'.join(' '.join(map('{0:.12f}'.format, p[i])) for i in range(dim)))
      f.write('\n')
      print('|'*20)
