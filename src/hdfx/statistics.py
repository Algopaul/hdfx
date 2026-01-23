import numpy as np


class Welford:

  def __init__(self, ndim, dtype=np.float64):
    self.count = 0
    self.ndim = ndim
    self.dtype = dtype
    self.mean = np.zeros(ndim, dtype=dtype)
    self.m2 = np.zeros(ndim, dtype=dtype)

  def __repr__(self):
    return (f"Welford(n={self.count}, ndim={self.ndim})")

  def update_batch(self, x):
    '''
    update statistics a batch x with shape (..., C)
    '''
    batch_size = np.prod(np.array(x.shape[:-1]))
    assert x.shape[-1] == self.ndim

    y = np.reshape(x, (-1, self.ndim)).astype(self.dtype)
    batch_mean = np.mean(y, axis=0)
    batch_m2 = np.sum((y - batch_mean)**2, axis=0)

    if self.count == 0:
      self.mean = batch_mean
      self.m2 = batch_m2
      self.count = batch_size
    else:
      delta = batch_mean - self.mean
      n_total = self.count + batch_size
      self.mean += delta * batch_size / n_total
      self.m2 += batch_m2 + delta**2 * self.count * batch_size / n_total
      self.count = n_total

  @property
  def std(self):
    return np.sqrt(self.m2 / (self.count))
