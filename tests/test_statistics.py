import numpy as np

from hdfx.statistics import Welford


def test_batched_welford():
  for seed in range(3):
    np.random.seed(seed)
    for N in [100, 1000, 10_000]:
      W = 32
      ndim = 5
      data = np.random.rand(N, W, ndim)
      data_flat = np.reshape(data, (-1, ndim))
      target_mean = np.mean(data_flat, axis=0)
      target_std = np.std(data_flat, axis=0)
      w = Welford(ndim)
      batches = np.array_split(data, 32, axis=0)
      for b in batches:
        w.update_batch(b)
      assert np.allclose(target_mean, w.mean)
      assert np.allclose(target_std, w.std)
