from typing import cast

import h5py
import numpy as np
import zarr

from hdfx.statistics import Welford, ds_statistics


# ---------------------------------------------------------------------------
# Welford: correctness
# ---------------------------------------------------------------------------

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


def test_welford_single_batch():
  """One call to update_batch should match numpy directly."""
  rng = np.random.default_rng(0)
  data = rng.standard_normal((200, 8))
  w = Welford(8)
  w.update_batch(data)
  assert np.allclose(w.mean, data.mean(axis=0))
  assert np.allclose(w.std, data.std(axis=0))


def test_welford_incremental_equals_full():
  """Row-by-row updates must match a single bulk update."""
  rng = np.random.default_rng(1)
  data = rng.standard_normal((50, 4))

  w_bulk = Welford(4)
  w_bulk.update_batch(data)

  w_inc = Welford(4)
  for row in data:
    w_inc.update_batch(row[np.newaxis, :])

  assert np.allclose(w_bulk.mean, w_inc.mean)
  assert np.allclose(w_bulk.std, w_inc.std)


def test_welford_count():
  """count should equal total number of rows after updates."""
  w = Welford(3)
  w.update_batch(np.zeros((10, 3)))
  w.update_batch(np.zeros((7, 3)))
  assert w.count == 17


def test_welford_repr():
  w = Welford(4)
  w.update_batch(np.zeros((5, 4)))
  assert "n=5" in repr(w)
  assert "ndim=4" in repr(w)


def test_welford_constant_data_zero_std():
  """Constant data should give std == 0."""
  data = np.full((100, 3), 7.0)
  w = Welford(3)
  w.update_batch(data)
  assert np.allclose(w.mean, 7.0)
  assert np.allclose(w.std, 0.0)


def test_welford_numerically_stable():
  """Large offset shouldn't degrade accuracy (Welford is stable)."""
  rng = np.random.default_rng(2)
  offset = 1e8
  noise = rng.standard_normal((500, 3))
  data = offset + noise

  w = Welford(3)
  for batch in np.array_split(data, 10):
    w.update_batch(batch)

  assert np.allclose(w.mean, data.mean(axis=0), rtol=1e-5)
  assert np.allclose(w.std, data.std(axis=0), rtol=1e-5)


# ---------------------------------------------------------------------------
# ds_statistics: h5py
# ---------------------------------------------------------------------------

def test_ds_statistics_h5_explicit_step(tmp_path):
  rng = np.random.default_rng(3)
  data = rng.standard_normal((80, 6)).astype(np.float32)
  p = tmp_path / "data.h5"
  with h5py.File(p, "w") as f:
    f.create_dataset("x", data=data, chunks=(16, 6))

  with h5py.File(p, "r") as f:
    mean, std = ds_statistics(cast(h5py.Dataset, f["x"]), step=16)

  assert mean.shape == (6,)
  assert std.shape == (6,)
  assert np.allclose(mean, data.mean(axis=0), atol=1e-4)
  assert np.allclose(std, data.std(axis=0), atol=1e-4)


def test_ds_statistics_h5_auto_step(tmp_path):
  """step=None should infer from chunks without error."""
  rng = np.random.default_rng(4)
  data = rng.standard_normal((60, 4)).astype(np.float32)
  p = tmp_path / "data.h5"
  with h5py.File(p, "w") as f:
    f.create_dataset("x", data=data, chunks=(20, 4))

  with h5py.File(p, "r") as f:
    mean, std = ds_statistics(cast(h5py.Dataset, f["x"]))

  assert np.allclose(mean, data.mean(axis=0), atol=1e-4)
  assert np.allclose(std, data.std(axis=0), atol=1e-4)


def test_ds_statistics_h5_no_chunks_fallback(tmp_path):
  """Contiguous (no-chunk) dataset should use the max(64, N//100) fallback."""
  rng = np.random.default_rng(5)
  data = rng.standard_normal((200, 3)).astype(np.float32)
  p = tmp_path / "data.h5"
  with h5py.File(p, "w") as f:
    f.create_dataset("x", data=data)  # no chunks arg → contiguous

  with h5py.File(p, "r") as f:
    mean, std = ds_statistics(cast(h5py.Dataset, f["x"]))

  assert np.allclose(mean, data.mean(axis=0), atol=1e-4)
  assert np.allclose(std, data.std(axis=0), atol=1e-4)



# ---------------------------------------------------------------------------
# ds_statistics: zarr
# ---------------------------------------------------------------------------

def test_ds_statistics_zarr_explicit_step(tmp_path):
  rng = np.random.default_rng(6)
  data = rng.standard_normal((80, 6)).astype(np.float32)
  p = tmp_path / "data.zarr"
  root = cast(zarr.Group, zarr.open(str(p), mode="w"))
  root.create_array("x", data=data, chunks=(16, 6))

  arr = cast(zarr.Array, root["x"])
  mean, std = ds_statistics(arr, step=16)

  assert mean.shape == (6,)
  assert np.allclose(mean, data.mean(axis=0), atol=1e-4)
  assert np.allclose(std, data.std(axis=0), atol=1e-4)


def test_ds_statistics_zarr_auto_step(tmp_path):
  """step=None on zarr should use chunks[0]."""
  rng = np.random.default_rng(7)
  data = rng.standard_normal((60, 4)).astype(np.float32)
  p = tmp_path / "data.zarr"
  root = cast(zarr.Group, zarr.open(str(p), mode="w"))
  root.create_array("x", data=data, chunks=(20, 4))

  arr = cast(zarr.Array, root["x"])
  mean, std = ds_statistics(arr)

  assert np.allclose(mean, data.mean(axis=0), atol=1e-4)
  assert np.allclose(std, data.std(axis=0), atol=1e-4)


# ---------------------------------------------------------------------------
# ds_statistics: result matches Welford directly
# ---------------------------------------------------------------------------

def test_ds_statistics_matches_welford(tmp_path):
  """ds_statistics result should equal running Welford manually."""
  rng = np.random.default_rng(8)
  data = rng.standard_normal((120, 5)).astype(np.float64)
  step = 30

  # ground truth via Welford
  w = Welford(5, dtype=np.float64)
  for i in range(0, 120, step):
    w.update_batch(data[i:i + step])

  p = tmp_path / "data.h5"
  with h5py.File(p, "w") as f:
    f.create_dataset("x", data=data, chunks=(step, 5))

  with h5py.File(p, "r") as f:
    mean, std = ds_statistics(cast(h5py.Dataset, f["x"]), step=step)

  assert np.allclose(mean, w.mean)
  assert np.allclose(std, w.std)
