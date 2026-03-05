from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
import zarr

from hdfx.merge import h5merge, h5stack
from hdfx.shard import h5shard
from hdfx.shuffle import h5shuffle, zarrshuffle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_h5(path: Path, **arrays) -> Path:
  with h5py.File(path, "w") as f:
    for name, arr in arrays.items():
      f.create_dataset(name, data=arr, chunks=True)
  return path


def make_zarr(path: Path, **arrays) -> Path:
  root = cast(zarr.Group, zarr.open(str(path), mode="w"))
  for name, arr in arrays.items():
    root[name] = arr  # type: ignore[index]  # zarr.Group.__setitem__ missing from stubs
  return path


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_basic(tmp_path):
  a = np.arange(20, dtype=np.float32).reshape(10, 2)
  b = np.arange(20, 40, dtype=np.float32).reshape(10, 2)
  f1 = make_h5(tmp_path / "in1.h5", data=a)
  f2 = make_h5(tmp_path / "in2.h5", data=b)
  out = tmp_path / "out.h5"

  h5merge([f1, f2], out, fields=["data"])

  with h5py.File(out, "r") as f:
    result = cast(h5py.Dataset, f["data"])[:]

  assert result.shape == (20, 2)
  np.testing.assert_array_equal(result[:10], a)
  np.testing.assert_array_equal(result[10:], b)


def test_merge_dtype_mismatch(tmp_path):
  a = np.zeros((5, 2), dtype=np.float32)
  b = np.zeros((5, 2), dtype=np.float64)
  f1 = make_h5(tmp_path / "in1.h5", data=a)
  f2 = make_h5(tmp_path / "in2.h5", data=b)
  out = tmp_path / "out.h5"

  with pytest.raises((TypeError, ValueError)):
    h5merge([f1, f2], out, fields=["data"])


# ---------------------------------------------------------------------------
# shard
# ---------------------------------------------------------------------------

def test_shard_basic(tmp_path):
  data = np.arange(50, dtype=np.float32).reshape(50, 1)
  src = make_h5(tmp_path / "src.h5", data=data)
  base = str(tmp_path / "shard")

  h5shard(src, base, n_outfiles=5, chunk_rows=10)

  for i in range(5):
    p = Path(f"{base}_{i:03d}.h5")
    assert p.exists()
    with h5py.File(p, "r") as f:
      assert cast(h5py.Dataset, f["data"]).shape[0] == 10


def test_shard_keep_remainder(tmp_path):
  data = np.arange(53, dtype=np.float32).reshape(53, 1)
  src = make_h5(tmp_path / "src.h5", data=data)
  base = str(tmp_path / "shard")

  h5shard(src, base, n_outfiles=5, chunk_rows=10, drop_remainder=False)

  sizes = []
  for i in range(5):
    p = Path(f"{base}_{i:03d}.h5")
    with h5py.File(p, "r") as f:
      sizes.append(cast(h5py.Dataset, f["data"]).shape[0])

  assert sizes[-1] == 13   # 50 + remainder 3 = 13
  assert all(s == 10 for s in sizes[:-1])


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------

def test_stack_physical(tmp_path):
  files = []
  T, C = 8, 3
  for i in range(4):
    arr = np.full((T, C), i, dtype=np.float32)
    files.append(make_h5(tmp_path / f"in{i}.h5", data=arr))
  out = tmp_path / "stacked.h5"

  h5stack(files, out, virtual=False, chunk_rows=1, target_chunk_mb=None)

  with h5py.File(out, "r") as f:
    result = cast(h5py.Dataset, f["data"])[:]

  assert result.shape == (4, T, C)
  for i in range(4):
    np.testing.assert_array_equal(result[i], np.full((T, C), i))


def test_stack_virtual(tmp_path):
  files = []
  T, C = 8, 3
  for i in range(4):
    arr = np.full((T, C), i, dtype=np.float32)
    files.append(make_h5(tmp_path / f"in{i}.h5", data=arr))
  out = tmp_path / "virtual.h5"

  h5stack(files, out, virtual=True)

  with h5py.File(out, "r") as f:
    ds = cast(h5py.Dataset, f["data"])
    assert ds.is_virtual
    assert ds.shape == (4, T, C)


# ---------------------------------------------------------------------------
# shuffle
# ---------------------------------------------------------------------------

def test_shuffle_h5_same_rows(tmp_path):
  rng = np.random.default_rng(0)
  data = rng.integers(0, 100, size=(50, 4)).astype(np.float32)
  src = make_h5(tmp_path / "src.h5", data=data)
  dst = tmp_path / "dst.h5"

  h5shuffle(src, dst, block_size=10, seed=42)

  with h5py.File(dst, "r") as f:
    result = cast(h5py.Dataset, f["data"])[:]

  assert result.shape == data.shape
  src_rows = set(map(tuple, data.tolist()))
  dst_rows = set(map(tuple, result.tolist()))
  assert src_rows == dst_rows


def test_shuffle_h5_deterministic(tmp_path):
  rng = np.random.default_rng(1)
  data = rng.random((40, 3)).astype(np.float32)
  src = make_h5(tmp_path / "src.h5", data=data)

  dst1 = tmp_path / "out1.h5"
  dst2 = tmp_path / "out2.h5"
  h5shuffle(src, dst1, block_size=10, seed=7)
  h5shuffle(src, dst2, block_size=10, seed=7)

  with h5py.File(dst1, "r") as f1, h5py.File(dst2, "r") as f2:
    np.testing.assert_array_equal(
        cast(h5py.Dataset, f1["data"])[:],
        cast(h5py.Dataset, f2["data"])[:],
    )


def test_shuffle_zarr_same_rows(tmp_path):
  rng = np.random.default_rng(2)
  data = rng.integers(0, 100, size=(40, 5)).astype(np.float32)
  src = make_zarr(tmp_path / "src.zarr", data=data)
  dst = tmp_path / "dst.zarr"

  zarrshuffle(src, dst, block_size=10, seed=99)

  root = cast(zarr.Group, zarr.open(str(dst), mode="r"))
  result: np.ndarray = np.asarray(cast(zarr.Array, root["data"]))

  assert result.shape == data.shape
  src_rows = set(map(tuple, data.tolist()))
  dst_rows = set(map(tuple, result.tolist()))
  assert src_rows == dst_rows


# ---------------------------------------------------------------------------
# expand_dims
# ---------------------------------------------------------------------------

def _run_expand_dims(infile, field, axis):
  from typer.testing import CliRunner
  from hdfx.cli import app
  runner = CliRunner()
  result = runner.invoke(app, ["modify", "expand-dims", str(infile), field, "--axis", str(axis)])
  assert result.exit_code == 0, result.output


def test_expand_dims_h5(tmp_path):
  data = np.arange(24, dtype=np.float32).reshape(4, 6)
  p = make_h5(tmp_path / "data.h5", x=data)

  _run_expand_dims(p, "x", 1)

  with h5py.File(p, "r") as f:
    result = cast(h5py.Dataset, f["x"])[:]

  assert result.shape == (4, 1, 6)
  np.testing.assert_array_equal(result[:, 0, :], data)


def test_expand_dims_h5_negative_axis(tmp_path):
  data = np.arange(12, dtype=np.float32).reshape(3, 4)
  p = make_h5(tmp_path / "data.h5", x=data)

  _run_expand_dims(p, "x", -1)

  with h5py.File(p, "r") as f:
    result = cast(h5py.Dataset, f["x"])[:]

  assert result.shape == (3, 4, 1)


def test_expand_dims_zarr(tmp_path):
  data = np.arange(24, dtype=np.float32).reshape(4, 6)
  p = make_zarr(tmp_path / "data.zarr", x=data)

  _run_expand_dims(p, "x", 1)

  root = cast(zarr.Group, zarr.open(str(p), mode="r"))
  result: np.ndarray = np.asarray(cast(zarr.Array, root["x"]))

  assert result.shape == (4, 1, 6)
  np.testing.assert_array_equal(result[:, 0, :], data)


def test_expand_dims_zarr_axis0(tmp_path):
  data = np.arange(20, dtype=np.float32).reshape(5, 4)
  p = make_zarr(tmp_path / "data.zarr", x=data)

  _run_expand_dims(p, "x", 0)

  root = cast(zarr.Group, zarr.open(str(p), mode="r"))
  result: np.ndarray = np.asarray(cast(zarr.Array, root["x"]))

  assert result.shape == (1, 5, 4)
  np.testing.assert_array_equal(result[0], data)
