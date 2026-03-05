from typing import cast

import h5py
import numpy as np
import pytest
import zarr

from hdfx.base import iter_chunks, list_fields, open_dataset, parse_slice

# ---------------------------------------------------------------------------
# list_fields
# ---------------------------------------------------------------------------


def test_list_fields_h5_flat(tmp_path):
  p = tmp_path / "flat.h5"
  with h5py.File(p, "w") as f:
    f.create_dataset("a", data=np.zeros(10))
    f.create_dataset("b", data=np.ones((5, 3)))
  fields = list_fields(p)
  assert sorted(fields) == ["a", "b"]


def test_list_fields_h5_nested(tmp_path):
  p = tmp_path / "nested.h5"
  with h5py.File(p, "w") as f:
    grp = f.create_group("g")
    grp.create_dataset("x", data=np.zeros(4))
    f.create_dataset("y", data=np.zeros(4))
  fields = list_fields(p)
  assert sorted(fields) == ["g/x", "y"]


def test_list_fields_zarr(tmp_path):
  p = tmp_path / "store.zarr"
  root = cast(zarr.Group, zarr.open(str(p), mode="w"))
  root.create_array("a", shape=(10,), dtype="f4")
  root.create_array("b", shape=(5, 3), dtype="f4")
  fields = list_fields(p)
  assert sorted(fields) == ["a", "b"]


def test_list_fields_unsupported(tmp_path):
  p = tmp_path / "data.npy"
  p.touch()
  with pytest.raises(ValueError, match="Unsupported"):
    list_fields(p)


# ---------------------------------------------------------------------------
# open_dataset
# ---------------------------------------------------------------------------


def test_open_dataset_h5(tmp_path):
  p = tmp_path / "data.h5"
  arr = np.arange(20, dtype=np.float32)
  with h5py.File(p, "w") as f:
    f.create_dataset("x", data=arr)

  with open_dataset(p, "x") as ds:
    result = np.asarray(ds[:])
    h5ds = cast(h5py.Dataset, ds)
    file_id = h5ds.file.id

  np.testing.assert_array_equal(result, arr)
  assert not file_id.valid, "file should be closed after context exit"


def test_open_dataset_zarr(tmp_path):
  p = tmp_path / "data.zarr"
  arr = np.arange(20, dtype=np.float32)
  root = cast(zarr.Group, zarr.open(str(p), mode="w"))
  root["x"] = arr  # type: ignore[index]  # zarr.Group.__setitem__ missing from stubs

  with open_dataset(p, "x") as ds:
    result = np.asarray(ds[:])

  np.testing.assert_array_equal(result, arr)


def test_open_dataset_unsupported(tmp_path):
  p = tmp_path / "data.csv"
  p.touch()
  with pytest.raises(ValueError, match="either .zarr or .h5"):
    with open_dataset(p, "x"):
      pass


# ---------------------------------------------------------------------------
# parse_slice
# ---------------------------------------------------------------------------


def test_parse_slice():
  assert parse_slice(":") == (slice(None),)
  assert parse_slice("0:10") == (slice(0, 10),)
  assert parse_slice("5") == (5,)
  assert parse_slice(":,0:5") == (slice(None), slice(0, 5))
  assert parse_slice("1:10:2") == (slice(1, 10, 2),)


# ---------------------------------------------------------------------------
# iter_chunks
# ---------------------------------------------------------------------------


def test_iter_chunks(tmp_path):
  p = tmp_path / "chunks.h5"
  data = np.arange(24, dtype=np.int32).reshape(4, 6)

  with h5py.File(p, "w") as f:
    f.create_dataset("d", data=data, chunks=(2, 3))

  with h5py.File(p, "r") as f:
    ds = cast(h5py.Dataset, f["d"])
    covered = np.zeros_like(data, dtype=bool)
    for sel in iter_chunks(ds):
      covered[sel] = True

  assert covered.all(), "iter_chunks should cover every element"
