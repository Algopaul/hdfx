from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
from tqdm import tqdm

from hdfx.base import default_fields, get_chunk_rows


def h5merge(
    infiles: Sequence[str | Path],
    outfile: str | Path,
    *,
    chunk_rows: Optional[int] = None,
    target_chunk_mb: Optional[float] = 4.0,
    fields: Optional[Sequence[str]] = None,
    add_dim: bool = False,
) -> None:
  """
  Merge multiple HDF5 files along axis 0.

  All datasets must:
  - exist in all files
  - have identical trailing dimensions
  - have identical dtypes
  """
  infiles = [Path(f) for f in infiles]
  outfile = Path(outfile)

  if not infiles:
    raise ValueError("No input files provided")

  for f in infiles:
    if not f.exists():
      raise FileNotFoundError(f)

  # --- Pass 1: validate + collect shapes/dtypes ---
  shapes: dict[str, list[tuple]] = {}
  dtypes: dict[str, np.dtype] = {}

  with h5py.File(infiles[0], "r") as f0:
    if fields is None:
      fields = default_fields(f0)
    fields = list(fields)

    if not fields:
      raise ValueError("No datasets found (or fields list empty).")

    for field in fields:
      if field not in f0:
        raise KeyError(f"Missing dataset '{field}' in {infiles[0]}")
      obj = f0[field]
      if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"'{field}' is not a dataset in {infiles[0]}")
      dtypes[field] = obj.dtype
      shapes[field] = [tuple(obj.shape)]

  # Remaining files
  for path in infiles[1:]:
    with h5py.File(path, "r") as f:
      for field in fields:
        if field not in f:
          raise KeyError(f"Missing dataset '{field}' in {path}")
        obj = f[field]
        if not isinstance(obj, h5py.Dataset):
          raise TypeError(f"'{field}' is not a dataset in {path}")
        if obj.dtype != dtypes[field]:
          raise TypeError(
              f"Dtype mismatch for '{field}': {path} has {obj.dtype}, "
              f"expected {dtypes[field]}")
        shapes[field].append(tuple(obj.shape))

  # Validate trailing shapes and compute final shape
  final_shapes: dict[str, tuple] = {}
  for field in fields:
    ref = shapes[field][0][1:]
    total = 0
    for shp in shapes[field]:
      if shp[1:] != ref:
        raise ValueError(
            f"Shape mismatch for '{field}': got {shp[1:]}, expected {ref}")
      total += shp[0]

    if add_dim:
      final_shapes[field] = (total, *ref, 1)
    else:
      final_shapes[field] = (total, *ref)

  base_shapes = {f: final_shapes[f] for f in fields}
  chunk_rows = get_chunk_rows(chunk_rows, target_chunk_mb, base_shapes, dtypes)

  # --- Create output file ---
  with h5py.File(outfile, "w") as fout:
    for field in fields:
      out_shape = final_shapes[field]
      cr = min(chunk_rows, out_shape[0])
      chunks = (cr, *out_shape[1:])
      fout.create_dataset(
          field,
          shape=out_shape,
          dtype=dtypes[field],
          chunks=chunks,
      )

    # --- Streaming copy ---
    offsets = {field: 0 for field in fields}

    for i, path in enumerate(tqdm(infiles, desc="merge files")):
      with h5py.File(path, "r") as fin:
        for field in fields:
          src = fin[field]
          n = shapes[field][i][0]
          off = offsets[field]

          for j in range(0, n, chunk_rows):
            j2 = min(j + chunk_rows, n)
            fout[field][off + j:off + j2] = src[j:j2]

          offsets[field] += n
