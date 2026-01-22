from typing import Optional

import h5py
import numpy as np


def parse_slice(s):

  def part(p):
    if p == ":":
      return slice(None)
    if ':' not in p:
      return int(p)
    a = [int(x) if x else None for x in p.split(":")]
    return slice(*a)

  return tuple(part(p) for p in s.split(","))


def get_chunk_rows(
    chunk_rows: Optional[int],
    target_chunk_mb: Optional[float],
    shapes: dict[str, tuple],
    dtypes: dict[str, np.dtype],
):
  if chunk_rows is not None and target_chunk_mb is not None:
    raise ValueError("Specify either chunk_rows or target_chunk_mb, not both.")

  if chunk_rows is None:
    if target_chunk_mb is None:
      target_chunk_mb = 4.0
    chunk_rows = auto_chunk_rows_multi(shapes, dtypes, target_chunk_mb)

  if chunk_rows <= 0:
    raise ValueError("chunk_rows must be > 0")

  return chunk_rows


def default_fields(f: h5py.File) -> list[str]:
  return [k for k, v in f.items() if isinstance(v, h5py.Dataset)]


def auto_chunk_rows_multi(shapes: dict[str, tuple], dtypes: dict[str, np.dtype],
                          target_mb: float) -> int:
  """
    Choose chunk_rows so the largest field produces ~target_mb sized chunks.
    """
  max_bytes_per_row = 0
  for f in shapes:
    bytes_per_row = np.prod(shapes[f][1:]) * np.dtype(dtypes[f]).itemsize
    max_bytes_per_row = max(max_bytes_per_row, bytes_per_row)

  if max_bytes_per_row == 0:
    return 1

  rows = int(target_mb * 1024**2 / max_bytes_per_row)
  return max(1, rows)
