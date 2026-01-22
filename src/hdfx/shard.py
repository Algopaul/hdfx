import logging
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np
from tqdm import tqdm

from hdfx.base import auto_chunk_rows_multi, default_fields, get_chunk_rows


def h5shard(
    infile: str | Path,
    outfile_base: str | Path,
    n_outfiles: int,
    *,
    chunk_rows: Optional[int] = None,
    target_chunk_mb: Optional[float] = None,
    fields: Optional[Sequence[str]] = None,
    drop_remainder: bool = True,
) -> None:
  """
    Shard datasets from one HDF5 file into N output files along axis 0.

    - Streams data in blocks to keep memory bounded.
    - Validates fields, shapes, and dtypes.
    - Uses max-field auto chunking unless chunk_rows is given.

    Output files: f"{outfile_base}_{i:03d}.h5"
    """
  infile = Path(infile)
  outfile_base = str(outfile_base)

  if n_outfiles <= 0:
    raise ValueError("n_outfiles must be > 0")

  if not infile.exists():
    raise FileNotFoundError(infile)

  with h5py.File(infile, "r") as fin:
    if fields is None:
      fields = default_fields(fin)
    fields = list(fields)

    if not fields:
      raise ValueError("No datasets found (or fields list empty).")

    dtypes: dict[str, np.dtype] = {}
    shapes: dict[str, tuple] = {}

    for field in fields:
      if field not in fin:
        raise KeyError(f"Missing dataset '{field}' in {infile}")
      obj = fin[field]
      if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"'{field}' is not a dataset in {infile}")
      dtypes[field] = obj.dtype
      shapes[field] = tuple(obj.shape)

    N = shapes[fields[0]][0]
    for field in fields[1:]:
      if shapes[field][0] != N:
        raise ValueError(
            f"Leading dimension mismatch: '{fields[0]}' has N={N}, "
            f"but '{field}' has N={shapes[field][0]}")

    n_per_file = N // n_outfiles
    remainder = N - n_per_file * n_outfiles

    if remainder != 0:
      msg = (f"{infile}: N={N} not divisible by n_outfiles={n_outfiles}; "
             f"n_per_file={n_per_file}, remainder={remainder}.")
      if drop_remainder:
        logging.warning("%s Dropping remainder.", msg)
      else:
        logging.warning("%s Keeping remainder (last shard larger).", msg)

    if n_per_file == 0:
      raise ValueError(
          f"n_outfiles={n_outfiles} is too large for N={N} (n_per_file=0).")

    chunk_rows = get_chunk_rows(chunk_rows, target_chunk_mb, shapes, dtypes)

    for i in tqdm(range(n_outfiles), desc="shard files"):
      start = i * n_per_file
      end = start + n_per_file

      # Optionally keep remainder in the last shard
      if (not drop_remainder) and (i == n_outfiles - 1):
        end = N

      out_path = f"{outfile_base}_{i:03d}.h5"
      Path(out_path).parent.mkdir(exist_ok=True, parents=True)

      with h5py.File(out_path, "w") as fout:
        # Create datasets
        for field in fields:
          full_shape = shapes[field]
          out_shape = (end - start, *full_shape[1:])
          cr = min(chunk_rows, out_shape[0])
          chunks = (cr, *out_shape[1:]) if out_shape[0] > 0 else None

          fout.create_dataset(
              field,
              shape=out_shape,
              dtype=dtypes[field],
              chunks=chunks,
          )

        for field in fields:
          src = fin[field]
          dst = fout[field]
          n_rows = end - start

          for j in range(0, n_rows, chunk_rows):
            j2 = min(j + chunk_rows, n_rows)
            dst[j:j2] = src[start + j:start + j2]
