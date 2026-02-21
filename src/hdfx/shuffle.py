from pathlib import Path
from typing import List, cast

import h5py
import numpy as np
import zarr
from tqdm import tqdm

from hdfx.base import default_fields


def getdata(filename, field):
  if str(filename)[-len('.zarr'):] == '.zarr':
    f = zarr.open(filename, mode='r')
    return f[field]
  elif str(filename)[-len('.h5'):] == '.h5':
    f = h5py.File(filename, 'r')
    return f[field]
  else:
    raise ValueError('either .zarr or .h5')


def _zarr_datasets(group, prefix="") -> List[str]:
  fields = []
  for key in group.keys():
    path = f"{prefix}/{key}" if prefix else key
    if isinstance(group[key], zarr.Array):
      fields.append(path)
    elif isinstance(group[key], zarr.Group):
      fields.extend(_zarr_datasets(group[key], path))
  return fields


def _h5_datasets(group, prefix="") -> List[str]:
  fields = []
  for key, value in group.items():
    path = f"{prefix}/{key}" if prefix else key
    if isinstance(value, h5py.Dataset):
      fields.append(path)
    elif isinstance(value, h5py.Group):
      fields.extend(_h5_datasets(value, path))
  return fields


def get_fields(filename) -> List[str]:
  filename = Path(filename)

  if filename.suffix == ".zarr":
    f = zarr.open(str(filename), mode="r")
    return _zarr_datasets(f)

  elif filename.suffix == ".h5":
    with h5py.File(filename, "r") as f:
      return _h5_datasets(f)

  else:
    raise ValueError("either .zarr or .h5")


def zarrshuffle(infile, outfile, block_size, seed):
  root_in = cast(zarr.Group, zarr.open(str(infile), mode="r"))
  root_out = zarr.create_group(
      str(outfile),
      overwrite=True,
  )

  fields = get_fields(infile)

  # assume first axis is sample axis
  first = cast(zarr.Array, root_in[fields[0]])
  N = first.shape[0]

  has_tail = N % block_size != 0
  n_full = N // block_size

  rng = np.random.default_rng(seed)
  perm = rng.permutation(n_full)

  for f in fields:
    src = cast(zarr.Array, root_in[f])

    dst = root_out.create_array(
        name=f,
        shape=src.shape,
        dtype=src.dtype,
        chunks=src.chunks,
        compressors=src.compressors,
    )

    for out_b in tqdm(range(n_full), desc=f"Shuffling {f}"):
      in_b = perm[out_b]
      i = in_b * block_size
      j = out_b * block_size
      dst[j:j + block_size] = src[i:i + block_size]

    if has_tail:
      tail_start = n_full * block_size
      dst[tail_start:N] = src[tail_start:N]


def h5shuffle(infile, outfile, block_size, seed):
  with h5py.File(infile, 'r') as f_in:
    fields = default_fields(f_in)
    N = cast(h5py.Dataset, f_in[fields[0]]).shape[0]
    has_tail = N % block_size != 0
    n_full = N // block_size

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_full)

    with h5py.File(outfile, 'w') as f_out:
      for f in fields:
        obj = cast(h5py.Dataset, f_in[f])
        f_out.create_dataset(
            name=f,
            shape=obj.shape,
            dtype=obj.dtype,
            chunks=obj.chunks,
        )
        src = cast(h5py.Dataset, f_in[f])
        dst = cast(h5py.Dataset, f_out[f])
        for out_b in tqdm(range(n_full), desc=f'Shuffling {f}'):
          in_b = perm[out_b]
          i = in_b * block_size
          j = out_b * block_size
          dst[j:j + block_size] = src[i:i + block_size]

        if has_tail:
          tail_start = n_full * block_size
          dst[tail_start:N] = src[tail_start:N]
