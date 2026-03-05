from typing import cast

import h5py
import numpy as np
import zarr
from tqdm import tqdm

from hdfx.base import list_fields


def _shuffle_index(N, block_size, seed):
  n_full = N // block_size
  has_tail = N % block_size != 0
  rng = np.random.default_rng(seed)
  perm = rng.permutation(n_full)
  return perm, n_full, has_tail


def zarrshuffle(infile, outfile, block_size, seed):
  fields = list_fields(infile)
  root_in = cast(zarr.Group, zarr.open(str(infile), mode="r"))
  root_out = zarr.create_group(str(outfile), overwrite=True)

  first = cast(zarr.Array, root_in[fields[0]])
  N = first.shape[0]

  perm, n_full, has_tail = _shuffle_index(N, block_size, seed)

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
  fields = list_fields(infile)

  with h5py.File(infile, 'r') as f_in:
    N = cast(h5py.Dataset, f_in[fields[0]]).shape[0]
    perm, n_full, has_tail = _shuffle_index(N, block_size, seed)

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
