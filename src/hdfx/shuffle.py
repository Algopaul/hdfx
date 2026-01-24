from typing import cast

import h5py
import numpy as np

from hdfx.base import default_fields


def block_shuffle(infile, outfile, block_size, seed):
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
            f,
            shape=obj.shape,
            dtype=obj.dtype,
            chunks=obj.chunks,
            compression=obj.compression,
        )
        for out_b in range(n_full):
          in_b = perm[out_b]
          i = in_b * block_size
          j = out_b * block_size

          for f in fields:
            src = cast(h5py.Dataset, f_in[f])
            dst = cast(h5py.Dataset, f_out[f])
            dst[j:j + block_size] = src[i:i + block_size]

        # copy tail block unchanged
        if has_tail:
          tail_start = n_full * block_size
          for f in fields:
            src = cast(h5py.Dataset, f_in[f])
            dst = cast(h5py.Dataset, f_out[f])
            dst[tail_start:N] = src[tail_start:N]
