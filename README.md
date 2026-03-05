# hdfx

Unix-style tools for working with HDF5 (and Zarr) files.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Add `.venv/bin` to your `$PATH`, then optionally enable shell completions:

```bash
hdfx --install-completion
```

---

## Commands

### `inspect`

Print all datasets in an HDF5 or Zarr file with shape, dtype, and chunk info.

```bash
hdfx inspect data.h5
hdfx inspect data.zarr
hdfx inspect data.h5 --with-statistics          # compute mean and std per dataset
hdfx inspect data.h5 --with-statistics 1024     # use batch size 1024
```

---

### `merge`

Concatenate multiple HDF5 files along axis 0 into a single output file.

```bash
hdfx merge a.h5 b.h5 c.h5 out.h5
hdfx merge 'shards/*.h5' out.h5                 # glob patterns supported
hdfx merge a.h5 b.h5 out.h5 -f data -f labels  # select specific datasets
hdfx merge a.h5 b.h5 out.h5 --target-chunk-mb 16
hdfx merge a.h5 b.h5 out.h5 --chunk-rows 8192
hdfx merge a.h5 b.h5 out.h5 --add-dim           # append trailing singleton dim
```

---

### `shard`

Split one HDF5 file into N equal shards along axis 0.

```bash
hdfx shard data.h5 shards/out 8                 # → shards/out_000.h5 … out_007.h5
hdfx shard data.h5 shards/out 8 -f data -f time # select specific datasets
hdfx shard data.h5 shards/out 8 --target-chunk-mb 16
hdfx shard data.h5 shards/out 8 --chunk-rows 4096
hdfx shard data.h5 shards/out 8 --keep-remainder  # last shard gets leftover rows
```

---

### `stack`

Stack multiple HDF5 files along a new leading axis (axis 0).
Each input file contributes one entry: `(T, C)` × N files → `(N, T, C)`.

```bash
hdfx stack a.h5 b.h5 c.h5 stacked.h5
hdfx stack 'frames/*.h5' stacked.h5             # glob patterns supported
hdfx stack a.h5 b.h5 stacked.h5 --virtual       # zero-copy virtual dataset
hdfx stack a.h5 b.h5 stacked.h5 -f data -f meta
hdfx stack a.h5 b.h5 stacked.h5 --target-chunk-mb 8
```

---

### `shuffle`

Block-shuffle an HDF5 or Zarr file along axis 0 using a fixed random seed.
All datasets are permuted identically so correspondence is preserved.

```bash
hdfx shuffle in.h5 out.h5 1000 42       # block_size=1000, seed=42
hdfx shuffle in.zarr out.zarr 512 0
```

---

### `slice`

Extract a slice of one dataset into a new file.

```bash
hdfx slice data.h5 out.h5 /images '0:100'
hdfx slice data.h5 out.h5 /images '0:100,:,:'
hdfx slice data.h5 out.h5 /labels ':,0'
```

Slice syntax is comma-separated NumPy-style: `:` for full axis, `a:b` for range, `i` for index.

---

### `modify normalize`

Normalize a dataset in-place (subtract mean, divide by std).

```bash
hdfx modify normalize data.h5 /features
```

---

### `modify expand-dims`

Insert a new size-1 axis into a dataset in-place.

```bash
hdfx modify expand-dims data.h5 /labels           # inserts at axis 0 (default)
hdfx modify expand-dims data.h5 /labels --axis -1 # inserts at last position
```
