# hdfx — improvement todo

Ordered roughly by impact.

---

## Bugs

- [x] **`getdata` leaks file handles** (`shuffle.py:12-20`)
  Opens `zarr` and `h5py.File` without context managers and returns the raw
  array. The file is never closed. Function is also dead code — neither
  `zarrshuffle` nor `h5shuffle` calls it. Remove it or fix + use it.

- [x] **Restore error handling in `shuffle` CLI** (`cli.py:137-138`)
  The `try/except` block was commented out. Re-enable it so errors surface
  cleanly instead of crashing with a traceback.

- [x] **`pytest` in runtime dependencies** (`pyproject.toml:15`)
  Move to `[project.optional-dependencies] dev`. Users installing the tool
  shouldn't pull in pytest.

---

## High-leverage features

- [ ] **Attribute preservation in `merge` / `stack` / `shard`**
  `slice` copies dataset attributes; the others don't. Any stored metadata
  (units, description, calibration coefficients) is silently dropped.
  Add a loop over `src.attrs` in each operation, mirroring what `slice` does.

- [x] **`inspect` support for Zarr files**
  The shuffle and statistics code already handles Zarr, but `inspect` only
  accepts `.h5`. Add a Zarr path through `visititems`-equivalent
  (`_zarr_datasets` already exists in `shuffle.py`) so users can inspect
  both formats with one command.

- [ ] **`merge --virtual` (VirtualDataset for concatenation)**
  `stack` has `--virtual`; `merge` doesn't. A virtual merge is free to create
  and very useful for large datasets you want to treat as one without copying.

---

## Code quality

- [x] **Unify `default_fields` / `get_fields` / `_h5_datasets`**
  `default_fields` (`base.py:92`) is flat (top-level only).
  `_h5_datasets` (`shuffle.py:34`) is recursive.
  `get_fields` (`shuffle.py:45`) dispatches by suffix.
  These three do overlapping jobs inconsistently. Consolidate into one
  `list_fields(path) -> list[str]` that handles both formats recursively,
  and use it everywhere.

- [x] **`zarrshuffle` / `h5shuffle` are near-identical**
  The permutation generation and tail-copy logic is duplicated verbatim.
  Extract a shared `_shuffle_index(N, block_size, seed)` helper that returns
  `(perm, has_tail, n_full)` and call it from both.

- [ ] **`libver="latest"` inconsistency**
  `stack` opens output with `libver="latest"`; `merge` and `shard` don't.
  Pick one and apply it consistently (or expose it as a flag).

---

## Tests

- [x] **Integration tests for `merge` and `shard`** (highest value — zero coverage now)
  Use `tmp_path` fixtures to create small HDF5 files, run the operation,
  and assert output shape, dtype, and values are correct.

- [x] **Integration tests for `stack`** including the `--virtual` path.

- [ ] **Test attribute preservation** once that is implemented.

- [x] **Test `shuffle`** — check that output contains exactly the same rows as
  input (just reordered) and that the seed makes it reproducible.

---

## Minor / polish

- [x] **README** — add a full command reference (shard, merge, stack, shuffle,
  slice, modify normalize/expand-dims). Right now only `inspect` is shown.

- [x] **`ds_statistics` fallback step** (`statistics.py:51`)
  Falls back to `step=10` when chunks are absent — likely too small for any
  real dataset. Use something like `max(10, ds.shape[0] // 100)` or require
  the caller to always pass a step.

- [x] **`normalize` missing error handling** (`cli.py:260-270`)
  Unlike the other modify commands it has no try/except and will crash raw
  on missing fields or type errors.
