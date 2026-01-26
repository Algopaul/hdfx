from glob import glob
from pathlib import Path
from typing import Annotated, List, Optional, cast

import h5py
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from hdfx.base import iter_chunks, parse_slice, resolve_files
from hdfx.merge import h5merge, h5stack
from hdfx.shard import h5shard
from hdfx.shuffle import h5shuffle
from hdfx.statistics import ds_statistics

app = typer.Typer(help="Unix-style tools for working with HDF5")
modify = typer.Typer()

app.add_typer(modify, name="modify")
console = Console()
err_console = Console(stderr=True)


def chunk_bytes(ds: h5py.Dataset):
  if ds.chunks is None:
    return None
  return np.prod(ds.chunks) * ds.dtype.itemsize


@app.command()
def inspect(
    path: Annotated[
        Path,
        typer.Argument(help="Input HDF5 file"),
    ],
    stats_step: Annotated[
        int | None,
        typer.Argument(help="Batch size for statistics computation"),
    ] = None,
    with_statistics: Annotated[
        bool,
        typer.Option("--with-statistics", help="Compute mean and std"),
    ] = False,
):
  """
  Print all datasets in an HDF5 file with shape and dtype.
  """
  if not path.exists():
    err_console.print(f"File not found: {path}", style="bold red")
    raise typer.Exit(code=1)

  table = Table(title=str(path))
  table.add_column("Dataset", style="cyan", no_wrap=True)
  table.add_column("Shape")
  table.add_column("Dtype", style="magenta")
  table.add_column("Chunks")
  table.add_column("Chunk MB", justify="right")

  if with_statistics:
    table.add_column("mean", justify="right")
    table.add_column("std", justify="right")

  with h5py.File(path, "r") as f:

    def visit(name, obj):
      if isinstance(obj, h5py.Dataset):
        chunks = obj.chunks if obj.chunks is not None else "-"
        cb = chunk_bytes(obj)
        cb_str = "-" if cb is None else f"{cb/1024**2:.2f} MB"
        col_args = [
            name,
            str(tuple(obj.shape)),
            str(obj.dtype),
            str(chunks), cb_str
        ]
        if with_statistics:
          mean, std = ds_statistics(obj, stats_step)
          col_args.extend([str(mean), str(std)])

        table.add_row(*col_args)

    f.visititems(visit)

  console.print(table)


@app.command()
def slice(
    infile: Path = typer.Argument(..., help="Input HDF5 file"),
    outfile: Path = typer.Argument(..., help="Base name for output shards"),
    dataset: str = typer.Argument(..., help="Which dataset to slice"),
    slice: str = typer.Argument(..., help="How to slice the dataset"),
):
  """
  Slice a dataset according to numpy slice str description
  """
  with h5py.File(infile, "r") as fi, h5py.File(outfile, "w") as fo:
    d = cast(h5py.Dataset, fi[dataset])
    data = d[parse_slice(slice)]
    out = fo.create_dataset(dataset, data=data, chunks=True)
    for k, v in d.attrs.items():
      out.attrs[k] = v


@app.command()
def shuffle(
    infile: Path = typer.Argument(..., help="Input HDF5 file"),
    outfile: Path = typer.Argument(..., help="Outpuf shuffled file"),
    block_size: int = typer.Argument(..., help="Block size"),
    seed: int = typer.Argument(default=0, help="Random seed to use"),
):
  Path(outfile).parent.mkdir(parents=True, exist_ok=True)
  h5shuffle(infile, outfile, block_size, seed)


@app.command()
def shard(
    infile: Path = typer.Argument(..., help="Input HDF5 file"),
    outfile_base: str = typer.Argument(..., help="Base name for output shards"),
    n_outfiles: int = typer.Argument(..., help="Number of shard files"),
    fields: Optional[List[str]] = typer.Option(
        None,
        "--field",
        "-f",
        help="Dataset to include (repeatable). Default: all datasets"),
    chunk_rows: Optional[int] = typer.Option(
        None,
        "--chunk-rows",
        help="Explicit number of rows per HDF5 chunk (expert mode)"),
    target_chunk_mb: Optional[float] = typer.Option(
        None, "--target-chunk-mb", help="Target chunk size in MB (auto mode)"),
    keep_remainder: bool = typer.Option(
        False,
        "--keep-remainder",
        help="Keep remainder samples in the last shard"),
):
  """
    Split an HDF5 file into multiple shard files along axis 0.

    Examples:

      hdfx shard data.h5 shards/out 64
      hdfx shard data.h5 shards/out 64 --target-chunk-mb 16
      hdfx shard data.h5 shards/out 64 -f data -f time
      hdfx shard data.h5 shards/out 64 --chunk-rows 8192
    """
  h5shard(
      infile=infile,
      outfile_base=outfile_base,
      n_outfiles=n_outfiles,
      chunk_rows=chunk_rows,
      target_chunk_mb=target_chunk_mb,
      fields=fields,
      drop_remainder=not keep_remainder,
  )


@app.command()
def stack(
    infiles: List[str] = typer.Argument(..., help="Input HDF5 files or globs"),
    outfile: Path = typer.Argument(..., help="Output stacked HDF5 file"),
    fields: Optional[List[str]] = typer.Option(
        None,
        "--field",
        "-f",
        help="Dataset to include (repeatable). Default: all datasets"),
    chunk_rows: Optional[int] = typer.Option(
        None,
        "--chunk-rows",
        help="Explicit number of rows per HDF5 chunk (expert mode)"),
    target_chunk_mb: Optional[float] = typer.Option(
        None, "--target-chunk-mb", help="Target chunk size in MB (auto mode)"),
    virtual: bool = typer.Option(
        False, "--virtual", help="Use a virtual dataset (no copying)"),
):
  """
    Merge multiple HDF5 files along axis 0.
    """
  try:
    paths = resolve_files(infiles)
    h5stack(
        infiles=paths,
        outfile=outfile,
        chunk_rows=chunk_rows,
        target_chunk_mb=target_chunk_mb,
        fields=fields,
        virtual=virtual,
    )
  except Exception as e:
    err_console.print(f"[red]Error:[/red] {e}")
    raise typer.Exit(code=1)


@app.command()
def merge(
    infiles: List[str] = typer.Argument(..., help="Input HDF5 files or globs"),
    outfile: Path = typer.Argument(..., help="Output merged HDF5 file"),
    fields: Optional[List[str]] = typer.Option(
        None,
        "--field",
        "-f",
        help="Dataset to include (repeatable). Default: all datasets"),
    chunk_rows: Optional[int] = typer.Option(
        None,
        "--chunk-rows",
        help="Explicit number of rows per HDF5 chunk (expert mode)"),
    target_chunk_mb: Optional[float] = typer.Option(
        None, "--target-chunk-mb", help="Target chunk size in MB (auto mode)"),
    add_dim: bool = typer.Option(
        False, "--add-dim", help="Append a trailing singleton dimension"),
):
  """
    Merge multiple HDF5 files along axis 0.
    """
  try:
    paths = resolve_files(infiles)

    h5merge(
        infiles=paths,
        outfile=outfile,
        chunk_rows=chunk_rows,
        target_chunk_mb=target_chunk_mb,
        fields=fields,
        add_dim=add_dim,
    )
  except Exception as e:
    err_console.print(f"[red]Error:[/red] {e}")
    raise typer.Exit(code=1)


@modify.command()
def normalize(
    infile: str,
    field: str,
):
  with h5py.File(infile, 'a') as f:
    obj = cast(h5py.Dataset, f[field])
    mean, std = ds_statistics(obj)
    for chunk in tqdm(iter_chunks(obj)):
      obj[chunk] = (obj[chunk] - mean) / std
  pass


@modify.command()
def expand_dims(
    infile: Annotated[
        str,
        typer.Argument(help="HDF5 file to modify in-place"),
    ],
    field: Annotated[
        str,
        typer.Argument(help="Dataset path inside the file"),
    ],
    axis: Annotated[
        int,
        typer
        .Option("--axis", help="Axis at which to insert the new dimension"),
    ] = 0,
):
  with h5py.File(infile, "r+") as f:
    src = cast(h5py.Dataset, f[field])
    old_shape = src.shape
    old_rank = len(old_shape)
    new_rank = old_rank + 1
    axis_norm = axis % new_rank
    new_shape = (old_shape[:axis_norm] + (1,) + old_shape[axis_norm:])
    if src.chunks is None:
      new_chunks = None
    else:
      new_chunks = (src.chunks[:axis_norm] + (1,) + src.chunks[axis_norm:])

    tmp_name = field + "__tmp"
    tmp = f.create_dataset(
        tmp_name,
        shape=new_shape,
        dtype=src.dtype,
        chunks=new_chunks,
        compression=src.compression,
        compression_opts=src.compression_opts,
        shuffle=src.shuffle,
        fletcher32=src.fletcher32,
    )

    for slc in tqdm(iter_chunks(src)):
      new_slc = (slc[:axis_norm] + (0,) + slc[axis_norm:])
      tmp[new_slc] = src[slc]

    for k, v in src.attrs.items():
      tmp.attrs[k] = v

    del f[field]
    f.move(tmp_name, field)


def main():
  app()
