from glob import glob
from pathlib import Path
from typing import Annotated, List, Optional, cast

import h5py
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from hdfx.base import parse_slice
from hdfx.merge import h5merge
from hdfx.shard import h5shard
from hdfx.statistics import Welford

app = typer.Typer(help="Unix-style tools for working with HDF5")
console = Console()
err_console = Console(stderr=True)


def chunk_bytes(ds: h5py.Dataset):
  if ds.chunks is None:
    return None
  return np.prod(ds.chunks) * ds.dtype.itemsize


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="Input HDF5 file"),
    *,
    with_statistics: Annotated[
        bool,
        typer.Option("--with-statistics", help="Compute mean and std")] = False,
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
          step = int(chunks[0]) if chunks else 10
          w = Welford(obj.shape[-1] if len(obj.shape) > 1 else 1)
          for i in tqdm(range(0, obj.shape[0], step), desc=f'Stats for {name}'):
            l = min(i + step, obj.shape[0])
            w.update_batch(obj[i:l])
          col_args.extend([str(w.mean), str(w.std)])

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
    paths = []
    for pat in infiles:
      matches = glob(pat)
      if not matches:
        raise ValueError(f"No files match pattern: {pat}")
      paths.extend(matches)

    paths = [Path(p) for p in sorted(paths)]

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


def main():
  app()
