from glob import glob
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from hdfx.merge import h5merge
from hdfx.shard import h5shard

app = typer.Typer(help="Unix-style tools for working with HDF5")
console = Console()
err_console = Console(stderr=True)


def chunk_bytes(ds: h5py.Dataset):
  if ds.chunks is None:
    return None
  return np.prod(ds.chunks) * ds.dtype.itemsize


@app.command()
def inspect(path: Path):
  """
  Print all datasets in an HDF5 file with shape and dtype.
  """
  if not path.exists():
    err_console.print(f"[red]File not found:[/red] {path}", style="bold red")
    raise typer.Exit(code=1)

  table = Table(title=str(path))
  table.add_column("Dataset", style="cyan", no_wrap=True)
  table.add_column("Shape")
  table.add_column("Dtype", style="magenta")
  table.add_column("Chunks")
  table.add_column("Chunk MB", justify="right")

  with h5py.File(path, "r") as f:

    def visit(name, obj):
      if isinstance(obj, h5py.Dataset):
        chunks = obj.chunks if obj.chunks is not None else "-"
        cb = chunk_bytes(obj)
        cb_str = "-" if cb is None else f"{cb/1024**2:.2f} MB"
        table.add_row(
            name,
            str(tuple(obj.shape)),
            str(obj.dtype),
            str(chunks),
            cb_str,
        )

    f.visititems(visit)

  console.print(table)


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
