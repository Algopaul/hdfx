import sys
from pathlib import Path

import h5py
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Unix-style tools for working with HDF5")
console = Console()
err_console = Console(stderr=True)


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

  with h5py.File(path, "r") as f:

    def visit(name, obj):
      if isinstance(obj, h5py.Dataset):
        table.add_row(
            name,
            str(tuple(obj.shape)),
            str(obj.dtype),
        )

    f.visititems(visit)

  console.print(table)


@app.command()
def bee(path: Path):
  pass


def main():
  app()
