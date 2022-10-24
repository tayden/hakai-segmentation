from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hakai_segmentation import lib

cli = typer.Typer()


@cli.command()
def list_versions():
    """List the available model versions"""
    versions = lib.list_versions()

    table = Table(title="Available Weights")

    table.add_column("Type", justify="left", style="magenta", no_wrap=True)
    table.add_column("Version ID", justify="left", style="green", min_width=32)
    table.add_column("Latest", justify="center", style="green")
    table.add_column("Created (UTC)", justify="right", style="green")
    table.add_column("Size (MB)", justify="right", style="green")

    for v in versions:
        table.add_row(
            v.model_type,
            v.version_id,
            "Y" if v.is_latest else "N",
            v.last_modified.strftime("%b %d, %Y, %H:%M"),
            str(v.size >> 20),
        )

    console = Console(width=100)
    console.print(table)


@cli.command()
def find_kelp(
    source: str = typer.Argument(..., help="Input image with Byte data type."),
    dest: str = typer.Argument(..., help="File path location to save output to."),
    species: bool = typer.Option(
        False,
        "--species/--presence",
        help="Segment to species or presence/absence level.",
    ),
    crop_size: int = typer.Option(
        256,
        help="The size for the cropped image squares run through the segmentation model.",
    ),
    padding: int = typer.Option(
        128, help="The number of context pixels added to each side of the image crops."
    ),
    batch_size: int = typer.Option(
        2, help="The batch size of cropped image sections to process together."
    ),
    use_gpu: bool = typer.Option(
        True, "--gpu/--no-gpu", help="Enable or disable GPU, if available."
    ),
    version: Optional[str] = typer.Option(
        None,
        help="String specifying the model version. "
        "See `kom list_versions` for available options."
        "By default, will use the latest model weights.",
    ),
):
    """Detect kelp in image at path SOURCE and output the resulting classification raster to file at path DEST."""
    lib.find_kelp(
        source=source,
        dest=dest,
        species=species,
        crop_size=crop_size,
        padding=padding,
        batch_size=batch_size,
        use_gpu=use_gpu,
        weights_version=version,
    )


@cli.command()
def find_mussels(
    source: str = typer.Argument(..., help="Input image with Byte data type."),
    dest: str = typer.Argument(..., help="File path location to save output to."),
    crop_size: int = typer.Option(
        256,
        help="The size for the cropped image squares run through the segmentation model.",
    ),
    padding: int = typer.Option(
        128, help="The number of context pixels added to each side of the image crops."
    ),
    batch_size: int = typer.Option(
        2, help="The batch size of cropped image sections to process together."
    ),
    use_gpu: bool = typer.Option(
        True, "--gpu/--no-gpu", help="Enable or disable GPU, if available."
    ),
    version: Optional[str] = typer.Option(
        None,
        help="String specifying the model version. "
        "See `kom list_versions` for available options."
        "By default, will use the latest model weights.",
    ),
):
    """Detect mussels in image at path SOURCE and output the resulting classification raster to file at path DEST."""
    lib.find_mussels(
        source=source,
        dest=dest,
        crop_size=crop_size,
        padding=padding,
        batch_size=batch_size,
        use_gpu=use_gpu,
        weights_version=version,
    )


if __name__ == "__main__":
    cli()
