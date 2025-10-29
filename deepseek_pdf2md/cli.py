"""CLI entrypoint for the DeepSeek OCR pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from .ocr_settings import load_settings
from .pipeline import run_pipeline


app = typer.Typer(help="Convert PDFs to Markdown using DeepSeek-OCR")


def _configure_logging(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=log_format)
    return logging.getLogger("deepseek_ocr")


@app.command()
def convert(
    settings: Path = typer.Option(
        Path("ocr_settings.toml"),
        "--settings",
        "-s",
        help="Path to the TOML settings file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run the DeepSeek OCR pipeline using the provided settings."""

    root = Path(__file__).resolve().parent.parent
    logger = _configure_logging(verbose)

    resolved_settings = load_settings(root=root, settings_file=settings)
    logger.info("Loaded settings from %s", settings)

    artifacts = run_pipeline(resolved_settings, logger.getChild("pipeline"))

    typer.echo(f"Markdown written to {artifacts.markdown_path}")
    typer.echo(f"Processed {artifacts.page_count} pages; extracted {artifacts.figure_count} figures.")
    for warning in artifacts.warnings:
        typer.secho(f"Warning: {warning}", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
