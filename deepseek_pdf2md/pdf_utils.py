"""Utilities for rasterising PDF documents to images."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_path: Path, dpi: int, logger: logging.Logger | None = None) -> List[Image.Image]:
    """Render each page in ``pdf_path`` to a PIL image at the given ``dpi``."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    images: List[Image.Image] = []
    doc = fitz.open(str(pdf_path))
    try:
        for index, page in enumerate(doc, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            images.append(image)
            if logger:
                logger.debug("Rendered page %s at %s dpi", index, dpi)
    finally:
        doc.close()

    if logger:
        logger.info("Rendered %s pages from %s", len(images), pdf_path)

    return images
