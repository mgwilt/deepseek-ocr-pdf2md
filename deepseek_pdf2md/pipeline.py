"""High-level orchestration for the DeepSeek OCR workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from PIL import Image

from .inference import generate_page_text
from .markdown_utils import MarkdownBuilder, PageContext
from .ocr_settings import ResolvedOCRSettings
from .pdf_utils import pdf_to_images


@dataclass
class PipelineArtifacts:
    markdown_path: Path
    page_count: int
    figure_count: int
    warnings: List[str] = field(default_factory=list)


def run_pipeline(settings: ResolvedOCRSettings, logger: logging.Logger) -> PipelineArtifacts:
    output_docs_dir = settings.output_docs_dir
    output_images_dir = settings.output_images_dir
    output_docs_dir.mkdir(parents=True, exist_ok=True)
    _clean_directory(output_images_dir, logger)

    logger.info("Processing PDF %s", settings.pdf_path)
    pdf_images: List[Image.Image] = pdf_to_images(settings.pdf_path, settings.pdf_dpi, logger)

    page_text = generate_page_text(pdf_images, settings, settings.prompt, logger)

    builder = MarkdownBuilder(output_images_dir, logger)

    markdown_sections: List[str] = []
    warnings: List[str] = []

    for index, (image, raw_text) in enumerate(zip(pdf_images, page_text), start=1):
        context = PageContext(index=index, image=image, raw_text=raw_text)
        section = builder.process_page(context)
        markdown_sections.append(section)
        if "_No OCR output" in section:
            warnings.append(f"No OCR output for page {index}")

    final_markdown = "\n\n".join(markdown_sections).strip() + "\n"

    markdown_path = settings.output_markdown
    markdown_path.write_text(final_markdown, encoding="utf-8")
    logger.info("Markdown written to %s", markdown_path)

    return PipelineArtifacts(
        markdown_path=markdown_path,
        page_count=len(markdown_sections),
        figure_count=builder.figure_count,
        warnings=warnings,
    )


def _clean_directory(directory: Path, logger: logging.Logger) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    removed = 0
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()
            removed += 1
    if removed:
        logger.debug("Removed %s stale files from %s", removed, directory)
