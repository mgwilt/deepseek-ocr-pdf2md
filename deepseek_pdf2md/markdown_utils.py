"""Markdown post-processing and sanitation helpers."""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from PIL import Image


DETECTION_PATTERN = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL)
DETECTION_CAPTURE_PATTERN = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|><\|det\|>(?P<coords>.*?)<\|/det\|>",
    re.DOTALL,
)

COORDINATE_BRACKET_PATTERN = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
REPEATED_WORD_PATTERN = re.compile(r"(\b\w+\b)(?:\s+\1\b){3,}", re.IGNORECASE)
REF_TAG_PATTERN = re.compile(r"<ref>.*?</ref>", re.IGNORECASE | re.DOTALL)

FIGURE_KEYWORDS = ("image", "figure", "chart", "diagram", "graph", "plot", "photo")

REPLACEMENTS = {
    "\\coloneqq": ":=",
    "\\eqqcolon": "=:",
    "<--- Page Split --->": "",
}


@dataclass
class PageContext:
    index: int
    image: Image.Image
    raw_text: str


class MarkdownBuilder:
    """Extracts figures and sanitises DeepSeek text into markdown."""

    def __init__(self, output_images_dir: Path, logger: logging.Logger | None = None) -> None:
        self.output_images_dir = output_images_dir
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.figure_count = 0
        self.logger = logger or logging.getLogger(__name__)

    def process_page(self, context: PageContext) -> str:
        page_filename = f"page-{context.index:03}.png"
        page_path = self.output_images_dir / page_filename
        context.image.save(page_path, format="PNG")
        self.logger.debug("Saved page image %s", page_path)

        section_header = f"<!-- Page {context.index} -->"
        image_note = f"<!-- Page image stored at images/{page_filename} -->"

        processed_text = self._replace_grounding_tokens(context)
        cleaned = sanitize_page_text(processed_text)

        if cleaned:
            body = cleaned
        else:
            body = "_No OCR output for this page._"
            self.logger.warning("No OCR text extracted for page %s", context.index)

        return f"{section_header}\n{image_note}\n\n{body}"

    def _replace_grounding_tokens(self, context: PageContext) -> str:
        figure_count = 0
        seen_boxes: set[Tuple[int, int, int, int]] = set()
        image_width, image_height = context.image.size

        def _replacement(match: re.Match[str]) -> str:
            nonlocal figure_count
            label = match.group("label") or ""
            coords_str = match.group("coords") or ""

            boxes = _parse_boxes(coords_str)
            if not boxes or not _label_is_visual(label):
                return ""

            pixel_boxes = _to_pixel_boxes(boxes, (image_width, image_height))
            if not pixel_boxes:
                return ""

            left = min(box[0] for box in pixel_boxes)
            top = min(box[1] for box in pixel_boxes)
            right = max(box[2] for box in pixel_boxes)
            bottom = max(box[3] for box in pixel_boxes)

            padding = max(4, int(0.01 * max(image_width, image_height)))
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image_width, right + padding)
            bottom = min(image_height, bottom + padding)

            box_key = (left, top, right, bottom)
            if box_key in seen_boxes:
                return ""
            seen_boxes.add(box_key)

            figure_count += 1
            global_figure_index = self.figure_count + figure_count
            filename = f"page-{context.index:03}-figure-{figure_count:02}.png"
            figure_path = self.output_images_dir / filename
            crop = context.image.crop((left, top, right, bottom))
            crop.save(figure_path, format="PNG")
            self.logger.debug("Saved figure image %s", figure_path)

            alt_label = label.strip() or f"Figure {global_figure_index}"
            if alt_label.lower() in {"image", "figure"}:
                alt_label = f"Figure {global_figure_index}"

            return f"\n\n![{alt_label}](images/{filename})\n\n"

        processed = DETECTION_CAPTURE_PATTERN.sub(_replacement, context.raw_text)
        self.figure_count += figure_count
        return processed


def sanitize_page_text(text: str) -> str:
    cleaned = text.replace("<｜end▁of▁sentence｜>", "")
    cleaned = DETECTION_PATTERN.sub("", cleaned)
    cleaned = REF_TAG_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"</?ref>", "", cleaned, flags=re.IGNORECASE)
    cleaned = COORDINATE_BRACKET_PATTERN.sub("", cleaned)
    cleaned = REPEATED_WORD_PATTERN.sub(lambda match: match.group(1), cleaned)
    cleaned = re.sub(r"\btext\b", "", cleaned, flags=re.IGNORECASE)

    for needle, replacement in REPLACEMENTS.items():
        cleaned = cleaned.replace(needle, replacement)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n(?:\s*\n)+", "\n\n", cleaned)
    return cleaned.strip()


def _label_is_visual(label: str) -> bool:
    normalized = label.strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in FIGURE_KEYWORDS)


def _parse_boxes(coords_str: str) -> list[tuple[float, float, float, float]]:
    try:
        parsed = ast.literal_eval(coords_str)
    except (SyntaxError, ValueError):
        return []

    boxes: list[tuple[float, float, float, float]] = []

    def _collect(candidate: object) -> None:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            if len(candidate) == 4 and all(isinstance(point, (int, float)) for point in candidate):
                x1, y1, x2, y2 = candidate  # type: ignore[assignment]
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
                return
            for entry in candidate:
                _collect(entry)

    _collect(parsed)
    return boxes


def _to_pixel_boxes(
    boxes: Sequence[tuple[float, float, float, float]],
    image_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    if width <= 0 or height <= 0:
        return []

    pixel_boxes: list[tuple[int, int, int, int]] = []

    def _convert(value: float, extent: int) -> int:
        scaled = int(round((value / 999.0) * extent))
        return max(0, min(extent, scaled))

    for box in boxes:
        x1, y1, x2, y2 = box
        px1, py1, px2, py2 = (
            _convert(x1, width),
            _convert(y1, height),
            _convert(x2, width),
            _convert(y2, height),
        )
        if px1 == px2 or py1 == py2:
            continue
        left, right = sorted((px1, px2))
        top, bottom = sorted((py1, py2))
        pixel_boxes.append((left, top, right, bottom))

    return pixel_boxes
