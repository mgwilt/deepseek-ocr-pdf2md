from __future__ import annotations

import ast
import io
import os
import re
import sys
from collections.abc import Sequence as AbcSequence
from pathlib import Path
from typing import List, Sequence, Tuple

import fitz
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

ROOT = Path(__file__).resolve().parent
DEEPSEEK_VLLM_DIR = ROOT / "DeepSeek-OCR" / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"

os.environ["VLLM_USE_V1"] = "1"
Image.MAX_IMAGE_PIXELS = None

if str(DEEPSEEK_VLLM_DIR) not in sys.path:
    sys.path.insert(0, str(DEEPSEEK_VLLM_DIR))

from config import CROP_MODE, MODEL_PATH, PROMPT  # type: ignore  # noqa: E402
from deepseek_ocr import (  # type: ignore  # noqa: E402
    DeepseekOCRForCausalLM,
    NGramPerReqLogitsProcessor,
)
from process.image_process import DeepseekOCRProcessor  # type: ignore  # noqa: E402

PDF_PATH = ROOT / "DeepSeek_OCR_paper.pdf"
OUTPUT_DOCS_DIR = ROOT / "outputs" / "docs"
OUTPUT_IMAGES_DIR = OUTPUT_DOCS_DIR / "images"
OUTPUT_MARKDOWN = OUTPUT_DOCS_DIR / "DeepSeek-OCR_paper.md"
PDF_DPI = 144

DETECTION_PATTERN = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL)
DETECTION_CAPTURE_PATTERN = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|><\|det\|>(?P<coords>.*?)<\|/det\|>",
    re.DOTALL,
)
FIGURE_KEYWORDS = ("image", "figure", "chart", "diagram", "graph", "plot", "photo")
REPLACEMENTS = {
    "\\coloneqq": ":=",
    "\\eqqcolon": "=:",
    "<--- Page Split --->": "",
}


def pdf_to_images(pdf_path: Path, dpi: int) -> List[Image.Image]:
    doc = fitz.open(str(pdf_path))
    images: List[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    try:
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            images.append(image)
    finally:
        doc.close()
    return images


def build_batch_inputs(images: List[Image.Image], processor: DeepseekOCRProcessor) -> List[dict]:
    batch_inputs: List[dict] = []
    for image in images:
        mm_data = processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)
        batch_inputs.append({
            "prompt": PROMPT,
            "multi_modal_data": {"image": mm_data},
        })
    return batch_inputs


def prepare_output_dirs() -> None:
    OUTPUT_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for asset in OUTPUT_IMAGES_DIR.iterdir():
        if asset.is_file():
            asset.unlink()


def _label_is_visual(label: str) -> bool:
    normalized = label.strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in FIGURE_KEYWORDS)


def _parse_boxes(coords_str: str) -> List[Tuple[float, float, float, float]]:
    try:
        parsed = ast.literal_eval(coords_str)
    except (SyntaxError, ValueError):
        return []

    boxes: List[Tuple[float, float, float, float]] = []

    def _collect(candidate: object) -> None:
        if isinstance(candidate, AbcSequence) and not isinstance(candidate, (str, bytes)):
            if len(candidate) == 4 and all(isinstance(point, (int, float)) for point in candidate):
                x1, y1, x2, y2 = candidate  # type: ignore[assignment]
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
                return
            for entry in candidate:
                _collect(entry)

    _collect(parsed)
    return boxes


def _to_pixel_boxes(
    boxes: Sequence[Tuple[float, float, float, float]],
    image_size: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    width, height = image_size
    if width <= 0 or height <= 0:
        return []

    pixel_boxes: List[Tuple[int, int, int, int]] = []

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


def replace_grounding_tokens(
    page_text: str,
    page_image: Image.Image,
    page_index: int,
) -> str:
    figure_count = 0
    seen: set[Tuple[int, int, int, int]] = set()
    image_width, image_height = page_image.size

    def _replacement(match: re.Match[str]) -> str:
        nonlocal figure_count
        label = match.group("label") or ""
        coords_str = match.group("coords") or ""

        boxes = _parse_boxes(coords_str)
        if not boxes:
            return ""

        if not _label_is_visual(label):
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
        if box_key in seen:
            return ""
        seen.add(box_key)

        figure_count += 1
        filename = f"page-{page_index:03}-figure-{figure_count:02}.png"
        figure_path = OUTPUT_IMAGES_DIR / filename
        crop = page_image.crop((left, top, right, bottom))
        crop.save(figure_path, format="PNG")

        alt_label = label.strip() or f"Figure {figure_count}"
        if alt_label.lower() in {"image", "figure"}:
            alt_label = f"Figure {figure_count}"

        return f"\n\n![{alt_label}](images/{filename})\n\n"

    return DETECTION_CAPTURE_PATTERN.sub(_replacement, page_text)


def sanitize_page_text(text: str) -> str:
    cleaned = text.replace("<｜end▁of▁sentence｜>", "")
    cleaned = DETECTION_PATTERN.sub("", cleaned)
    for needle, replacement in REPLACEMENTS.items():
        cleaned = cleaned.replace(needle, replacement)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def ensure_model_registered() -> None:
    try:
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
    except ValueError:
        pass


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    ensure_model_registered()
    prepare_output_dirs()

    pdf_images = pdf_to_images(PDF_PATH, PDF_DPI)
    if not pdf_images:
        raise RuntimeError("No pages extracted from PDF.")

    processor = DeepseekOCRProcessor()

    max_concurrency = 1  # max(1, min(8, len(pdf_images)))
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=max_concurrency,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
        extra_args={
            "ngram_size": 20,
            "window_size": 50,
            "whitelist_token_ids": [128821, 128822],
        },
    )

    batch_inputs = build_batch_inputs(pdf_images, processor)
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

    markdown_sections: List[str] = []
    for idx, output in enumerate(outputs, start=1):
        image_filename = f"page-{idx:03}.png"
        image_path = OUTPUT_IMAGES_DIR / image_filename
        page_image = pdf_images[idx - 1]
        page_image.save(image_path, format="PNG")

        section_header = f"<!-- Page {idx} -->"
        image_note = f"<!-- Page image stored at images/{image_filename} -->"

        if output.outputs:
            page_text = output.outputs[0].text
            grounded = replace_grounding_tokens(page_text, page_image, idx)
            cleaned = sanitize_page_text(grounded)
            if cleaned:
                markdown_sections.append(f"{section_header}\n{image_note}\n\n{cleaned}")
                continue

        markdown_sections.append(
            f"{section_header}\n{image_note}\n\n_No OCR output for this page._"
        )

    final_markdown = "\n\n".join(markdown_sections).strip() + "\n"
    OUTPUT_MARKDOWN.write_text(final_markdown, encoding="utf-8")
    print(f"Markdown written to {OUTPUT_MARKDOWN}")


if __name__ == "__main__":
    main()
