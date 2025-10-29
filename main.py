from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from typing import List

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
OUTPUT_MARKDOWN = ROOT / "DeepSeek-OCR_paper.md"
PDF_DPI = 144

DETECTION_PATTERN = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL)
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

    pdf_images = pdf_to_images(PDF_PATH, PDF_DPI)
    if not pdf_images:
        raise RuntimeError("No pages extracted from PDF.")

    processor = DeepseekOCRProcessor()

    max_concurrency = max(1, min(8, len(pdf_images)))
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
        if not output.outputs:
            continue
        page_text = output.outputs[0].text
        cleaned = sanitize_page_text(page_text)
        section_header = f"<!-- Page {idx} -->"
        if cleaned:
            markdown_sections.append(f"{section_header}\n{cleaned}")
        else:
            markdown_sections.append(section_header)

    final_markdown = "\n\n".join(markdown_sections).strip() + "\n"
    OUTPUT_MARKDOWN.write_text(final_markdown, encoding="utf-8")
    print(f"Markdown written to {OUTPUT_MARKDOWN}")


if __name__ == "__main__":
    main()
