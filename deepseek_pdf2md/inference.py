"""vLLM-based DeepSeek OCR inference helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable, List

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from vllm.v1.engine.exceptions import EngineDeadError

import vllm.transformers_utils.processors.deepseek_ocr as deepseek_processor_module

from .ocr_settings import ResolvedOCRSettings


def generate_page_text(
    images: List[Image.Image],
    settings: ResolvedOCRSettings,
    prompt: str,
    logger: logging.Logger,
) -> List[str]:
    """Run DeepSeek-OCR over the provided images and return raw text outputs."""

    _apply_processor_overrides(settings, logger)

    llm = _create_llm(settings, logger)

    try:
        batch_inputs = _build_batch_inputs(images, prompt)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            extra_args={
                "ngram_size": settings.ngram_size,
                "window_size": settings.window_size,
                "whitelist_token_ids": [128821, 128822],
            },
        )

        logger.info("Submitting %s pages to DeepSeek-OCR", len(batch_inputs))
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    except EngineDeadError as exc:
        raise RuntimeError(
            "DeepSeek-OCR engine terminated unexpectedly. Consider lowering "
            "max_concurrency or disabling crop_mode."
        ) from exc
    finally:
        if hasattr(llm, "shutdown"):
            llm.shutdown()

    page_text: List[str] = []
    for index, output in enumerate(outputs, start=1):
        if output.outputs:
            text = output.outputs[0].text
        else:
            text = ""
            logger.warning("No generation returned for page %s", index)
        page_text.append(text)

    return page_text


def _apply_processor_overrides(settings: ResolvedOCRSettings, logger: logging.Logger) -> None:
    deepseek_processor_module.CROP_MODE = settings.crop_mode
    deepseek_processor_module.IMAGE_SIZE = settings.image_size
    deepseek_processor_module.BASE_SIZE = settings.base_size
    deepseek_processor_module.MIN_CROPS = settings.min_crops  # type: ignore[attr-defined]
    deepseek_processor_module.MAX_CROPS = settings.max_crops  # type: ignore[attr-defined]
    logger.debug(
        "Processor overrides set: crop_mode=%s image_size=%s base_size=%s min_crops=%s max_crops=%s",
        settings.crop_mode,
        settings.image_size,
        settings.base_size,
        settings.min_crops,
        settings.max_crops,
    )


def _create_llm(settings: ResolvedOCRSettings, logger: logging.Logger) -> LLM:
    max_concurrency = max(1, settings.max_concurrency)
    logger.info("Initialising vLLM engine (max_concurrency=%s)", max_concurrency)
    return LLM(
        model=settings.model_path,
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


def _build_batch_inputs(images: Iterable[Image.Image], prompt: str) -> List[dict]:
    batch_inputs: List[dict] = []
    for image in images:
        batch_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": [image]},
        })
    return batch_inputs
