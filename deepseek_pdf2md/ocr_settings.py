"""Configuration model for the DeepSeek OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import vllm.transformers_utils.processors.deepseek_ocr as deepseek_processor_module
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


class OCRSettings(BaseSettings):
    """Raw settings populated from env vars, `.env`, or defaults."""

    model_config = SettingsConfigDict(
        env_prefix="DEEPSEEK_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        toml_file="ocr_settings.toml",
    )

    model: str = Field(default="deepseek-ai/DeepSeek-OCR")
    prompt: str = Field(default=DEFAULT_PROMPT)
    pdf_path: Path = Field(default=Path("DeepSeek_OCR_paper.pdf"))
    output_docs_dir: Path = Field(default=Path("outputs/docs"))
    output_images_dir: Path | None = Field(default=None)
    output_markdown: Path = Field(default=Path("DeepSeek-OCR_paper.md"))
    pdf_dpi: int = Field(default=144)
    crop_mode: bool = Field(default=True)
    max_concurrency: int = Field(default=1)
    image_size: int = Field(default=getattr(deepseek_processor_module, "IMAGE_SIZE", 640))
    base_size: int = Field(default=getattr(deepseek_processor_module, "BASE_SIZE", 1024))
    min_crops: int = Field(default=deepseek_processor_module.MIN_CROPS if hasattr(deepseek_processor_module, "MIN_CROPS") else 2)  # type: ignore[name-defined]
    max_crops: int = Field(default=deepseek_processor_module.MAX_CROPS if hasattr(deepseek_processor_module, "MAX_CROPS") else 6)  # type: ignore[name-defined]
    ngram_size: int = Field(default=20)
    window_size: int = Field(default=50)


    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        override_path = os.getenv("DEEPSEEK_TOML")
        if override_path:
            toml_settings = TomlConfigSettingsSource(settings_cls, Path(override_path))
        else:
            toml_settings = TomlConfigSettingsSource(settings_cls)
        return (
            init_settings,
            toml_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


@dataclass(frozen=True)
class ResolvedOCRSettings:
    model_path: str
    prompt: str
    pdf_path: Path
    output_docs_dir: Path
    output_images_dir: Path
    output_markdown: Path
    pdf_dpi: int
    crop_mode: bool
    max_concurrency: int
    image_size: int
    base_size: int
    min_crops: int
    max_crops: int
    ngram_size: int
    window_size: int


def load_settings(root: Path, settings_file: Path | None = None) -> ResolvedOCRSettings:
    """Instantiate OCR settings and resolve any relative file system paths."""

    env_var = "DEEPSEEK_TOML"
    previous_override = os.environ.get(env_var)
    try:
        if settings_file is not None:
            settings_file = settings_file if settings_file.is_absolute() else root / settings_file
            os.environ[env_var] = str(settings_file.resolve())

        raw = OCRSettings()
    finally:
        if settings_file is not None:
            if previous_override is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = previous_override

    pdf_path = raw.pdf_path
    if not pdf_path.is_absolute():
        pdf_path = root / pdf_path

    output_docs_dir = raw.output_docs_dir
    if not output_docs_dir.is_absolute():
        output_docs_dir = root / output_docs_dir

    if raw.output_images_dir is not None:
        output_images_dir = raw.output_images_dir
        if not output_images_dir.is_absolute():
            output_images_dir = root / output_images_dir
    else:
        output_images_dir = output_docs_dir / "images"

    output_markdown = raw.output_markdown
    if not output_markdown.is_absolute():
        output_markdown = output_docs_dir / output_markdown

    max_concurrency = max(1, raw.max_concurrency)
    image_size = max(64, raw.image_size)
    base_size = max(image_size, raw.base_size)
    min_crops = max(1, raw.min_crops)
    max_crops = max(min_crops, raw.max_crops)
    ngram_size = max(1, raw.ngram_size)
    window_size = max(ngram_size, raw.window_size)

    return ResolvedOCRSettings(
        model_path=raw.model,
        prompt=raw.prompt,
        pdf_path=pdf_path.resolve(),
        output_docs_dir=output_docs_dir.resolve(),
        output_images_dir=output_images_dir.resolve(),
        output_markdown=output_markdown.resolve(),
        pdf_dpi=raw.pdf_dpi,
        crop_mode=raw.crop_mode,
        max_concurrency=max_concurrency,
        image_size=image_size,
        base_size=base_size,
        min_crops=min_crops,
        max_crops=max_crops,
        ngram_size=ngram_size,
        window_size=window_size,
    )
