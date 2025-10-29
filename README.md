## DeepSeek OCR PDF-to-Markdown Pipeline

This repository wraps the upstream DeepSeek-OCR vision-language model (served via vLLM) to convert PDFs into Markdown with linked images and figure crops. The local code is intentionally thin—the `deepseek_pdf2md` package orchestrates the workflow, `ocr_settings.toml` exposes all user-tunable knobs with documented defaults, and vendored vLLM sources are treated as read-only snapshots.

### Prerequisites

- Python 3.12 (the repo uses [`uv`](https://github.com/astral-sh/uv) for dependency management)
- CUDA 12.8-capable GPU drivers
- DeepSeek-OCR weights available either from Hugging Face (`deepseek-ai/DeepSeek-OCR`) or a local checkpoint directory

Install the environment once:

```bash
uv sync
```

Optional sanity check:

```bash
uv run python -m vllm.collect_env
```

### Quick Start

1. Edit `ocr_settings.toml` (or export `DEEPSEEK_*` variables) to point at your PDF and adjust runtime options.
2. Run the pipeline (optionally pass `--settings` or `--verbose`):

   ```bash
   uv run deepseek-pdf2md
   ```

3. Review outputs under `outputs/docs/`: the Markdown file, per-page PNGs (`page-###.png`), and any cropped figures (`page-###-figure-##.png`).

### Pipeline Flow

1. **Configuration** – loads `ocr_settings.toml` through `ocr_settings.load_settings`, resolving relative paths and honouring `.env`/environment overrides.
2. **Output prep** – ensures `outputs/docs/` (or your configured directory) exists and wipes stale images.
3. **PDF rasterisation** – converts each page to a PIL image at `pdf_dpi` using PyMuPDF (`fitz`).
4. **vLLM setup** – applies DeepSeek processor overrides (`crop_mode`, `image_size`, `base_size`, `min_crops`, `max_crops`) and instantiates `vllm.LLM` with the DeepSeek OCR architecture. Concurrency is capped by `max_concurrency` to avoid GPU OOM.
5. **Generation** – feeds each page as a `<image>` prompt (with optional grounding tags) and enables the n-gram repeat guard using `ngram_size`/`window_size`.
6. **Sanitisation** – removes DeepSeek grounding tokens (`<|ref|>`, `<|det|>`, `<ref>…</ref>`), coordinate dumps, and long single-word loops before writing Markdown.
7. **Embedding** – saves page images and figure crops, inserting `![alt](images/…)` references inline.

Key dependencies used at runtime:

- `vllm.transformers_utils.processors.deepseek_ocr` – tokenisation & tiling (unmodified upstream code)
- `vllm.model_executor.models.deepseek_ocr.NGramPerReqLogitsProcessor` – repeat suppression
- `deepseek_pdf2md.ocr_settings` – Pydantic `BaseSettings` loader with TOML/env integration

### Configuration Reference (`ocr_settings.toml`)

Every setting below is documented in the TOML file and can be overridden via environment variables (`DEEPSEEK_*`) or `.env`.

| Key | Description |
|-----|-------------|
| `model` | Model identifier or filesystem path. Examples: `deepseek-ai/DeepSeek-OCR`, `/models/dpsk`. |
| `prompt` | Prompt sent to the model. Supports DeepSeek tags such as `<image>`, `<|grounding|>`, `<|ref|>…</|ref|>`, `<|det|>…</|det|>`. Sample prompts are included in the TOML for document OCR, free OCR, figure parsing, and targeted text lookup. |
| `pdf_path` | Source PDF. Relative paths resolve against the repository root. |
| `output_docs_dir` | Root folder for generated Markdown and assets. |
| `output_images_dir` | Optional explicit images directory; defaults to `output_docs_dir/images`. |
| `output_markdown` | Markdown filename (relative to `output_docs_dir`). |
| `pdf_dpi` | DPI used when rasterising the PDF with PyMuPDF. |
| `crop_mode` | Enables tiled crops (`true`) or single global view (`false`). Leave `false` unless you have GPU headroom and need detailed layouts. |
| `max_concurrency` | Maximum simultaneous page prompts processed by vLLM. Lower this if you encounter CUDA OOM. |
| `image_size`, `base_size` | Vision encoder dimensions. Defaults align with DeepSeek “Gundam” (640×1024). |
| `min_crops`, `max_crops` | Crop tiling bounds (applicable when `crop_mode=true`). |
| `ngram_size`, `window_size` | Passed to `NGramPerReqLogitsProcessor` to suppress repeated n-grams. |

To override via environment variables, use the `DEEPSEEK_` prefix (e.g. `DEEPSEEK_PROMPT`, `DEEPSEEK_MAX_CONCURRENCY`). Values in `.env` follow the same naming scheme.
You can also point to an alternate TOML file by exporting `DEEPSEEK_TOML=/path/to/settings.toml` or passing `--settings` on the CLI.

### Prompt Cheat Sheet

DeepSeek supports grounding-aware prompts:

- **Document OCR**: `<image>
<|grounding|>Convert the document to markdown.`
- **Free OCR**: `<image>
Free OCR.`
- **Figure parsing**: `<image>
Parse the figure.`
- **Target lookup**: `<image>
Locate <|ref|>Invoice Number<|/ref|> in the image.`

Grounding outputs embed `<|ref|>`/`<|det|>` tokens identifying detected spans; the sanitiser in `deepseek_pdf2md.markdown_utils` removes them while still saving cropped figures.

### Troubleshooting Tips

- **CUDA out of memory**: Lower `max_concurrency`, disable tiling (`crop_mode=false`), or reduce `image_size`/`base_size`.
- **Repetitive text loops**: Decrease `ngram_size` and/or `window_size` in the TOML to catch shorter repetitions.
- **Placeholder tags left in output**: Ensure you’re on the latest pipeline—`sanitize_page_text` removes known tags. If new tokens appear, extend the regex filters in `deepseek_pdf2md.markdown_utils`.
- **Prompt mismatch errors**: The model expects one `<image>` placeholder per supplied image; the pipeline handles this automatically when building batch inputs.

### Repository Layout

```
deepseek_pdf2md/    # Python package (CLI, settings loader, pipeline modules)
ocr_settings.toml   # Documented defaults for all runtime knobs
vllm/               # Vendored vLLM snapshot (read-only)
outputs/            # Generated markdown and image artefacts
```

### Contributing Guidelines

- Avoid modifying vendored vLLM files; prefer runtime overrides in the package and documented settings.
- Document any new configuration options in both `ocr_settings.toml` and this README.
- After config or sanitiser changes, re-run `uv run deepseek-pdf2md` to confirm the pipeline still produces clean Markdown without placeholder artefacts.
