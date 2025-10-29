# Repository Guidelines

## Project Structure & Module Organization
The entrypoint `main.py` orchestrates the PDF-to-Markdown pipeline, loading DeepSeek OCR components and writing results to `deepseek_optical_memory_chat.md`. Model code lives under `DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/` (notably `config.py`, `deepseek_ocr.py`, and `process/`). Treat these files as upstream snapshots: isolate local changes under clear comments. Vendored runtime logic is in `vllm/`; sync against the upstream submodule instead of patching files piecemeal. Use `outputs/` for generated images and intermediate artifacts; keep sample assets inside `DeepSeek-OCR/assets`.

## Build, Test, and Development Commands
We standardize on `uv` for dependency management. Run `uv sync` once to install the locked toolchain, then `uv run python main.py` to convert the bundled `deepseek_optical_memory_chat.pdf`. GPU availability (CUDA 12.8) is assumed; confirm with `uv run python -m vllm.collect_env` before debugging inference issues. Prefer `uv run` for all scripts so the locked environment is always used.

## Coding Style & Naming Conventions
Target Python 3.12 syntax with 4-space indentation and explicit type hints (mirroring `main.py`). Use snake_case for functions and modules, PascalCase only for classes, and ALL_CAPS for configuration constants (see `config.py`). When editing shared runtime code, mirror the upstream `vllm/pyproject.toml` `ruff` rules; for repository-specific modules run `uvx ruff check .` before submitting. Keep imports sorted (stdlib → third party → local).

## Testing Guidelines
Scenario tests rely on `pytest`. Validate local changes with focused suites, e.g. `uv run pytest vllm/tests/v1/engine -k decode` for engine updates, or bespoke tests placed under a new `tests/` folder for repository code. Always regenerate `deepseek_optical_memory_chat.md` via `uv run python main.py` and sanity-check the diff—attach representative excerpts in review notes. Call out any GPU-specific assumptions in PR descriptions.

## Commit & Pull Request Guidelines
Follow the existing history: short, imperative summaries (`Update vllm submodule…`, `correct project name`). Reference issue IDs when applicable and keep body lines wrapped at ~72 chars. PRs should describe the motivation, mention affected scripts or configs, and include before/after evidence (logs, rendered markdown, screenshots). Flag any required secrets or model checkpoints, and confirm you ran the relevant commands from the sections above.

## Configuration Tips
Model endpoints and crop behaviour are controlled in `DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`; update `MODEL_PATH`, `CROP_MODE`, and concurrency settings to match available GPUs. For custom PDFs, adjust `PDF_PATH`/`OUTPUT_MARKDOWN` at the top of `main.py` instead of hard-coding inside functions. Document any environment variables or resource tweaks directly in your PR for reproducibility.
