#!/bin/bash
# One-shot setup for Live Translate: Python deps + system deps + models.
# Usage: ./setup.sh
set -e
cd "$(dirname "$0")"

echo "==> 1/4  Python venv + pip dependencies"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "==> 2/4  System dependencies (Homebrew: BlackHole audio + Ollama)"
if command -v brew >/dev/null 2>&1; then
    brew bundle --file=Brewfile
else
    echo "    Homebrew не найден — поставь вручную: https://brew.sh"
    echo "    затем: brew bundle --file=Brewfile"
fi

echo "==> 3/4  Ollama translation model (optional translategemma backend)"
if command -v ollama >/dev/null 2>&1; then
    ollama pull translategemma:12b || echo "    пропускаю (запусти 'ollama serve' и повтори при желании)"
else
    echo "    ollama не найден — пропускаю (нужен только для --translator ollama)"
fi

echo "==> 4/4  Pre-fetch speech + translation models (Whisper large-v3, Qwen3.5-9B)"
./.venv/bin/python - <<'PY' || echo "    модели докачаются при первом запуске"
from huggingface_hub import snapshot_download
for repo in ("mlx-community/whisper-large-v3-mlx", "mlx-community/Qwen3.5-9B-MLX-4bit"):
    print("    fetching", repo)
    snapshot_download(repo)
PY

echo ""
echo "Готово. Запуск: ./live_translate_overlay.py   (или двойной клик по LiveTranslate.app)"
echo "Не забудь в Системных настройках направить системный звук в BlackHole 2ch."
