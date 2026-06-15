"""Local translation backends used by the live overlay."""

import json
import threading
import urllib.error
import urllib.request

from live_translation.text_pipeline import (
    language_name,
    strip_llm_noise,
    translategemma_prompt,
)


class LanguageSettings:
    def __init__(self, source, target):
        self._lock = threading.Lock()
        self._source = source
        self._target = target

    def get(self):
        with self._lock:
            return self._source, self._target

    def set_source(self, source):
        with self._lock:
            self._source = source

    def set_target(self, target):
        with self._lock:
            self._target = target


class OllamaTranslator:
    def __init__(self, model, target, url, max_tokens, temperature, reasoning, source="auto"):
        self.model = model
        self.target = target
        self.source = source
        self.url = url.rstrip("/") + "/api/generate"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.translategemma = "translategemma" in model.lower() or "translate-gemma" in model.lower()

    def set_target(self, target):
        self.target = target

    def set_source(self, source):
        self.source = source

    def translate(self, text):
        target = language_name(self.target)
        if self.translategemma and self.source and self.source != "auto":
            prompt = translategemma_prompt(self.source, self.target, text)
        else:
            think_prefix = "" if self.reasoning else "/no_think\n"
            prompt = (
                f"{think_prefix}Translate this live speech transcript into {target}.\n"
                "Return only the translation. Keep names, numbers, and technical terms accurate. "
                "Fix obvious speech-recognition glitches only when the meaning is clear.\n\n"
                f"Transcript:\n{text}"
            )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": 0.9,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama не отвечает. Запусти `ollama serve` и скачай модель: "
                f"`ollama pull {self.model}`."
            ) from exc
        return strip_llm_noise(body.get("response", ""))


class MLXTranslator:
    def __init__(self, model_repo, target, max_tokens, temperature, reasoning=False):
        # mlx_lm uses thread-local GPU streams, so import/load on the worker thread.
        self.model_repo = model_repo
        self.target = target
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning = reasoning
        self.model = None
        self.tokenizer = None
        self.generate = None
        self.make_sampler = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            try:
                from mlx_lm import generate, load
                from mlx_lm.sample_utils import make_sampler
            except ImportError as exc:
                raise RuntimeError("Для --translator mlx установи `pip install mlx-lm`.") from exc
            self.generate = generate
            self.make_sampler = make_sampler
            loaded = load(self.model_repo)
            self.model, self.tokenizer = loaded[0], loaded[1]

    def set_target(self, target):
        self.target = target

    def set_source(self, source):
        pass

    def translate(self, text):
        self._ensure_loaded()
        model, tokenizer = self.model, self.tokenizer
        generate, make_sampler = self.generate, self.make_sampler
        assert (
            model is not None
            and tokenizer is not None
            and generate is not None
            and make_sampler is not None
        ), "MLX translator not loaded"
        target = language_name(self.target)
        think_prefix = "" if self.reasoning else "/no_think\n"
        system = (
            "You are a low-latency translation engine. Return only the translation. "
            "Do not explain. Do not include alternatives."
        )
        user = (
            f"{think_prefix}Translate this live speech transcript into {target}. "
            "Keep names, numbers, and technical terms accurate. "
            "Fix obvious speech-recognition glitches only when the meaning is clear.\n\n"
            f"Transcript:\n{text}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.reasoning,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = f"{system}\n\nUser:\n{user}\n\nAssistant:\n"

        out = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=self.temperature, top_p=0.9),
            verbose=False,
        )
        return strip_llm_noise(out)
