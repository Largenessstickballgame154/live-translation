#!/usr/bin/env python3
"""
live_translate_overlay.py

Потоково слушает системный звук macOS через BlackHole, транскрибирует короткими
чанками через MLX Whisper, переводит локальной LLM и выводит результат в
полупрозрачное macOS-окно с frosted-glass эффектом.

Быстрый старт:

    brew install blackhole-2ch ffmpeg
    pip install mlx-whisper mlx-lm sounddevice soundfile numpy pyobjc-framework-Cocoa

    ./live_translate_overlay.py --target ru

По умолчанию перевод идёт через MLX-версию Qwen Instruct:

    mlx-community/Qwen3.5-4B-MLX-4bit

Если хочешь использовать Ollama вместо MLX:

    brew install ollama
    ollama pull qwen3.5:4b
    ./live_translate_overlay.py --translator ollama --ollama-model qwen3.5:4b

Звук настраивается так же, как в record_and_transcribe.py:
Multi-Output Device = твои наушники/колонки + BlackHole 2ch.
"""

import argparse
import json
import os
import queue
import re
import signal
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import sounddevice as sd
import soundfile as sf


WHISPER_MODELS = {
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

LANG_NAMES = {
    "auto": "the target language",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "zh": "Chinese",
}

LANG_MENU = [
    ("auto", "Auto"),
    ("en", "English"),
    ("ru", "Russian"),
    ("es", "Spanish"),
    ("de", "German"),
    ("fr", "French"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("uk", "Ukrainian"),
    ("zh", "Chinese"),
]

WAITING_ORIGINAL = "Waiting for speech…"
WAITING_TRANSLATION = {
    "en": "Waiting for translation…",
    "ru": "Жду перевод…",
    "es": "Esperando traducción…",
    "de": "Warte auf Übersetzung…",
    "fr": "En attente de traduction…",
    "it": "In attesa di traduzione…",
    "pt": "Aguardando tradução…",
    "uk": "Чекаю на переклад…",
    "zh": "等待翻译…",
}


def language_name(code):
    return LANG_NAMES.get((code or "").lower(), code or "the target language")


def language_label(code):
    for item_code, label in LANG_MENU:
        if item_code == code:
            return label
    return code or "Auto"


# Extra names for languages Whisper may auto-detect (beyond the UI menu), used to fill
# TranslateGemma's required "{LANG} ({code})" prompt slots.
_TG_EXTRA_NAMES = {
    "ca": "Catalan", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "nl": "Dutch",
    "pl": "Polish", "tr": "Turkish", "sv": "Swedish", "cs": "Czech", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "fi": "Finnish", "da": "Danish", "no": "Norwegian",
    "he": "Hebrew", "hi": "Hindi", "id": "Indonesian", "vi": "Vietnamese", "th": "Thai",
    "gl": "Galician", "eu": "Basque",
}


def _tg_name(code):
    code = (code or "").lower()
    return LANG_NAMES.get(code) or _TG_EXTRA_NAMES.get(code) or (code.upper() if code else "the language")


def _tg_code(code):
    return {"zh": "zh-Hans"}.get((code or "").lower(), (code or "").lower())


def translategemma_prompt(src_code, tgt_code, text):
    """Official TranslateGemma prompt (note: two blank lines before the text)."""
    sname, scode = _tg_name(src_code), _tg_code(src_code)
    tname, tcode = _tg_name(tgt_code), _tg_code(tgt_code)
    return (
        f"You are a professional {sname} ({scode}) to {tname} ({tcode}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {sname} "
        f"text while adhering to {tname} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tname} translation, without any additional explanations or "
        f"commentary. Please translate the following {sname} text into {tname}:\n\n\n{text}\n"
    )


# Phrases Whisper hallucinates on silence/music/chunk-boundaries (its training data was
# full of subtitle boilerplate). Stripped from the transcript as standalone tokens.
HALLUCINATION_PATTERNS = [
    r"¡?\s*gracias\s*!?",
    r"muchas\s+gracias",
    r"subt[ií]tulos?(?:\s+(?:realizados?|por)[^.\n]*)?",
    r"amara\.org",
    r"thanks?\s+for\s+watching",
    r"please\s+subscribe",
    r"subtitles?\s+by[^.\n]*",
    r"спасибо\s+за\s+просмотр",
    r"продолжение\s+следует",
]
_HALLUCINATION_RE = re.compile(
    r"(?<![\w¡])(?:" + "|".join(HALLUCINATION_PATTERNS) + r")(?![\w])",
    re.IGNORECASE,
)


def strip_hallucinations(text):
    """Remove standalone Whisper boilerplate hallucinations (e.g. a stray "Gracias")."""
    cleaned = _HALLUCINATION_RE.sub(" ", text)
    cleaned = re.sub(r"\s+([,.;:!?…])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" ,.;:!-—")


def strip_llm_noise(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"(?is)^\s*thinking process:.*?(final answer:|answer:)", "", text)
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip().strip('"')


def sentence_case_text(text):
    chars = list(text)
    should_capitalize = True
    for idx, char in enumerate(chars):
        if not char.isalpha():
            if char in ".!?":
                should_capitalize = True
            continue
        if should_capitalize:
            chars[idx] = char.upper()
            should_capitalize = False
        else:
            should_capitalize = False
    return "".join(chars)


def _word_matches(text):
    return list(re.finditer(r"[\w'’-]+", text, flags=re.UNICODE))


def _normalized_words(text):
    return [m.group(0).casefold().strip("'’-") for m in _word_matches(text)]


def _drop_prefix_words(text, word_count):
    end = 0
    for idx, match in enumerate(_word_matches(text), 1):
        if idx == word_count:
            end = match.end()
            break
    return text[end:].lstrip(" ,.;:!?…-—")


def _sentence_end_matches(text):
    return list(re.finditer(r"(?<!\.)[.!?](?!\.)(?:[\"')\]]+)?(?=\s|$)|…(?:[\"')\]]+)?(?=\s|$)", text))


def merge_overlap_text(previous_tail, incoming, min_overlap=12, max_overlap=120):
    incoming = re.sub(r"\s+", " ", incoming).strip()
    previous_tail = re.sub(r"\s+", " ", previous_tail).strip()
    if not previous_tail or not incoming:
        return incoming

    prev_words = _normalized_words(previous_tail)
    incoming_words = _normalized_words(incoming)
    if incoming_words and " ".join(incoming_words) in " ".join(prev_words):
        return ""

    max_words = min(len(prev_words), len(incoming_words), 28)
    for size in range(max_words, 2, -1):
        if prev_words[-size:] == incoming_words[:size]:
            return _drop_prefix_words(incoming, size)
    if prev_words and incoming_words:
        if prev_words[-1] == incoming_words[0] and len(incoming_words[0]) >= 5:
            return _drop_prefix_words(incoming, 1)

    prev_lower = previous_tail.lower()
    incoming_lower = incoming.lower()
    max_len = min(len(prev_lower), len(incoming_lower), max_overlap)
    for size in range(max_len, min_overlap - 1, -1):
        if prev_lower[-size:] == incoming_lower[:size]:
            return incoming[size:].lstrip()
    return incoming


def merge_partial_buffer(buffer, incoming):
    buffer = re.sub(r"\s+", " ", buffer).strip()
    incoming = re.sub(r"\s+", " ", incoming).strip()
    if not buffer:
        return incoming
    if not incoming:
        return buffer

    buffer_words = _normalized_words(buffer)
    incoming_words = _normalized_words(incoming)
    if len(incoming_words) >= 8:
        window = " ".join(incoming_words[: min(10, len(incoming_words))])
        joined_buffer = " ".join(buffer_words)
        if window and window in joined_buffer:
            prefix = joined_buffer.split(window, 1)[0].split()
            keep_words = len(prefix)
            matches = _word_matches(buffer)
            cut = matches[keep_words].start() if keep_words < len(matches) else len(buffer)
            return f"{buffer[:cut].strip()} {incoming}".strip()

    merged = merge_overlap_text(buffer[-500:], incoming, min_overlap=8, max_overlap=160)
    if not merged:
        return buffer
    return f"{buffer} {merged}".strip()


def split_complete_sentences(buffer, min_sentence_chars):
    text = re.sub(r"\s+", " ", buffer).strip()
    if not text:
        return [], ""

    matches = _sentence_end_matches(text)
    if not matches:
        return [], text

    cut = matches[-1].end()
    complete = text[:cut].strip()
    rest = text[cut:].strip()
    sentences = []
    start = 0
    for match in _sentence_end_matches(complete):
        sentence = complete[start : match.end()].strip()
        start = match.end()
        if len(sentence) >= min_sentence_chars:
            sentences.append(sentence)
        elif sentences:
            sentences[-1] = f"{sentences[-1]} {sentence}".strip()
        elif sentence:
            sentences.append(sentence)
    return sentences, rest


def take_sentences_for_translation(buffer, min_sentence_chars, max_sentences, force=False):
    sentences, rest = split_complete_sentences(buffer, min_sentence_chars)
    if force and not sentences and buffer.strip():
        return [buffer.strip()], ""
    if not sentences:
        return [], rest
    selected = sentences[:max_sentences]
    remaining_sentences = sentences[max_sentences:]
    remaining = " ".join(remaining_sentences + ([rest] if rest else [])).strip()
    return selected, remaining


def take_blocks_for_translation(buffer, min_chars, max_chars, max_sentences, force=False):
    sentences, rest = split_complete_sentences(buffer, min_sentence_chars=1)
    if force and not sentences:
        return [], buffer.strip()

    blocks = []
    current = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        should_flush = (
            current
            and (
                current_len >= min_chars
                or len(current) >= max_sentences
                or current_len + sentence_len + 1 > max_chars
            )
        )
        if should_flush:
            blocks.append(" ".join(current).strip())
            current = []
            current_len = 0
        current.append(sentence)
        current_len += sentence_len + 1

    if current and (current_len >= min_chars or force):
        blocks.append(" ".join(current).strip())
        current = []

    remaining = " ".join(current + ([rest] if rest else [])).strip()
    return blocks, remaining


_SOFT_PUNCT = re.compile(r"[,;:—–](?=\s)")


def split_soft_boundaries(text, min_chars, max_chars):
    """Split a punctuation-less run into readable paragraphs.

    When Whisper drifts and stops emitting sentence punctuation, the buffer would
    otherwise be dumped as one long wall. Instead, cut it at the nearest clause
    boundary (comma/semicolon/colon/dash) — or, failing that, at a word boundary —
    around a comfortable target length, never mid-word. Language-agnostic: relies
    only on commas/spaces, not on a per-language word list."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    target = min(max_chars, max(min_chars * 2, 200))
    blocks = []
    while len(text) > target:
        cut = None
        for m in _SOFT_PUNCT.finditer(text[: target + 1]):
            if m.end() >= min_chars:
                cut = m.end()  # keep the last clause boundary within the window
        if cut is None:
            sp = text.rfind(" ", min_chars, target + 1)
            cut = sp if sp >= min_chars else target
        blocks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        blocks.append(text)
    return blocks


def take_endpoint_blocks(buffer, min_chars, max_chars, min_words):
    blocks, remaining = take_blocks_for_translation(
        buffer,
        min_chars=min_chars,
        max_chars=max_chars,
        max_sentences=100,
        force=True,
    )
    if blocks:
        return blocks, remaining

    text = re.sub(r"\s+", " ", buffer).strip()
    if len(text) >= min_chars or len(_normalized_words(text)) >= min_words:
        # An endpoint is a real speech pause, so the whole utterance is complete even
        # without trailing punctuation. But Whisper sometimes drops punctuation for a
        # long stretch — emit that as several readable paragraphs (clause/word cuts)
        # rather than one wall.
        return split_soft_boundaries(text, min_chars, max_chars), ""
    return [], text


def punctuated_context(text, max_unpunctuated_tail=80):
    """Recent text to feed Whisper as initial_prompt — but only while it's still
    well-punctuated right up to the end.

    initial_prompt biases the next chunk. Feeding back a punctuation-less run makes
    Whisper keep omitting punctuation (a self-reinforcing drift). It's not enough to
    check for *any* period: an old period from an earlier block keeps the gate open
    while a fresh punctuation-less run is already trailing it, so the drift persists
    until that old period scrolls out of the window (the "fixes itself eventually"
    lag). So we also require the tail after the last sentence end to be short — once
    a long unpunctuated run is in progress, drop the context immediately and let the
    next chunk re-introduce sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()[-240:]
    if not text:
        return None
    matches = _sentence_end_matches(text)
    if not matches:
        return None
    tail = text[matches[-1].end():].strip()
    if len(tail) > max_unpunctuated_tail:
        return None
    return text


def strip_to_recent_text(text, max_chars):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def last_word_end_seconds(result):
    last_end = None
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            end = word.get("end")
            if end is not None:
                last_end = float(end)
    return last_end


def find_blackhole():
    for idx, dev in enumerate(sd.query_devices()):
        dev = cast(dict, dev)  # sounddevice yields dict-like entries
        if dev["max_input_channels"] > 0 and "blackhole" in dev["name"].lower():
            return idx, dev["name"]
    return None, None


def list_devices():
    print(sd.query_devices())
    print("\nИщи входное устройство BlackHole. Его номер можно передать через --device N.")


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
            # Use TranslateGemma's required prompt template for best quality.
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
        # NB: do NOT import mlx_lm or load the model here. mlx_lm.generate creates a
        # thread-local GPU stream at import time, and the model must be evaluated on
        # the same thread that owns that stream. __init__ runs on the main thread, but
        # translate() runs in the transcribe/translate worker thread. So we defer both
        # the import and the load to the first translate() call (worker thread),
        # otherwise long prompts crash with "There is no Stream(gpu, 1) in current thread".
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
        pass  # Qwen prompt doesn't need an explicit source language

    def translate(self, text):
        self._ensure_loaded()
        # _ensure_loaded() guarantees these are populated; bind to locals so the
        # invariant is explicit (and the optional-None warnings go away).
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


class ConsoleOverlay:
    def __init__(self, stop_event):
        self.stop_event = stop_event

    def post_pair(self, source, translated, pause_ms=0):
        print("\n--- transcript ---")
        print(source)
        print("--- translation ---")
        print(translated, flush=True)

    def post_status(self, lag_chunks):
        if lag_chunks:
            print(f"lag: {lag_chunks} chunks", flush=True)

    def post_partial(self, source):
        if source:
            print("\n--- partial ---")
            print(source, flush=True)

    def run(self):
        while not self.stop_event.is_set():
            time.sleep(0.2)

    def stop(self):
        self.stop_event.set()


_GLASS_PDF_VIEW_CLASS = None


def _glass_pdf_view_class():
    """Lazily build & cache a flipped NSView subclass that vector-draws the
    'liquid glass' PDF: a soft diagonal gradient backdrop, frosted rounded cards
    with a hairline rim and a soft drop shadow, and the transcript text on top.

    Cached because an Obj-C class can only be registered once per process; and
    built lazily so importing this module (e.g. for --help) doesn't pull AppKit."""
    global _GLASS_PDF_VIEW_CLASS
    if _GLASS_PDF_VIEW_CLASS is not None:
        return _GLASS_PDF_VIEW_CLASS

    import objc
    from AppKit import (
        NSBezierPath,
        NSColor,
        NSGraphicsContext,
        NSShadow,
        NSStringDrawingUsesLineFragmentOrigin,
        NSView,
    )

    class _GlassPDFView(NSView):
        def initWithFrame_(self, frame):
            self = objc.super(_GlassPDFView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._gradient = None
            self._cards = []   # list of NSRect (card backgrounds)
            self._texts = []   # list of (NSAttributedString, NSRect)
            return self

        def isFlipped(self):  # top-down layout
            return True

        def drawRect_(self, _dirty):
            bounds = self.bounds()
            if self._gradient is not None:
                self._gradient.drawInRect_angle_(bounds, -60.0)

            fill = NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.62)
            rim = NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.9)
            for rect in self._cards:
                path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    rect, 20.0, 20.0
                )
                NSGraphicsContext.saveGraphicsState()
                shadow = NSShadow.alloc().init()
                shadow.setShadowColor_(NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.13))
                shadow.setShadowBlurRadius_(16.0)
                shadow.setShadowOffset_((0.0, 3.0))  # flipped view -> downward
                shadow.set()
                fill.set()
                path.fill()
                NSGraphicsContext.restoreGraphicsState()
                rim.set()
                path.setLineWidth_(1.0)
                path.stroke()

            for attr, rect in self._texts:
                attr.drawWithRect_options_(rect, NSStringDrawingUsesLineFragmentOrigin)

    _GLASS_PDF_VIEW_CLASS = _GlassPDFView
    return _GLASS_PDF_VIEW_CLASS


class GlassOverlay:
    def __init__(self, stop_event, settings, title, width, height, opacity):
        try:
            from Cocoa import (
                NSApp,
                NSApplication,
                NSApplicationActivationPolicyRegular,
                NSBackingStoreBuffered,
                NSButton,
                NSColor,
                NSFloatingWindowLevel,
                NSFont,
                NSMakeRange,
                NSMakeRect,
                NSMakeSize,
                NSMutableAttributedString,
                NSObject,
                NSPopUpButton,
                NSScrollView,
                NSSlider,
                NSTextField,
                NSTextView,
                NSFontAttributeName,
                NSForegroundColorAttributeName,
                NSMomentaryPushInButton,
                NSNormalWindowLevel,
                NSOffState,
                NSOnState,
                NSSwitchButton,
                NSView,
                NSViewHeightSizable,
                NSViewWidthSizable,
                NSVisualEffectBlendingModeBehindWindow,
                NSVisualEffectMaterialHUDWindow,
                NSVisualEffectStateActive,
                NSVisualEffectView,
                NSWindow,
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSWindowStyleMaskClosable,
                NSWindowStyleMaskFullSizeContentView,
                NSWindowStyleMaskResizable,
                NSWindowStyleMaskTitled,
            )
            from PyObjCTools import AppHelper
        except ImportError as exc:
            raise RuntimeError(
                "Для стеклянного окна установи `pip install pyobjc-framework-Cocoa`."
            ) from exc

        self.AppHelper = AppHelper
        self.NSMakeRange = NSMakeRange
        self.NSMakeRect = NSMakeRect
        self.NSMakeSize = NSMakeSize
        self.NSButton = NSButton
        self.NSColor = NSColor
        self.NSFont = NSFont
        self.NSFontAttributeName = NSFontAttributeName
        self.NSForegroundColorAttributeName = NSForegroundColorAttributeName
        self.NSNormalWindowLevel = NSNormalWindowLevel
        self.NSFloatingWindowLevel = NSFloatingWindowLevel
        self.NSOffState = NSOffState
        self.NSOnState = NSOnState
        self.NSSlider = NSSlider
        self.NSMutableAttributedString = NSMutableAttributedString
        self.NSMomentaryPushInButton = NSMomentaryPushInButton
        self.NSSwitchButton = NSSwitchButton
        self.NSView = NSView
        self.NSViewHeightSizable = NSViewHeightSizable
        self.NSViewWidthSizable = NSViewWidthSizable
        self.stop_event = stop_event
        self.settings = settings
        self.original_text = ""
        self.partial_text = ""
        self.translated_text = ""
        self.original_blocks = []
        self.translation_blocks = []
        self.history_pairs = []  # full, uncapped transcript+translation for export
        self.font_size = 20
        self.compact_mode = False
        self.pin_enabled = True
        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(NSApplicationActivationPolicyRegular)

        class MenuTarget(NSObject):
            def compactChanged_(menu_self, sender):
                self._compact_changed(sender)

            def pinChanged_(menu_self, sender):
                self._pin_changed(sender)

            def opacityChanged_(menu_self, sender):
                self._opacity_changed(sender)

            def smallerText_(menu_self, sender):
                self._adjust_text_size(-1)

            def largerText_(menu_self, sender):
                self._adjust_text_size(1)

            def sourceChanged_(menu_self, sender):
                self._source_changed(sender)

            def targetChanged_(menu_self, sender):
                self._target_changed(sender)

            def clearAll_(menu_self, sender):
                self._clear_all(sender)

            def savePDF_(menu_self, sender):
                self._save_pdf(sender)

        class WindowDelegate(NSObject):
            def windowDidResize_(delegate_self, notification):
                self._relayout()

        self.menu_target = MenuTarget.alloc().init()
        self.window_delegate = WindowDelegate.alloc().init()

        frame = NSMakeRect(120, 700, width, height)
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskResizable
            | NSWindowStyleMaskFullSizeContentView
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            NSBackingStoreBuffered,
            False,
        )
        self._layout_ready = False
        self.window.setDelegate_(self.window_delegate)
        self.window.setContentMinSize_(NSMakeSize(420, 220))
        self.window.setTitle_(title)
        self.window.setTitlebarAppearsTransparent_(True)
        self.window.setMovableByWindowBackground_(True)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setAlphaValue_(opacity)
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = self.window.contentView()
        visual = NSVisualEffectView.alloc().initWithFrame_(content.bounds())
        visual.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        visual.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        visual.setMaterial_(NSVisualEffectMaterialHUDWindow)
        visual.setState_(NSVisualEffectStateActive)
        content.addSubview_(visual)

        inset = 18
        menu_height = 70
        column_label_height = 20
        gap = 32
        column_width = int((width - inset * 2 - gap) / 2)
        scroll_height = height - inset * 2 - menu_height - column_label_height
        scroll_y = inset
        label_y = scroll_y + scroll_height + 6
        menu_y = label_y + column_label_height + 18
        status_width = 120
        target_width = 140
        source_width = 126
        control_gap = 18
        status_x = width - inset - status_width
        target_label_x = status_x - control_gap - target_width - 44
        target_popup_x = target_label_x + 44
        source_label_x = target_label_x - control_gap - source_width - 48
        source_popup_x = source_label_x + 48

        controls_x = inset
        self.compact_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 78, 28),
            "Compact",
            "compactChanged:",
            momentary=True,
        )
        controls_x += 86
        self.pin_button = self._make_button(
            NSMakeRect(controls_x, menu_y + 2, 48, 24),
            "Pin",
            "pinChanged:",
        )
        self.pin_button.setState_(NSOnState)
        controls_x += 56
        self.opacity_label = self._make_label(
            NSTextField,
            "Opacity",
            NSMakeRect(controls_x, menu_y + 5, 46, 18),
            size=12,
            alpha=0.62,
        )
        visual.addSubview_(self.opacity_label)
        controls_x += 50
        self.opacity_slider = NSSlider.alloc().initWithFrame_(
            NSMakeRect(controls_x, menu_y + 2, 86, 24)
        )
        self.opacity_slider.setMinValue_(0.35)
        self.opacity_slider.setMaxValue_(1.0)
        self.opacity_slider.setDoubleValue_(opacity)
        self.opacity_slider.setTarget_(self.menu_target)
        self.opacity_slider.setAction_("opacityChanged:")
        controls_x += 96
        self.smaller_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 32, 28),
            "A-",
            "smallerText:",
            momentary=True,
        )
        controls_x += 36
        self.larger_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 32, 28),
            "A+",
            "largerText:",
            momentary=True,
        )
        self.clear_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 40, 28),
            "Clear",
            "clearAll:",
            momentary=True,
        )
        self._set_trash_icon(self.clear_button)
        self.save_button = self._make_button(
            NSMakeRect(controls_x, menu_y, 50, 28),
            "PDF",
            "savePDF:",
            momentary=True,
        )
        for control in (
            self.compact_button,
            self.pin_button,
            self.opacity_slider,
            self.smaller_button,
            self.larger_button,
            self.clear_button,
            self.save_button,
        ):
            visual.addSubview_(control)

        self.source_popup = self._make_popup(
            NSPopUpButton,
            NSMakeRect(source_popup_x, menu_y, source_width, 28),
            LANG_MENU,
            self.settings.get()[0],
            "sourceChanged:",
        )
        self.target_popup = self._make_popup(
            NSPopUpButton,
            NSMakeRect(target_popup_x, menu_y, target_width, 28),
            [item for item in LANG_MENU if item[0] != "auto"],
            self.settings.get()[1],
            "targetChanged:",
        )

        self.source_label = self._make_label(
            NSTextField,
            "Source",
            NSMakeRect(source_label_x, menu_y + 5, 44, 18),
            size=12,
            alpha=0.72,
        )
        visual.addSubview_(self.source_label)
        visual.addSubview_(self.source_popup)
        self.target_label = self._make_label(
            NSTextField,
            "Target",
            NSMakeRect(target_label_x, menu_y + 5, 40, 18),
            size=12,
            alpha=0.72,
        )
        visual.addSubview_(self.target_label)
        visual.addSubview_(self.target_popup)
        self.status_label = self._make_label(
            NSTextField,
            "lag: 0 chunks",
            NSMakeRect(status_x, menu_y + 5, status_width, 18),
            size=12,
            alpha=0.62,
        )
        visual.addSubview_(self.status_label)

        left_x = inset
        right_x = inset + column_width + gap
        divider_x = inset + column_width + int(gap / 2)
        divider_height = scroll_height + column_label_height + 2
        self.expanded_frames = {
            "left_scroll": NSMakeRect(left_x, scroll_y, divider_x - left_x - 2, scroll_height),
            "right_scroll": NSMakeRect(right_x, scroll_y, width - right_x, scroll_height),
            "original_label": NSMakeRect(left_x, label_y, column_width, 18),
            "translation_label": NSMakeRect(right_x, label_y, column_width, 18),
        }
        self.compact_frames = {
            # Right edge flush to the window border (x == width) so the overlay
            # scrollbar tucks against the edge exactly like in expanded mode; keep the
            # left inset for text padding.
            "right_scroll": NSMakeRect(inset, scroll_y, width - inset, scroll_height),
            "translation_label": NSMakeRect(inset, label_y, width - inset * 2, 18),
        }
        self.divider_shadow = self._make_line(
            NSMakeRect(divider_x - 1, scroll_y, 1, divider_height),
            NSColor.blackColor().colorWithAlphaComponent_(0.36),
        )
        self.divider_highlight = self._make_line(
            NSMakeRect(divider_x, scroll_y, 1, divider_height),
            NSColor.whiteColor().colorWithAlphaComponent_(0.12),
        )
        visual.addSubview_(self.divider_shadow)
        visual.addSubview_(self.divider_highlight)
        self.original_label = self._make_label(
            NSTextField,
            "Original",
            self.expanded_frames["original_label"],
            size=12,
            alpha=0.62,
        )
        self.translation_label = self._make_label(
            NSTextField,
            "Translation",
            self.expanded_frames["translation_label"],
            size=12,
            alpha=0.62,
        )
        visual.addSubview_(self.original_label)
        visual.addSubview_(self.translation_label)

        self.left_scroll = NSScrollView.alloc().initWithFrame_(self.expanded_frames["left_scroll"])
        self.right_scroll = NSScrollView.alloc().initWithFrame_(self.expanded_frames["right_scroll"])
        self.left_scroll.setAutoresizingMask_(0)
        self.right_scroll.setAutoresizingMask_(0)
        self.left_scroll.setDrawsBackground_(False)
        self.right_scroll.setDrawsBackground_(False)
        self.left_scroll.setHasVerticalScroller_(True)
        self.right_scroll.setHasVerticalScroller_(True)
        # Thin, modern overlay scrollbars with a light knob (for the dark glass), that
        # float at the very edge and auto-hide when idle.
        try:
            from AppKit import NSScrollerKnobStyleLight, NSScrollerStyleOverlay

            for scroll in (self.left_scroll, self.right_scroll):
                scroll.setScrollerStyle_(NSScrollerStyleOverlay)
                scroll.setAutohidesScrollers_(True)
                scroller = scroll.verticalScroller()
                if scroller is not None:
                    scroller.setKnobStyle_(NSScrollerKnobStyleLight)
        except Exception:
            pass
        visual.addSubview_(self.left_scroll)
        visual.addSubview_(self.right_scroll)

        self.original_view = self._make_text_view(NSTextView, self.left_scroll)
        self.translated_view = self._make_text_view(NSTextView, self.right_scroll)
        self.original_view.setString_(WAITING_ORIGINAL)
        self.translated_view.setString_(self._waiting_translation())
        self.left_scroll.setDocumentView_(self.original_view)
        self.right_scroll.setDocumentView_(self.translated_view)

        self.NSApp = NSApp
        self._layout_ready = True
        self._relayout()

    def _relayout(self):
        if not getattr(self, "_layout_ready", False):
            return
        size = self.window.contentView().bounds().size
        width = size.width
        height = size.height
        inset = 18
        menu_height = 70
        column_label_height = 20
        gap = 32
        column_width = int((width - inset * 2 - gap) / 2)
        scroll_height = height - inset * 2 - menu_height - column_label_height
        if scroll_height < 1:
            scroll_height = 1
        scroll_y = inset
        label_y = scroll_y + scroll_height + 6
        menu_y = label_y + column_label_height + 18
        status_width = 120
        target_width = 140
        source_width = 126
        control_gap = 18
        save_x = width - inset - 50
        clear_x = save_x - 10 - 40
        status_x = clear_x - control_gap - status_width
        target_label_x = status_x - control_gap - target_width - 44
        target_popup_x = target_label_x + 44
        source_label_x = target_label_x - control_gap - source_width - 48
        source_popup_x = source_label_x + 48

        controls_x = inset
        self.compact_button.setFrame_(self.NSMakeRect(controls_x, menu_y, 78, 28))
        controls_x += 86
        self.pin_button.setFrame_(self.NSMakeRect(controls_x, menu_y + 2, 48, 24))
        controls_x += 56
        self.opacity_label.setFrame_(self.NSMakeRect(controls_x, menu_y + 5, 46, 18))
        controls_x += 50
        self.opacity_slider.setFrame_(self.NSMakeRect(controls_x, menu_y + 2, 86, 24))
        controls_x += 96
        self.smaller_button.setFrame_(self.NSMakeRect(controls_x, menu_y, 32, 28))
        controls_x += 36
        self.larger_button.setFrame_(self.NSMakeRect(controls_x, menu_y, 32, 28))

        self.save_button.setFrame_(self.NSMakeRect(save_x, menu_y, 50, 28))
        self.clear_button.setFrame_(self.NSMakeRect(clear_x, menu_y, 40, 28))
        self.source_label.setFrame_(self.NSMakeRect(source_label_x, menu_y + 5, 44, 18))
        self.source_popup.setFrame_(self.NSMakeRect(source_popup_x, menu_y, source_width, 28))
        self.target_label.setFrame_(self.NSMakeRect(target_label_x, menu_y + 5, 40, 18))
        self.target_popup.setFrame_(self.NSMakeRect(target_popup_x, menu_y, target_width, 28))
        self.status_label.setFrame_(self.NSMakeRect(status_x, menu_y + 5, status_width, 18))

        # Right-side controls hide progressively when the window is too narrow, so they
        # never overlap the left cluster (Compact/Pin/Opacity/A-/A+). Clear/Save are
        # right-anchored and the last to go; Source is the first.
        left_end = inset + 86 + 56 + 50 + 96 + 36 + 32  # right edge of left cluster
        edge = left_end + 12
        show_save = save_x >= edge
        show_clear = clear_x >= edge
        show_status = status_x >= edge
        show_target = target_label_x >= edge
        show_source = source_label_x >= edge
        self.save_button.setHidden_(not show_save)
        self.clear_button.setHidden_(not show_clear)
        self.status_label.setHidden_(not show_status)
        self.target_label.setHidden_(not show_target)
        self.target_popup.setHidden_(not show_target)
        self.source_label.setHidden_(not show_source)
        self.source_popup.setHidden_(not show_source)

        left_x = inset
        right_x = inset + column_width + gap
        divider_x = inset + column_width + int(gap / 2)
        divider_height = scroll_height + column_label_height + 2
        self.expanded_frames = {
            "left_scroll": self.NSMakeRect(left_x, scroll_y, divider_x - left_x - 2, scroll_height),
            "right_scroll": self.NSMakeRect(right_x, scroll_y, width - right_x, scroll_height),
            "original_label": self.NSMakeRect(left_x, label_y, column_width, 18),
            "translation_label": self.NSMakeRect(right_x, label_y, column_width, 18),
        }
        self.compact_frames = {
            # Right edge flush to the window border (x == width) so the overlay
            # scrollbar tucks against the edge exactly like in expanded mode; keep the
            # left inset for text padding.
            "right_scroll": self.NSMakeRect(inset, scroll_y, width - inset, scroll_height),
            "translation_label": self.NSMakeRect(inset, label_y, width - inset * 2, 18),
        }
        self.divider_shadow.setFrame_(self.NSMakeRect(divider_x - 1, scroll_y, 1, divider_height))
        self.divider_highlight.setFrame_(self.NSMakeRect(divider_x, scroll_y, 1, divider_height))
        self.original_label.setFrame_(self.expanded_frames["original_label"])
        self.left_scroll.setFrame_(self.expanded_frames["left_scroll"])
        if self.compact_mode:
            self.right_scroll.setFrame_(self.compact_frames["right_scroll"])
            self.translation_label.setFrame_(self.compact_frames["translation_label"])
        else:
            self.right_scroll.setFrame_(self.expanded_frames["right_scroll"])
            self.translation_label.setFrame_(self.expanded_frames["translation_label"])
        self.original_view.setFrame_(self.left_scroll.contentView().bounds())
        self.translated_view.setFrame_(self.right_scroll.contentView().bounds())
        # Re-flow text into the new frames; without this the resized text views go blank.
        self._render_original()
        self._render_translation()

    def _make_label(self, text_field_cls, value, frame, size, alpha):
        label = text_field_cls.alloc().initWithFrame_(frame)
        label.setStringValue_(value)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setTextColor_(self.NSColor.whiteColor().colorWithAlphaComponent_(alpha))
        label.setFont_(self.NSFont.systemFontOfSize_weight_(size, 0.18))
        return label

    def _make_button(self, frame, title, action, momentary=False):
        button = self.NSButton.alloc().initWithFrame_(frame)
        button.setTitle_(title)
        button.setButtonType_(self.NSMomentaryPushInButton if momentary else self.NSSwitchButton)
        button.setBezelStyle_(1)
        button.setTarget_(self.menu_target)
        button.setAction_(action)
        button.setFont_(self.NSFont.systemFontOfSize_weight_(12, 0.18))
        return button

    def _set_trash_icon(self, button):
        # Native vector trash glyph (SF Symbol), tinted white to match the toolbar.
        try:
            from AppKit import NSImageOnly, NSImage

            img = NSImage.imageWithSystemSymbolName_accessibilityDescription_("trash", "Clear")
            if img is not None:
                button.setImage_(img)
                button.setImagePosition_(NSImageOnly)
                button.setTitle_("")
                try:
                    button.setContentTintColor_(self.NSColor.whiteColor())
                except Exception:
                    pass
        except Exception:
            pass  # fall back to the "Clear" text title

    def _make_line(self, frame, color):
        line = self.NSView.alloc().initWithFrame_(frame)
        line.setWantsLayer_(True)
        line.layer().setBackgroundColor_(color.CGColor())
        return line

    def _make_popup(self, popup_cls, frame, items, selected_code, action):
        popup = popup_cls.alloc().initWithFrame_pullsDown_(frame, False)
        for code, label in items:
            popup.addItemWithTitle_(label)
            popup.itemWithTitle_(label).setRepresentedObject_(code)
        popup.selectItemWithTitle_(language_label(selected_code))
        popup.setTarget_(self.menu_target)
        popup.setAction_(action)
        return popup

    def _make_text_view(self, text_view_cls, scroll):
        view = text_view_cls.alloc().initWithFrame_(scroll.contentView().bounds())
        view.setAutoresizingMask_(self.NSViewWidthSizable | self.NSViewHeightSizable)
        view.setEditable_(False)
        view.setSelectable_(True)
        view.setDrawsBackground_(False)
        view.setTextColor_(self.NSColor.whiteColor())
        view.setFont_(self.NSFont.systemFontOfSize_weight_(20, 0.22))
        # Padding so text doesn't butt against the divider / window edge.
        view.setTextContainerInset_(self.NSMakeSize(10, 6))
        return view

    def _selected_code(self, popup):
        item = popup.selectedItem()
        code = item.representedObject() if item else None
        return str(code) if code else "auto"

    def _source_changed(self, sender):
        self.settings.set_source(self._selected_code(sender))

    def _waiting_translation(self):
        try:
            code = self.settings.get()[1]
        except Exception:
            code = "en"
        return WAITING_TRANSLATION.get(code, WAITING_TRANSLATION["en"])

    def _target_changed(self, sender):
        code = self._selected_code(sender)
        if code != "auto":
            self.settings.set_target(code)
        self._render_translation()  # refresh placeholder language if empty

    def _compact_changed(self, sender):
        self.compact_mode = not self.compact_mode
        self.left_scroll.setHidden_(self.compact_mode)
        self.original_label.setHidden_(self.compact_mode)
        self.divider_shadow.setHidden_(self.compact_mode)
        self.divider_highlight.setHidden_(self.compact_mode)
        if self.compact_mode:
            self.right_scroll.setFrame_(self.compact_frames["right_scroll"])
            self.translation_label.setFrame_(self.compact_frames["translation_label"])
            self.compact_button.setTitle_("Expanded")
        else:
            self.right_scroll.setFrame_(self.expanded_frames["right_scroll"])
            self.translation_label.setFrame_(self.expanded_frames["translation_label"])
            self.compact_button.setTitle_("Compact")
        self.translated_view.setFrame_(self.right_scroll.contentView().bounds())
        self._render_translation()

    def _pin_changed(self, sender):
        self.pin_enabled = sender.state() == self.NSOnState
        self.window.setLevel_(self.NSFloatingWindowLevel if self.pin_enabled else self.NSNormalWindowLevel)

    def _opacity_changed(self, sender):
        self.window.setAlphaValue_(sender.doubleValue())

    def _adjust_text_size(self, delta):
        self.font_size = max(14, min(32, self.font_size + delta))
        self._render_original()
        self._render_translation()

    def _clear_all(self, sender=None):
        self.original_blocks = []
        self.translation_blocks = []
        self.history_pairs = []
        self.partial_text = ""
        # Also wipe both workers' rolling state (audio buffer, tail, sentence buffer,
        # Whisper context) and pending queues so it restarts from a clean slate.
        reset_gen = getattr(self, "reset_gen", None)
        if reset_gen is not None:
            reset_gen[0] += 1
        self._render_original()
        self._render_translation()

    def _notify(self, title, message):
        try:
            from AppKit import NSAlert

            alert = NSAlert.alloc().init()
            alert.setMessageText_(str(title))
            alert.setInformativeText_(str(message))
            alert.runModal()
        except Exception:
            print(f"[{title}] {message}", file=sys.stderr)

    def _save_pdf(self, sender=None):
        import datetime

        from AppKit import NSSavePanel

        if not self.history_pairs:
            self._notify("Нечего сохранять", "Пока нет переведённого текста.")
            return

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        panel = NSSavePanel.savePanel()
        panel.setNameFieldStringValue_(f"LiveTranslate-{ts}.pdf")
        try:
            panel.setAllowedFileTypes_(["pdf"])
        except Exception:
            pass
        self.app.activateIgnoringOtherApps_(True)
        if panel.runModal() != 1:  # 1 == NSModalResponseOK
            return
        path = panel.URL().path()

        from AppKit import (
            NSGradient,
            NSKernAttributeName,
            NSMutableParagraphStyle,
            NSParagraphStyleAttributeName,
        )

        NSColor = self.NSColor
        NSFont = self.NSFont
        NSFontAttr = self.NSFontAttributeName
        NSFgAttr = self.NSForegroundColorAttributeName

        # --- "liquid glass" palette --------------------------------------
        page_width = 612.0          # US Letter width
        outer = 46.0                # page margin
        card_gap = 14.0             # vertical gap between cards
        pad_x, pad_top, pad_bottom = 26.0, 18.0, 20.0
        inner_w = page_width - outer * 2 - pad_x * 2

        c_title = NSColor.colorWithCalibratedWhite_alpha_(0.13, 1.0)
        c_subtitle = NSColor.colorWithCalibratedWhite_alpha_(0.45, 1.0)
        c_label = NSColor.colorWithCalibratedWhite_alpha_(0.50, 1.0)
        c_src = NSColor.colorWithCalibratedWhite_alpha_(0.32, 1.0)
        c_tr = NSColor.colorWithCalibratedWhite_alpha_(0.13, 1.0)
        c_accent = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.20, 0.36, 0.92, 1.0)

        gradient = NSGradient.alloc().initWithColors_(
            [
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.88, 0.92, 0.99, 1.0),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.93, 0.91, 0.99, 1.0),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.99, 0.94, 0.96, 1.0),
            ]
        )

        title_font = NSFont.systemFontOfSize_weight_(26.0, 0.32)
        subtitle_font = NSFont.systemFontOfSize_(12.0)
        label_font = NSFont.systemFontOfSize_weight_(9.5, 0.34)
        body_font = NSFont.systemFontOfSize_(13.0)

        def para(line_spacing=0.0, after=0.0, before=0.0):
            p = NSMutableParagraphStyle.alloc().init()
            p.setLineSpacing_(line_spacing)
            p.setParagraphSpacing_(after)
            p.setParagraphSpacingBefore_(before)
            return p

        ps_body = para(line_spacing=3.5)
        ps_label = para(after=3.0)
        ps_label_before = para(after=3.0, before=15.0)
        ps_title = para(after=4.0)

        def piece(text, font, color, ps, kern=None):
            attrs = {NSFgAttr: color, NSFontAttr: font, NSParagraphStyleAttributeName: ps}
            if kern is not None:
                attrs[NSKernAttributeName] = kern
            return self.NSMutableAttributedString.alloc().initWithString_attributes_(text, attrs)

        from AppKit import NSTextView

        # Render text via NSTextView (TextKit) so what we measure is exactly what
        # gets drawn — boundingRectWithSize mis-measures paragraphSpacingBefore and
        # silently clips the last line on short blocks.
        def make_tv(attr, width):
            tv = NSTextView.alloc().initWithFrame_(((0, 0), (width, 1.0e6)))
            tv.setDrawsBackground_(False)
            tv.setEditable_(False)
            tv.setSelectable_(False)
            tv.textContainer().setLineFragmentPadding_(0.0)
            tv.setTextContainerInset_((0.0, 0.0))
            tv.textStorage().setAttributedString_(attr)
            lm, tc = tv.layoutManager(), tv.textContainer()
            lm.ensureLayoutForTextContainer_(tc)
            h = float(lm.usedRectForTextContainer_(tc).size.height)
            return tv, h

        # --- header ------------------------------------------------------
        human = datetime.datetime.now().strftime("%d %B %Y · %H:%M")
        header = self.NSMutableAttributedString.alloc().init()
        header.appendAttributedString_(piece("Live Translation\n", title_font, c_title, ps_title))
        header.appendAttributedString_(piece(human, subtitle_font, c_subtitle, para()))
        header_w = page_width - outer * 2
        header_tv, header_h = make_tv(header, header_w)
        header_gap = 20.0

        # --- build one card per pair ------------------------------------
        cards = []  # (text_view, content_h, card_h)
        for pair in self.history_pairs:
            src = (pair.get("source") or "").strip()
            tr = (pair.get("translated") or "").strip()
            if not src and not tr:
                continue
            content = self.NSMutableAttributedString.alloc().init()
            if src:
                content.appendAttributedString_(
                    piece("ORIGINAL\n", label_font, c_label, ps_label, kern=1.3)
                )
                content.appendAttributedString_(
                    piece(src + ("\n" if tr else ""), body_font, c_src, ps_body)
                )
            if tr:
                lbl_ps = ps_label_before if src else ps_label
                content.appendAttributedString_(
                    piece("TRANSLATION\n", label_font, c_accent, lbl_ps, kern=1.3)
                )
                content.appendAttributedString_(piece(tr, body_font, c_tr, ps_body))
            tv, content_h = make_tv(content, inner_w)
            cards.append((tv, content_h, pad_top + content_h + pad_bottom))

        if not cards:
            self._notify("Нечего сохранять", "Пока нет переведённого текста.")
            return

        total_h = (
            outer
            + header_h
            + header_gap
            + sum(card_h for _, _, card_h in cards)
            + card_gap * (len(cards) - 1)
            + outer
        )

        ViewClass = _glass_pdf_view_class()
        view = ViewClass.alloc().initWithFrame_(((0, 0), (page_width, total_h)))
        view._gradient = gradient
        view._texts = []  # text is drawn by NSTextView subviews, not by the view

        # --- lay out (flipped view: y grows downward from the top) ------
        card_rects = []
        header_tv.setFrame_(((outer, outer), (header_w, header_h)))
        view.addSubview_(header_tv)
        y = outer + header_h + header_gap
        for tv, content_h, card_h in cards:
            card_rects.append(((outer, y), (page_width - outer * 2, card_h)))
            tv.setFrame_(((outer + pad_x, y + pad_top), (inner_w, content_h)))
            view.addSubview_(tv)
            y += card_h + card_gap
        view._cards = card_rects

        data = view.dataWithPDFInsideRect_(view.bounds())
        ok = bool(data) and bool(data.writeToFile_atomically_(path, True))
        self._notify(
            "PDF сохранён" if ok else "Ошибка",
            path if ok else "Не удалось записать файл.",
        )

    def post_pair(self, source, translated, pause_ms=0):
        self.AppHelper.callAfter(self._append_pair, source, translated, pause_ms)

    def post_partial(self, source):
        self.AppHelper.callAfter(self._set_partial, source)

    def post_status(self, lag_chunks):
        self.AppHelper.callAfter(self._set_status, lag_chunks)

    def _set_status(self, lag_chunks):
        self.status_label.setStringValue_(f"lag: {lag_chunks} chunks")
        alpha = 0.9 if lag_chunks else 0.62
        self.status_label.setTextColor_(self.NSColor.whiteColor().colorWithAlphaComponent_(alpha))

    def _append_pair(self, source, translated, pause_ms=0):
        # Guard the main-thread render path: an exception here can stop the run loop from
        # delivering further UI updates, freezing the display while workers keep running.
        try:
            now = time.monotonic()
            speaker_gap = pause_ms >= 1400
            if source and not str(translated).startswith("Ошибка:"):
                self.history_pairs.append({"source": source, "translated": translated or ""})
                if len(self.history_pairs) > 5000:
                    self.history_pairs = self.history_pairs[-5000:]
            if source:
                self.original_blocks.append({"text": source, "time": now, "gap": speaker_gap})
                self.original_blocks = self.original_blocks[-16:]
                self.partial_text = ""
                self._render_original()
            if translated:
                self.translation_blocks.append({"text": translated, "time": now, "gap": speaker_gap})
                self.translation_blocks = self.translation_blocks[-16:]
                self._render_translation()
        except Exception as exc:
            print(f"[ui] _append_pair: {exc!r}", file=sys.stderr)

    def _set_partial(self, source):
        try:
            partial = source.strip()
            if len(partial) > 700:
                partial = partial[-700:].lstrip()
            self.partial_text = partial
            self._render_original()
        except Exception as exc:
            print(f"[ui] _set_partial: {exc!r}", file=sys.stderr)

    def _render_original(self):
        full_text, spans = self._compose_blocks(self.original_blocks)
        partial = self.partial_text.strip()
        if partial:
            if full_text and not full_text.endswith("\n\n"):
                full_text += "\n\n"
            full_text += partial

        if not full_text:
            full_text = WAITING_ORIGINAL

        font = self.NSFont.systemFontOfSize_weight_(self.font_size, 0.22)
        committed_attrs = {
            self.NSForegroundColorAttributeName: self.NSColor.whiteColor(),
            self.NSFontAttributeName: font,
        }
        attributed = self.NSMutableAttributedString.alloc().initWithString_attributes_(
            full_text,
            committed_attrs,
        )
        self._apply_block_fade(attributed, spans, font)
        if partial:
            partial_start = len(full_text) - len(partial)
            partial_attrs = {
                self.NSForegroundColorAttributeName: self.NSColor.whiteColor().colorWithAlphaComponent_(0.32),
                self.NSFontAttributeName: font,
            }
            attributed.addAttributes_range_(
                partial_attrs,
                self.NSMakeRange(partial_start, len(partial)),
            )
        self.original_view.textStorage().setAttributedString_(attributed)
        self.original_view.scrollRangeToVisible_(self.NSMakeRange(len(full_text), 0))

    def _render_translation(self):
        full_text, spans = self._compose_blocks(self.translation_blocks)
        if not full_text:
            full_text = self._waiting_translation()
        font = self.NSFont.systemFontOfSize_weight_(self.font_size, 0.22)
        attrs = {
            self.NSForegroundColorAttributeName: self.NSColor.whiteColor(),
            self.NSFontAttributeName: font,
        }
        attributed = self.NSMutableAttributedString.alloc().initWithString_attributes_(
            full_text,
            attrs,
        )
        self._apply_block_fade(attributed, spans, font)
        self.translated_view.textStorage().setAttributedString_(attributed)
        self.translated_view.scrollRangeToVisible_(self.NSMakeRange(len(full_text), 0))

    def _compose_blocks(self, blocks):
        chunks = []
        spans = []
        pos = 0
        for idx, block in enumerate(blocks):
            sep = "\n\n\n" if block.get("gap") and chunks else ("\n\n" if chunks else "")
            chunks.append(sep)
            pos += len(sep)
            text = block["text"].strip()
            chunks.append(text)
            spans.append((pos, len(text), idx))
            pos += len(text)
        return "".join(chunks), spans

    def _apply_block_fade(self, attributed, spans, font):
        total = len(spans)
        for start, length, idx in spans:
            age_from_end = total - idx - 1
            alpha = 1.0 if age_from_end < 3 else max(0.48, 0.82 - age_from_end * 0.06)
            attrs = {
                self.NSForegroundColorAttributeName: self.NSColor.whiteColor().colorWithAlphaComponent_(alpha),
                self.NSFontAttributeName: font,
            }
            attributed.addAttributes_range_(attrs, self.NSMakeRange(start, length))

    def run(self):
        self.window.makeKeyAndOrderFront_(None)
        self.app.activateIgnoringOtherApps_(True)
        self.AppHelper.runEventLoop()

    def stop(self):
        self.stop_event.set()
        self.AppHelper.callAfter(self.AppHelper.stopEventLoop)


def put_drop_oldest(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def put_wait(q, item, stop_event):
    while not stop_event.is_set():
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def load_vad():
    """Lazily load Silero VAD. Returns (model, get_speech_timestamps) or (None, None)
    so the pipeline degrades gracefully to fixed-size chunking if VAD is unavailable."""
    try:
        import torch  # noqa: F401
        from silero_vad import get_speech_timestamps, load_silero_vad

        return load_silero_vad(), get_speech_timestamps
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[VAD] недоступен, фиксированная нарезка ({exc})", file=sys.stderr)
        return None, None


def _to_mono_16k(audio, samplerate):
    import torch
    import torchaudio

    a = audio.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    tensor = torch.from_numpy(np.ascontiguousarray(a))
    if int(samplerate) != 16000:
        tensor = torchaudio.functional.resample(tensor, int(samplerate), 16000)
    return tensor


def vad_analyze(audio, samplerate, vad_model, get_speech_timestamps, min_keep_frames, min_speech_ms=250):
    """Analyze a chunk with Silero VAD. Returns (has_speech, cut):
    - has_speech: whether enough *real* speech was detected. We require the total speech
      duration to reach min_speech_ms — a stray blip/noise that Silero marks as a tiny
      speech island won't pass, which keeps short bursts of noise out of Whisper.
    - cut: a sample index (original samplerate) to cut at, in the silence right after the
      last speech (never mid-word), or None to keep the fixed-size cut.
    On any error returns (True, None) so the caller falls back to the RMS gate."""
    try:
        wav = _to_mono_16k(audio, samplerate)
        timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    except Exception:
        return True, None
    if not timestamps:
        return False, None
    speech_16k = sum(int(t["end"]) - int(t["start"]) for t in timestamps)
    if speech_16k < int(min_speech_ms / 1000.0 * 16000):
        return False, None
    total_16k = int(wav.shape[-1])
    last_end_16k = int(timestamps[-1]["end"])
    # Require real trailing silence after the last speech, otherwise we'd risk a mid-word cut.
    if total_16k - last_end_16k < int(0.10 * 16000):
        return True, None
    cut = int(last_end_16k * samplerate / 16000)
    if cut < min_keep_frames:
        return True, None
    return True, cut


# ---------------------------------------------------------------------------
# LocalAgreement-2 streaming (Whisper-Streaming style). Instead of cutting audio
# into independent chunks, we keep a rolling buffer, re-transcribe it on each
# update, and only *commit* the longest common prefix that two consecutive
# transcriptions agree on. Committed text never changes; the unconfirmed tail is
# shown as a live draft. This removes word breaks, dupes and false periods at
# chunk boundaries that plain chunking produced.
# ---------------------------------------------------------------------------

def _norm_word(text):
    return re.sub(r"[^\w'’-]", "", text).casefold()


class HypothesisBuffer:
    def __init__(self):
        self.committed_in_buffer = []  # (start, end, text) absolute-time words
        self.buffer = []               # previous hypothesis tail (unconfirmed)
        self.new = []
        self.last_committed_time = 0.0

    def insert(self, words, offset):
        words = [(s + offset, e + offset, t) for (s, e, t) in words]
        self.new = [w for w in words if w[0] > self.last_committed_time - 0.1]

    def flush(self):
        # Commit the longest common prefix between the new and the previous hypothesis.
        commit = []
        while self.new:
            ns, ne, nt = self.new[0]
            if not self.buffer:
                break
            if _norm_word(nt) and _norm_word(nt) == _norm_word(self.buffer[0][2]):
                commit.append((ns, ne, nt))
                self.last_committed_time = ne
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.committed_in_buffer.extend(commit)
        return commit


class OnlineProcessor:
    def __init__(self, samplerate, max_active_seconds=14.0):
        self.sr = samplerate
        self.audio = None
        self.buffer_offset = 0.0  # seconds of audio already trimmed away
        self.hyp = HypothesisBuffer()
        self.max_active = max_active_seconds

    def active_seconds(self):
        return 0.0 if self.audio is None else len(self.audio) / self.sr

    def insert_audio(self, block):
        self.audio = block if self.audio is None else np.concatenate((self.audio, block), axis=0)

    def reset(self):
        self.audio = None
        self.buffer_offset = 0.0
        self.hyp = HypothesisBuffer()

    def finalize(self):
        # Utterance ended (silence): there won't be another pass to agree with, so accept
        # the current unconfirmed draft as final instead of dropping it. Returns its text.
        tail = " ".join(t for (_, _, t) in self.hyp.buffer).strip()
        self.hyp.committed_in_buffer.extend(self.hyp.buffer)
        self.hyp.buffer = []
        self.hyp.new = []
        return tail

    def drop_active(self):
        # Discard buffered (silent) audio without committing anything.
        if self.audio is not None:
            self.buffer_offset += len(self.audio) / self.sr
            self.audio = None
        self.hyp.buffer = []
        self.hyp.new = []

    def _trim_to(self, t):
        cut_idx = int((t - self.buffer_offset) * self.sr)
        if self.audio is not None and 0 < cut_idx < len(self.audio):
            self.audio = self.audio[cut_idx:]
            self.buffer_offset = t
        self.hyp.committed_in_buffer = [
            w for w in self.hyp.committed_in_buffer if w[1] >= t - 30
        ]

    def process(self, transcribe_words_fn):
        """Returns (committed_words, draft_text)."""
        words = transcribe_words_fn(self.audio)
        self.hyp.insert(words, self.buffer_offset)
        committed = self.hyp.flush()
        if committed:
            self._trim_to(self.hyp.last_committed_time)
        elif self.active_seconds() > self.max_active:
            # Unstable for too long — force the buffer down so we never blow past
            # Whisper's window or transcribe an ever-growing clip.
            self._trim_to(self.buffer_offset + self.active_seconds() - self.max_active * 0.6)
        draft = " ".join(t for (_, _, t) in self.hyp.buffer)
        return committed, draft


def audio_capture_worker(audio_q, stop_event, device, samplerate, channels, block_seconds):
    blocksize = max(256, int(samplerate * block_seconds))
    last_audio = {"t": time.monotonic()}

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        last_audio["t"] = time.monotonic()
        put_drop_oldest(audio_q, indata.copy())

    def open_stream():
        stream = sd.InputStream(
            samplerate=samplerate,
            device=device,
            channels=channels,
            blocksize=blocksize,
            callback=callback,
        )
        stream.start()
        return stream

    # The audio device keeps delivering frames (even silence) on its own clock, so a
    # gap with no callback means the stream stalled — e.g. CoreAudio re-inits the route
    # when a YouTube video switches. Watch for that and transparently reopen the stream.
    stream = None
    while not stop_event.is_set():
        try:
            if stream is None:
                stream = open_stream()
                last_audio["t"] = time.monotonic()
            stalled = (time.monotonic() - last_audio["t"]) > 3.0
            inactive = not stream.active
            if stalled or inactive:
                print("[audio] поток залип — перезапуск", file=sys.stderr)
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                stream = None
                time.sleep(0.3)
                continue
        except Exception as exc:
            print(f"[audio] ошибка потока: {exc!r} — перезапуск", file=sys.stderr)
            try:
                if stream is not None:
                    stream.stop()
                    stream.close()
            except Exception:
                pass
            stream = None
            time.sleep(0.5)
            continue
        time.sleep(0.2)

    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def chunk_worker(
    audio_q,
    chunk_q,
    overlay,
    stop_event,
    samplerate,
    chunk_seconds,
    overlap_seconds,
    endpointing_ms,
    silence_rms,
    use_vad=True,
    min_chunk_seconds=1.2,
    pause_emit_ms=350.0,
    reset_gen=None,
    vad_min_speech_ms=250.0,
):
    # chunk_seconds is the MAX audio window; we emit earlier whenever the speaker
    # makes a natural pause (>= pause_emit_ms) after at least min_chunk_seconds. This
    # keeps latency low while giving Whisper longer, sentence-complete context on
    # run-on speech (short fixed chunks made Whisper end every piece with a period).
    frames_needed = int(samplerate * chunk_seconds)
    min_chunk_frames = int(samplerate * min_chunk_seconds)
    overlap_frames = max(0, min(int(samplerate * overlap_seconds), frames_needed - 1))
    vad_model, get_speech_timestamps = load_vad() if use_vad else (None, None)
    min_keep_frames = int(samplerate * min(1.0, min_chunk_seconds * 0.5))
    buffer = np.empty((0, 1), dtype=np.float32)
    channel_count = None
    trailing_silence_ms = 0.0
    endpoint_sent = False
    frames = 0
    seen_gen = reset_gen[0] if reset_gen is not None else 0

    while not stop_event.is_set():
        if reset_gen is not None and reset_gen[0] != seen_gen:
            # Clear button: drop buffered audio so old speech can't leak through.
            seen_gen = reset_gen[0]
            buffer = np.empty((0, channel_count or 1), dtype=np.float32)
            trailing_silence_ms = 0.0
            endpoint_sent = False
            while True:
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    break
        try:
            block = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            block_rms = float(np.sqrt(np.mean(np.square(block.astype(np.float32)))))
            if block_rms >= silence_rms:
                trailing_silence_ms = 0.0
                endpoint_sent = False
            else:
                trailing_silence_ms += len(block) / samplerate * 1000.0
                if trailing_silence_ms >= endpointing_ms and not endpoint_sent:
                    endpoint_sent = True

            if channel_count is None:
                channel_count = block.shape[1] if block.ndim > 1 else 1
                buffer = np.empty((0, channel_count), dtype=block.dtype)
            buffer = np.concatenate((buffer, block), axis=0)
            frames = len(buffer)
            pause_now = frames >= min_chunk_frames and trailing_silence_ms >= pause_emit_ms
            if frames < frames_needed and not pause_now:
                continue

            cut = min(frames, frames_needed)
            has_speech = True
            if vad_model is not None:
                has_speech, vad_cut = vad_analyze(
                    buffer[:cut],
                    samplerate,
                    vad_model,
                    get_speech_timestamps,
                    min_keep_frames,
                    vad_min_speech_ms,
                )
                if vad_cut is not None:
                    cut = vad_cut

            audio = buffer[:cut].copy()
            tail_start = max(0, cut - overlap_frames) if overlap_frames else cut
            buffer = buffer[tail_start:]
            frames = len(buffer)

            # Don't send silence/music to Whisper: VAD says no speech (or RMS too low when
            # VAD is off). Feeding long pauses to Whisper is what stalled the pipeline.
            if not has_speech:
                continue
            rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))
            if rms < silence_rms:
                continue
            # No-loss: block until there's room rather than dropping a chunk. The transcribe
            # worker keeps up (each chunk is transcribed exactly once), so this rarely waits.
            put_wait(
                chunk_q,
                {
                    "audio": audio,
                    "endpoint": trailing_silence_ms >= endpointing_ms,
                    "trailing_silence_ms": trailing_silence_ms,
                },
                stop_event,
            )
            overlay.post_status(chunk_q.qsize())
        except Exception as exc:
            # Never let the chunk thread die silently — that permanently stops transcription.
            print(f"[chunk] ошибка: {exc!r} — продолжаю", file=sys.stderr)
            buffer = np.empty((0, channel_count or 1), dtype=np.float32)
            trailing_silence_ms = 0.0
            time.sleep(0.1)
            continue


def transcribe_translate_worker(
    chunk_q,
    overlay,
    translator,
    stop_event,
    samplerate,
    whisper_model,
    settings,
    sentence_mode,
    flush_after,
    min_block_chars,
    max_block_chars,
    max_sentences,
    mutable_tail_seconds,
    endpointing_ms,
    endpoint_min_words,
    max_partial_seconds,
    word_timestamps,
    reset_gen=None,
):
    import mlx_whisper

    last_text = ""
    tail_history = []
    sentence_buffer = ""
    buffer_started_at = None
    recent_context = ""  # last committed/emitted text, fed to Whisper as initial_prompt
    seen_gen = reset_gen[0] if reset_gen is not None else 0

    def note_context(text):
        # Keep the most recent ~240 chars of finalized text to condition the next chunk.
        nonlocal recent_context
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            recent_context = f"{recent_context} {text}".strip()[-240:]

    def refresh_tail(now):
        nonlocal tail_history
        tail_history = [
            (stamp, value)
            for stamp, value in tail_history
            if now - stamp <= mutable_tail_seconds
        ]
        return " ".join(value for _, value in tail_history)[-600:]

    def finalize_endpoint(pause_ms=0):
        nonlocal sentence_buffer, buffer_started_at
        blocks, sentence_buffer = take_endpoint_blocks(
            sentence_buffer,
            min_chars=min_block_chars,
            max_chars=max_block_chars,
            min_words=endpoint_min_words,
        )
        for source_text in blocks:
            source_language, target_language = settings.get()
            translator.set_target(target_language)
            source_text = sentence_case_text(source_text)
            translated = translator.translate(source_text)
            if translated:
                overlay.post_pair(source_text, translated, pause_ms=pause_ms)
                note_context(source_text)
        overlay.post_partial(sentence_case_text(strip_to_recent_text(sentence_buffer, 700)))
        buffer_started_at = time.monotonic() if sentence_buffer else None

    while not stop_event.is_set():
        if reset_gen is not None and reset_gen[0] != seen_gen:
            # Clear button pressed: wipe all rolling state and pending chunks so
            # transcription/translation truly start from a clean slate.
            seen_gen = reset_gen[0]
            last_text = ""
            tail_history.clear()
            sentence_buffer = ""
            buffer_started_at = None
            recent_context = ""
            while True:
                try:
                    chunk_q.get_nowait()
                except queue.Empty:
                    break

        try:
            item = chunk_q.get(timeout=0.2)
        except queue.Empty:
            continue
        overlay.post_status(chunk_q.qsize())

        if isinstance(item, dict):
            audio = item.get("audio")
            is_endpoint = bool(item.get("endpoint"))
            trailing_silence_ms = float(item.get("trailing_silence_ms") or 0.0)
        else:
            audio = item
            is_endpoint = False
            trailing_silence_ms = 0.0

        if audio is None:
            if is_endpoint and sentence_mode:
                finalize_endpoint(pause_ms=trailing_silence_ms)
            continue

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="live_translate_")
            os.close(fd)
            sf.write(temp_path, audio, int(samplerate), subtype="PCM_16")
            source_language, target_language = settings.get()
            # Feed the recently finalized text as initial_prompt: gives cross-chunk
            # context (punctuation, casing, word continuity at boundaries) without the
            # runaway-hallucination risk of condition_on_previous_text=True. Skip it
            # when that context has no punctuation, or it reinforces Whisper's drift
            # into never punctuating (see punctuated_context).
            context_prompt = punctuated_context(f"{recent_context} {sentence_buffer}")
            gen_at_start = reset_gen[0] if reset_gen is not None else seen_gen
            result = mlx_whisper.transcribe(
                temp_path,
                path_or_hf_repo=whisper_model,
                language=None if source_language == "auto" else source_language,
                condition_on_previous_text=False,
                initial_prompt=context_prompt,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                temperature=(0.0, 0.2, 0.4),
                word_timestamps=word_timestamps,
                verbose=False,
            )
            if reset_gen is not None and reset_gen[0] != gen_at_start:
                # Clear was pressed while this chunk was inside Whisper — drop its result.
                continue
            if word_timestamps:
                last_word_end = last_word_end_seconds(result)
                if last_word_end is not None:
                    audio_duration = len(audio) / samplerate
                    word_gap_ms = max(0.0, (audio_duration - last_word_end) * 1000.0)
                    trailing_silence_ms = max(trailing_silence_ms, word_gap_ms)
                    is_endpoint = is_endpoint or trailing_silence_ms >= endpointing_ms

            text = re.sub(r"\s+", " ", result.get("text", "")).strip()
            text = strip_hallucinations(text)
            if len(text) < 2:
                continue
            if text == last_text:
                continue
            last_text = text

            now = time.monotonic()
            last_tail = refresh_tail(now)
            text = merge_overlap_text(last_tail, text)
            if not text:
                continue
            tail_history.append((now, text))

            translator.set_target(target_language)
            if sentence_mode:
                # Whisper terminates every chunk with a period, even when the speaker
                # only paused for breath. Drop that spurious terminator unless this chunk
                # ended on a real (long) pause, so a single sentence isn't fragmented.
                chunk_text = text
                if not is_endpoint:
                    stripped = re.sub(r"\s*[.!?…]+$", "", chunk_text).rstrip()
                    chunk_text = stripped or chunk_text
                sentence_buffer = merge_partial_buffer(sentence_buffer, chunk_text)
                if sentence_buffer and buffer_started_at is None:
                    buffer_started_at = now
                overlay.post_partial(sentence_case_text(strip_to_recent_text(sentence_buffer, 700)))
                buffer_age = (now - buffer_started_at) if buffer_started_at else 0.0
                stable_timeout = max_partial_seconds > 0 and buffer_age >= max_partial_seconds
                too_long = len(sentence_buffer) >= max_block_chars
                if is_endpoint or stable_timeout or too_long:
                    finalize_endpoint(pause_ms=trailing_silence_ms)
            else:
                translated = translator.translate(text)
                if translated:
                    overlay.post_pair(text, translated)
                    note_context(text)
        except Exception as exc:
            overlay.post_pair("", f"Ошибка: {exc}")
            time.sleep(1.0)
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)


def streaming_worker(
    audio_q,
    overlay,
    translator,
    stop_event,
    samplerate,
    whisper_model,
    settings,
    min_block_chars,
    max_block_chars,
    max_sentences,
    endpoint_min_words,
    reset_gen=None,
    use_vad=True,
    vad_min_speech_ms=250.0,
    update_seconds=1.0,
):
    """LocalAgreement-2 streaming transcription + translation."""
    import mlx_whisper

    proc = OnlineProcessor(samplerate)
    committed_text = ""   # confirmed words not yet emitted as a full sentence
    recent_context = ""   # fed to Whisper as initial_prompt
    detected_lang = "auto"  # Whisper's detected source language (for TranslateGemma)
    pending_seconds = 0.0
    seen_gen = reset_gen[0] if reset_gen is not None else 0
    vad_model, get_speech_timestamps = load_vad() if use_vad else (None, None)
    fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="live_stream_")
    os.close(fd)

    def note_context(text):
        nonlocal recent_context
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            recent_context = f"{recent_context} {text}".strip()[-240:]

    def transcribe_words(audio):
        nonlocal detected_lang
        source_language, _ = settings.get()
        sf.write(temp_path, audio, int(samplerate), subtype="PCM_16")
        context = punctuated_context(f"{recent_context} {committed_text}")
        result = mlx_whisper.transcribe(
            temp_path,
            path_or_hf_repo=whisper_model,
            language=None if source_language == "auto" else source_language,
            condition_on_previous_text=False,
            initial_prompt=context,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            temperature=0.0,  # no fallback retries — re-transcription self-corrects next pass
            word_timestamps=True,
            verbose=False,
        )
        detected_lang = result.get("language") or detected_lang
        words = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []) or []:
                t = (w.get("word") or "").strip()
                if t:
                    words.append((float(w.get("start", 0.0)), float(w.get("end", 0.0)), t))
        return words

    def emit_sentences(force=False, pause_ms=0):
        nonlocal committed_text
        committed_text = strip_hallucinations(committed_text)
        blocks, committed_text = take_blocks_for_translation(
            committed_text, min_block_chars, max_block_chars, max_sentences, force=force
        )
        if not blocks and (force or len(committed_text) >= max_block_chars):
            extra, committed_text = take_endpoint_blocks(
                committed_text, min_block_chars, max_block_chars, endpoint_min_words
            )
            blocks += extra
        for block in blocks:
            source_language, target_language = settings.get()
            translator.set_target(target_language)
            translator.set_source(detected_lang if source_language == "auto" else source_language)
            src = sentence_case_text(block)
            translated = translator.translate(src)
            if translated:
                overlay.post_pair(src, translated, pause_ms=pause_ms)
                note_context(src)

    while not stop_event.is_set():
        if reset_gen is not None and reset_gen[0] != seen_gen:
            seen_gen = reset_gen[0]
            proc.reset()
            committed_text = ""
            recent_context = ""
            pending_seconds = 0.0
            while True:
                try:
                    audio_q.get_nowait()
                except queue.Empty:
                    break

        # Catch-up: if transcription can't keep pace and the audio queue is piling up,
        # skip the stale backlog and jump back to live (one gap beats a growing delay).
        if audio_q.qsize() > 40:
            dropped = 0
            while audio_q.qsize() > 8:
                try:
                    audio_q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if dropped:
                print(f"[stream] отстаём — пропущено ~{dropped * 0.2:.1f}s аудио, догоняю", file=sys.stderr)

        try:
            block = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            proc.insert_audio(block)
            pending_seconds += len(block) / samplerate
            if pending_seconds < update_seconds:
                continue
            pending_seconds = 0.0

            # Skip pure silence/non-speech: don't re-transcribe it (hallucination source).
            if vad_model is not None and proc.audio is not None:
                has_speech, _ = vad_analyze(
                    proc.audio, samplerate, vad_model, get_speech_timestamps, 0, vad_min_speech_ms
                )
                if not has_speech:
                    # Commit the still-unconfirmed tail before dropping the audio, otherwise
                    # the last words of every utterance are lost at the pause.
                    tail = proc.finalize()
                    if tail:
                        committed_text = f"{committed_text} {tail}".strip()
                    proc.drop_active()
                    if committed_text.strip():
                        emit_sentences(force=True, pause_ms=1000)  # pause => flush sentence
                    overlay.post_partial("")
                    continue

            gen_before = reset_gen[0] if reset_gen is not None else seen_gen
            committed, draft = proc.process(transcribe_words)
            if reset_gen is not None and reset_gen[0] != gen_before:
                continue
            for (_, _, t) in committed:
                committed_text = f"{committed_text} {t}".strip()

            emit_sentences(force=False)

            draft_clean = strip_hallucinations(draft)
            live = f"{committed_text} {draft_clean}".strip()
            overlay.post_partial(sentence_case_text(strip_to_recent_text(live, 700)))
            overlay.post_status(audio_q.qsize())
        except Exception as exc:
            print(f"[stream] ошибка: {exc!r} — продолжаю", file=sys.stderr)
            time.sleep(0.2)

    Path(temp_path).unlink(missing_ok=True)


def build_translator(args):
    if args.translator == "ollama":
        return OllamaTranslator(
            model=args.ollama_model,
            target=args.target,
            url=args.ollama_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning=args.reasoning,
            source=args.source,
        )
    return MLXTranslator(
        model_repo=args.mlx_llm,
        target=args.target,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning=args.reasoning,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time speech transcription + local LLM translation + macOS glass overlay."
    )
    p.add_argument("--list", action="store_true", help="показать аудио-устройства и выйти")
    p.add_argument("--device", type=int, default=None, help="номер входного аудио-устройства")
    p.add_argument("--source", default="auto", help="язык речи: auto/en/ru/es/... (по умолчанию auto)")
    p.add_argument("--target", default="ru", help="язык перевода: ru/en/es/... (по умолчанию ru)")
    p.add_argument(
        "--whisper",
        choices=WHISPER_MODELS.keys(),
        default="large-v3",
        help="модель транскрибации: large-v3 точнее, small/base быстрее",
    )
    p.add_argument(
        "--translator",
        choices=("mlx", "ollama"),
        default="mlx",
        help="локальный backend перевода",
    )
    p.add_argument("--ollama-model", default="qwen3.5:4b", help="модель Ollama")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="адрес Ollama")
    p.add_argument(
        "--mlx-llm",
        default="mlx-community/Qwen3.5-4B-MLX-4bit",
        help="MLX-LM repo на Hugging Face",
    )
    p.add_argument("--chunk-seconds", type=float, default=6.0, help="МАКС размер аудио-чанка (окна)")
    p.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=1.2,
        help="минимум аудио перед тем, как резать чанк на паузе",
    )
    p.add_argument(
        "--pause-emit-ms",
        type=float,
        default=350.0,
        help="пауза (мс), на которой эмитить чанк, не дожидаясь МАКС окна",
    )
    p.add_argument("--overlap-seconds", type=float, default=0.75, help="overlap между чанками")
    p.add_argument("--block-seconds", type=float, default=0.20, help="размер блока записи")
    p.add_argument(
        "--endpointing-ms",
        type=float,
        default=1000.0,
        help="сколько мс тишины считать концом utterance",
    )
    p.add_argument(
        "--endpoint-min-words",
        type=int,
        default=8,
        help="минимум слов для перевода на endpoint даже без пунктуации",
    )
    p.add_argument(
        "--max-partial-seconds",
        type=float,
        default=5.0,
        help="максимально держать live partial перед stable commit",
    )
    p.add_argument(
        "--mutable-tail-seconds",
        type=float,
        default=8.0,
        help="сколько секунд recent text держать изменяемым для overlap-дедупликации",
    )
    p.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="не использовать word timestamps Whisper для endpointing",
    )
    p.add_argument(
        "--audio-queue",
        type=int,
        default=0,
        help="буфер сырых аудио-блоков; 0 = без лимита (без потерь)",
    )
    p.add_argument("--chunk-queue", type=int, default=8, help="буфер чанков на транскрибацию")
    p.add_argument(
        "--no-sentence-mode",
        action="store_true",
        help="переводить каждый распознанный чанк сразу, без ожидания конца предложения",
    )
    p.add_argument(
        "--flush-after",
        type=float,
        default=0.0,
        help="legacy fallback; endpointing-ms является основным механизмом финализации",
    )
    p.add_argument(
        "--min-sentence-chars",
        type=int,
        default=20,
        help="совместимость: если --min-block-chars не задан, используется как минимум блока",
    )
    p.add_argument(
        "--min-block-chars",
        type=int,
        default=90,
        help="минимальный размер смыслового блока перед переводом",
    )
    p.add_argument(
        "--max-block-chars",
        type=int,
        default=600,
        help="максимальный размер смыслового блока перед переводом",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=5,
        help="максимум предложений за один вызов переводчика",
    )
    p.add_argument("--silence-rms", type=float, default=0.006, help="порог тишины (строже = выше)")
    p.add_argument(
        "--no-vad",
        action="store_true",
        help="не выравнивать нарезку по тишине через Silero VAD (фиксированные чанки)",
    )
    p.add_argument(
        "--vad-min-speech-ms",
        type=float,
        default=250.0,
        help="минимум суммарной речи в чанке (мс), иначе чанк считается без речи и пропускается",
    )
    p.add_argument(
        "--legacy-chunking",
        action="store_true",
        help="старая нарезка на чанки вместо LocalAgreement-стриминга",
    )
    p.add_argument(
        "--update-seconds",
        type=float,
        default=1.5,
        help="как часто (с накопленного аудио) пере-распознавать буфер в LocalAgreement-режиме",
    )
    p.add_argument("--max-tokens", type=int, default=180, help="лимит токенов перевода")
    p.add_argument("--temperature", type=float, default=0.1, help="температура LLM")
    p.add_argument(
        "--reasoning",
        action="store_true",
        help="включить reasoning/thinking у Qwen; по умолчанию выключено",
    )
    p.add_argument("--width", type=int, default=1100, help="ширина окна")
    p.add_argument("--height", type=int, default=520, help="высота окна")
    p.add_argument("--opacity", type=float, default=0.92, help="прозрачность окна")
    p.add_argument("--no-window", action="store_true", help="выводить в терминал вместо окна")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        list_devices()
        return

    device = args.device
    if device is None:
        device, name = find_blackhole()
        if device is None:
            sys.exit(
                "Не нашёл BlackHole. Установи `brew install blackhole-2ch`, "
                "настрой Multi-Output Device или передай --device N после --list."
            )
        print(f"Найдено устройство: [{device}] {name}")

    info = sd.query_devices(device)
    samplerate = float(info["default_samplerate"])
    channels = min(2, int(info["max_input_channels"])) or 1
    whisper_model = WHISPER_MODELS[args.whisper]

    stop_event = threading.Event()
    reset_gen = [0]  # bumped by the Clear button to wipe both workers' state
    settings = LanguageSettings(args.source, args.target)
    min_block_chars = args.min_block_chars or args.min_sentence_chars
    audio_q = queue.Queue(maxsize=args.audio_queue)
    chunk_q = queue.Queue(maxsize=args.chunk_queue)

    print("Загружаю переводчик...")
    translator = build_translator(args)
    print(
        f"Старт: {info['name']} -> Whisper {args.whisper} -> "
        f"{args.translator} -> {args.target}"
    )

    overlay = (
        ConsoleOverlay(stop_event)
        if args.no_window
        else GlassOverlay(
            stop_event,
            settings=settings,
            title="Live Translation",
            width=args.width,
            height=args.height,
            opacity=args.opacity,
        )
    )
    overlay.reset_gen = reset_gen

    workers = [
        threading.Thread(
            target=audio_capture_worker,
            args=(audio_q, stop_event, device, samplerate, channels, args.block_seconds),
            daemon=True,
        ),
    ]
    if args.legacy_chunking:
        workers += [
            threading.Thread(
                target=chunk_worker,
                args=(
                    audio_q,
                    chunk_q,
                    overlay,
                    stop_event,
                    samplerate,
                    args.chunk_seconds,
                    args.overlap_seconds,
                    args.endpointing_ms,
                    args.silence_rms,
                    not args.no_vad,
                    args.min_chunk_seconds,
                    args.pause_emit_ms,
                    reset_gen,
                    args.vad_min_speech_ms,
                ),
                daemon=True,
            ),
            threading.Thread(
                target=transcribe_translate_worker,
                args=(
                    chunk_q,
                    overlay,
                    translator,
                    stop_event,
                    samplerate,
                    whisper_model,
                    settings,
                    not args.no_sentence_mode,
                    args.flush_after,
                    min_block_chars,
                    args.max_block_chars,
                    args.max_sentences,
                    args.mutable_tail_seconds,
                    args.endpointing_ms,
                    args.endpoint_min_words,
                    args.max_partial_seconds,
                    not args.no_word_timestamps,
                    reset_gen,
                ),
                daemon=True,
            ),
        ]
    else:
        workers.append(
            threading.Thread(
                target=streaming_worker,
                args=(
                    audio_q,
                    overlay,
                    translator,
                    stop_event,
                    samplerate,
                    whisper_model,
                    settings,
                    min_block_chars,
                    args.max_block_chars,
                    args.max_sentences,
                    args.endpoint_min_words,
                    reset_gen,
                    not args.no_vad,
                    args.vad_min_speech_ms,
                    args.update_seconds,
                ),
                daemon=True,
            )
        )

    def handle_signal(signum, frame):
        overlay.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for worker in workers:
        worker.start()

    try:
        overlay.run()
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
