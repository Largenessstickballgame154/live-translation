#!/usr/bin/env python3
"""
record_and_transcribe.py

Записывает системный звук на Mac (что угодно, что играет — браузер, плеер,
звонок, любое приложение) и транскрибирует локально через MLX Whisper,
который использует GPU на Apple Silicon (быстро + точно).

------------------------------------------------------------------------------
ТРЕБОВАНИЯ (поставить один раз):

    brew install blackhole-2ch        # виртуальное аудио-устройство (loopback)
    brew install ffmpeg               # нужен mlx-whisper для чтения аудио
    pip install mlx-whisper sounddevice soundfile numpy

НАСТРОЙКА ЗВУКА (один раз):
    1. Открой "Audio MIDI Setup" (Настройка Audio-MIDI).
    2. Создай "Multi-Output Device" (нижний "+").
    3. Отметь в нём свои колонки/наушники И "BlackHole 2ch".
    4. Сделай этот Multi-Output Device системным выводом звука
       (меню громкости / Системные настройки -> Звук -> Вывод).
    Теперь ты и слышишь звук, и он одновременно уходит в BlackHole.
------------------------------------------------------------------------------

ПРИМЕРЫ:

    # посмотреть список аудио-устройств и их номера
    python record_and_transcribe.py --list

    # записывать пока не нажмёшь Ctrl+C, потом сразу транскрибировать
    python record_and_transcribe.py --transcribe

    # записать ровно 5 минут английской речи большой моделью
    python record_and_transcribe.py --duration 300 --language en --transcribe

    # транскрибировать уже готовый файл (без записи)
    python record_and_transcribe.py --input some_audio.wav --transcribe
"""

import argparse
import datetime as dt
import json
import queue
import sys
from pathlib import Path

import sounddevice as sd
import soundfile as sf

# Модели MLX Whisper с Hugging Face (скачиваются автоматически при первом запуске).
#   large-v3  -> максимальная точность (рекомендуется, раз скорость не проблема)
#   small     -> быстрее, менее точно
#   tiny      -> очень быстро, для черновиков
MODELS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def list_devices():
    print(sd.query_devices())
    print("\nПодсказка: ищи строку с 'BlackHole' и её номер слева.")


def find_blackhole():
    """Автоматически находит входное устройство BlackHole."""
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and "blackhole" in dev["name"].lower():
            return idx, dev["name"]
    return None, None


def record(output_path, device, samplerate, channels, duration=None):
    """Пишет звук с входного устройства в WAV потоково (без жора памяти)."""
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print(f"\n● Запись -> {output_path}")
    if duration:
        print(f"  Длительность: {duration} c")
    else:
        print("  Останови нажатием Ctrl+C")
    print(
        f"  Устройство: {sd.query_devices(device)['name']}  "
        f"({samplerate:.0f} Гц, {channels} кан.)\n"
    )

    frames_written = 0
    target_frames = int(duration * samplerate) if duration else None

    with sf.SoundFile(
        output_path,
        mode="w",
        samplerate=int(samplerate),
        channels=channels,
        subtype="PCM_16",
    ) as f:
        with sd.InputStream(
            samplerate=samplerate,
            device=device,
            channels=channels,
            callback=callback,
        ):
            try:
                while True:
                    block = q.get()
                    f.write(block)
                    frames_written += len(block)
                    if target_frames and frames_written >= target_frames:
                        break
            except KeyboardInterrupt:
                pass

    print(f"✓ Запись сохранена: {output_path}")
    return output_path


def _fmt_ts(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_outputs(result, base_path):
    """Сохраняет .txt, .srt и .json рядом с базовым именем."""
    base = Path(base_path).with_suffix("")
    txt = base.with_suffix(".txt")
    srt = base.with_suffix(".srt")
    js = base.with_suffix(".json")

    txt.write_text(result["text"].strip() + "\n", encoding="utf-8")

    with open(srt, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.get("segments", []), 1):
            f.write(
                f"{i}\n"
                f"{_fmt_ts(seg['start'])} --> {_fmt_ts(seg['end'])}\n"
                f"{seg['text'].strip()}\n\n"
            )

    js.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✓ Текст:     {txt}")
    print(f"✓ Субтитры:  {srt}")
    print(f"✓ JSON:      {js}")


def transcribe(audio_path, model_repo, language):
    """Транскрибирует через MLX Whisper с защитой от галлюцинаций на тишине."""
    import mlx_whisper  # импорт здесь, чтобы --list работал без mlx

    print(f"\n● Транскрибирую через {model_repo} ...")
    print("  (первый запуск модели качает её с Hugging Face — это разово)")

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model_repo,
        language=language,  # None => автоопределение языка
        # --- меры против зацикливания/галлюцинаций на тихих участках ---
        condition_on_previous_text=False,  # не даём модели уходить в петлю по своему же тексту
        hallucination_silence_threshold=2.0,  # подозрительно тихие куски пропускаем
        no_speech_threshold=0.6,  # порог "тут нет речи"
        compression_ratio_threshold=2.4,  # отсекает повторяющийся мусор ("sniff sniff...")
        logprob_threshold=-1.0,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # фолбэк при неудаче
        verbose=False,
    )
    print("✓ Готово.")
    return result


def main():
    p = argparse.ArgumentParser(
        description="Запись системного звука Mac + локальная транскрипция (MLX Whisper)."
    )
    p.add_argument("--list", action="store_true", help="показать аудио-устройства и выйти")
    p.add_argument(
        "--device",
        type=int,
        default=None,
        help="номер входного устройства (по умолчанию ищется BlackHole)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="сколько секунд писать (по умолчанию — до Ctrl+C)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="имя выходного WAV (по умолчанию с датой-временем)",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="транскрибировать готовый аудиофайл вместо записи",
    )
    p.add_argument("--transcribe", action="store_true", help="транскрибировать после записи")
    p.add_argument(
        "--model",
        choices=MODELS.keys(),
        default="large-v3",
        help="модель Whisper (по умолчанию large-v3)",
    )
    p.add_argument(
        "--language",
        type=str,
        default=None,
        help="код языка, напр. en / ru (по умолчанию — автоопределение)",
    )
    args = p.parse_args()

    if args.list:
        list_devices()
        return

    # Режим "только транскрипция" готового файла
    if args.input:
        audio = Path(args.input)
        if not audio.exists():
            sys.exit(f"Файл не найден: {audio}")
        result = transcribe(audio, MODELS[args.model], args.language)
        write_outputs(result, audio)
        return

    # Выбираем устройство записи
    device = args.device
    if device is None:
        device, name = find_blackhole()
        if device is None:
            sys.exit(
                "Не нашёл BlackHole. Поставь его (brew install blackhole-2ch) и\n"
                "настрой Multi-Output Device, либо укажи --device N (см. --list)."
            )
        print(f"Найдено устройство: [{device}] {name}")

    info = sd.query_devices(device)
    samplerate = info["default_samplerate"]
    channels = min(2, info["max_input_channels"]) or 1

    if args.output:
        wav_path = args.output
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = f"recording_{stamp}.wav"

    record(wav_path, device, samplerate, channels, args.duration)

    if args.transcribe:
        result = transcribe(wav_path, MODELS[args.model], args.language)
        write_outputs(result, wav_path)
    else:
        print("\nЧтобы транскрибировать позже:")
        print(f"  python {Path(__file__).name} --input {wav_path} --transcribe")


if __name__ == "__main__":
    main()
