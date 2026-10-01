# coding=utf-8
"""Очередь GPU → UI: чанки текста, преролл, живой Web Audio, допись wav, стоп."""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

TEXT_CHUNK_CHARS = 500
PREROLL_SEC = 10.0
CHUNK_PAUSE_SEC = 0.06
QUEUE_MAX = 16
DEFAULT_SR = 24000

PcmGen = Callable[[str], Iterator[Tuple[np.ndarray, int]]]
StopFn = Callable[[], bool]
AudioOut = Union[None, str, Tuple[int, np.ndarray]]
UiOut = Tuple[AudioOut, str, str]


def _pcm1d(pcm: np.ndarray) -> np.ndarray:
    x = np.asarray(pcm, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1).astype(np.float32)
    return np.ascontiguousarray(x)


def _trim_trailing_silence(pcm: np.ndarray, sr: int, floor: float = 0.012, keep_sec: float = 0.04) -> np.ndarray:
    """Срезает хвост тишины у последнего куска чанка, не трогая саму речь."""
    x = _pcm1d(pcm)
    if x.size < 8:
        return x
    mag = np.abs(x)
    voiced = np.flatnonzero(mag > floor)
    if voiced.size == 0:
        keep = int(keep_sec * sr)
        return x[:keep] if keep < x.size else x
    end = min(x.size, int(voiced[-1]) + int(keep_sec * sr) + 1)
    return x[:end]


def _live_dir(output_path: str) -> Path:
    d = Path(output_path).resolve().parent / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pcm_to_live_cmd(pcm: np.ndarray, sr: int, seq: int, output_path: str) -> str:
    """Пишет короткий wav и шлёт в UI только имя файла — base64 в Textbox Gradio рвёт."""
    x = np.clip(_pcm1d(pcm), -1.0, 1.0)
    stem = Path(output_path).stem
    name = f"{stem}_{int(seq):05d}.wav"
    path = _live_dir(output_path) / name
    sf.write(str(path), x, int(sr), subtype="PCM_16")
    return json.dumps({"cmd": "push", "file": name, "n": int(seq), "sr": int(sr)}, ensure_ascii=True)


def live_stop_cmd(seq: int = 0) -> str:
    return json.dumps({"cmd": "stop", "n": int(seq)}, ensure_ascii=True)


class _WavWriter:
    def __init__(self, path: str):
        self.path = path
        self._file = None
        self.sr = None
        self.samples = 0

    def write(self, pcm: np.ndarray, sr: int):
        pcm = _pcm1d(pcm)
        if self._file is None:
            self.sr = int(sr)
            self._file = sf.SoundFile(
                self.path,
                mode="w",
                samplerate=self.sr,
                channels=1,
                subtype="PCM_16",
            )
        self._file.write(pcm)
        self.samples += int(pcm.shape[0])

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def duration(self) -> float:
        if not self.sr:
            return 0.0
        return self.samples / float(self.sr)


def new_output_path(output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return str(output_dir / f"qwen3_tts_{stamp}.wav")


def stream_pcm_to_ui(
    chunks: List[str],
    generate_chunk: PcmGen,
    *,
    output_path: str,
    autoplay: bool,
    stop_fn: StopFn,
    title: str,
    engine_name: str,
    pause_sec: float = CHUNK_PAUSE_SEC,
    preroll_sec: float = PREROLL_SEC,
) -> Iterator[UiOut]:
    """
    GPU считает в очередь. В браузер уходят PCM-команды для Web Audio (без пересоздания <audio>).
    Готовый wav пишется на диск и в конце кладётся в обычный плеер (без autoplay).
    """
    total = max(len(chunks), 1)
    q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
    writer = _WavWriter(output_path)
    start = time.time()

    def _put(item) -> bool:
        while not stop_fn():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        try:
            last_sr = DEFAULT_SR
            for i, text in enumerate(chunks):
                if stop_fn():
                    break
                if not _put(("status", f"{title}\nЧасть {i + 1}/{total}: {text[:70]}...")):
                    break
                got = False
                pending = None
                pending_sr = last_sr
                for pcm, sr in generate_chunk(text):
                    if stop_fn():
                        break
                    piece = _pcm1d(pcm)
                    piece_sr = int(sr)
                    last_sr = piece_sr
                    if pending is not None:
                        if not _put(("pcm", i, pending_sr, pending)):
                            pending = None
                            break
                    pending = piece
                    pending_sr = piece_sr
                    got = True
                if pending is not None:
                    pending = _trim_trailing_silence(pending, pending_sr)
                    if pending.size:
                        _put(("pcm", i, pending_sr, pending))
                if stop_fn():
                    break
                if got and i < total - 1 and pause_sec > 0:
                    silence = np.zeros(int(pause_sec * last_sr), dtype=np.float32)
                    _put(("pcm", i, last_sr, silence))
            _put(("done", None))
        except Exception as exc:
            _put(("error", exc))

    threading.Thread(target=producer, name="tts-producer", daemon=True).start()

    preroll: List[np.ndarray] = []
    preroll_samples = 0
    started = False
    last_sr = DEFAULT_SR
    seq = 0
    last_status = f"{title}\nДвижок: {engine_name}\nПодготовка..."

    def live_status(extra: str = "") -> str:
        elapsed = time.time() - start
        dur = writer.duration
        rtf = (elapsed / dur) if dur > 0 else 0.0
        speed = (dur / elapsed) if elapsed > 0 and dur > 0 else 0.0
        lines = [
            last_status,
            extra,
            f"Движок: {engine_name}",
            f"Аудио: {dur:.1f} с | генерация: {elapsed:.1f} с | {speed:.2f}× realtime (RTF {rtf:.2f})",
            f"Файл: {output_path}",
        ]
        return "\n".join(x for x in lines if x)

    def _kill(message: str) -> UiOut:
        nonlocal seq
        writer.close()
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        seq += 1
        return None, message, live_stop_cmd(seq)

    try:
        while True:
            if stop_fn():
                yield _kill("Остановлено. Генерация и воспроизведение прерваны.")
                return

            try:
                item = q.get(timeout=0.15)
            except queue.Empty:
                continue

            kind = item[0]
            if kind == "status":
                last_status = item[1]
                if not started:
                    yield None, live_status(), ""
                continue

            if kind == "error":
                writer.close()
                seq += 1
                yield None, f"Ошибка: {type(item[1]).__name__}: {item[1]}", live_stop_cmd(seq)
                return

            if kind == "done":
                break

            _, _idx, sr, pcm = item
            last_sr = sr
            writer.write(pcm, sr)

            if not autoplay:
                if not started:
                    yield None, live_status(), ""
                continue

            if not started:
                preroll.append(pcm)
                preroll_samples += int(pcm.shape[0])
                if preroll_samples / float(sr) >= preroll_sec:
                    block = np.concatenate(preroll)
                    preroll = []
                    started = True
                    seq += 1
                    yield None, live_status("Играю..."), pcm_to_live_cmd(block, sr, seq, output_path)
                else:
                    yield None, live_status(f"Преролл {preroll_samples / float(sr):.1f}/{preroll_sec:.0f} с"), ""
            else:
                seq += 1
                yield None, live_status("Играю..."), pcm_to_live_cmd(pcm, sr, seq, output_path)

        if stop_fn():
            yield _kill("Остановлено. Генерация и воспроизведение прерваны.")
            return

        writer.close()
        if writer.samples > 0:
            yield output_path, live_status("Готово"), ""
        else:
            yield None, live_status("Готово"), ""
    finally:
        writer.close()
