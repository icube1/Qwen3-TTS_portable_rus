# coding=utf-8
"""Адаптер TTS-движка: CUDA Graphs (faster-qwen3-tts) с откатом на официальный qwen_tts."""

from __future__ import annotations

import importlib
import traceback
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


ENGINE_FAST = "cuda_graphs"
ENGINE_OFFICIAL = "official"

_loaded: Dict[tuple, "TtsBackend"] = {}
_fast_import_error: Optional[str] = None
_fast_cls = None


def _patch_static_cache_init():
    """transformers 4.57 принимает lazy_initialization(key), старый faster-qwen3-tts звал (k, v)."""
    try:
        from faster_qwen3_tts.predictor_graph import PredictorGraph
        from faster_qwen3_tts.talker_graph import TalkerGraph
    except Exception:
        return

    def _wrap(cls, config_attr: str):
        orig = cls._init_cache_layers

        def _init_cache_layers(self):
            try:
                return orig(self)
            except TypeError as exc:
                if "lazy_initialization" not in str(exc):
                    raise
                config = getattr(self, config_attr).config
                num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
                dummy_k = torch.zeros(1, num_kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
                for layer in self.static_cache.layers:
                    if not getattr(layer, "is_initialized", True):
                        layer.lazy_initialization(dummy_k)

        cls._init_cache_layers = _init_cache_layers

    _wrap(PredictorGraph, "pred_model")
    _wrap(TalkerGraph, "model")


def _try_load_fast_cls():
    global _fast_cls, _fast_import_error
    if _fast_cls is not None or _fast_import_error is not None:
        return _fast_cls
    try:
        mod = importlib.import_module("faster_qwen3_tts")
        _fast_cls = getattr(mod, "FasterQwen3TTS")
        _patch_static_cache_init()
        return _fast_cls
    except Exception as exc:
        _fast_import_error = f"{type(exc).__name__}: {exc}"
        print(f"[tts_engine] faster-qwen3-tts недоступен: {_fast_import_error}")
        return None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_pcm(chunk) -> np.ndarray:
    if isinstance(chunk, tuple) and len(chunk) >= 2 and isinstance(chunk[0], (int, float)):
        chunk = chunk[1]
    if torch.is_tensor(chunk):
        chunk = chunk.detach().float().cpu().numpy()
    pcm = np.asarray(chunk, dtype=np.float32)
    if pcm.ndim > 1:
        pcm = np.mean(pcm, axis=-1).astype(np.float32)
    return pcm


def _token_budget(text: str, requested: int) -> int:
    """Не даём модели молотить до 2048 токенов на коротком куске — отсюда лишние слова."""
    n = max(96, int(len(text or "") * 2.2) + 96)
    return max(64, min(int(requested), n))


def _iter_fast_stream(gen) -> Iterator[Tuple[np.ndarray, int]]:
    for item in gen:
        if item is None:
            continue
        if isinstance(item, tuple):
            if len(item) >= 2 and isinstance(item[1], int):
                pcm, sr = item[0], item[1]
            elif len(item) >= 2 and isinstance(item[0], int):
                sr, pcm = item[0], item[1]
            else:
                pcm, sr = item[0], 24000
        else:
            pcm, sr = item, 24000
        yield _as_pcm(pcm), int(sr)


class TtsBackend:
    def __init__(self, model: Any, engine: str):
        self.model = model
        self.engine = engine

    @property
    def is_fast(self) -> bool:
        return self.engine == ENGINE_FAST

    @property
    def inner(self):
        if self.is_fast and hasattr(self.model, "model"):
            return self.model.model
        return self.model

    def create_voice_clone_prompt(self, **kwargs):
        target = self.inner
        return target.create_voice_clone_prompt(**kwargs)

    def stream_clone(
        self,
        text: str,
        language: str,
        *,
        voice_clone_prompt=None,
        ref_audio=None,
        ref_text=None,
        x_vector_only_mode: bool = False,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        chunk_size: int = 8,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        max_new_tokens = _token_budget(text, max_new_tokens)
        if self.is_fast and hasattr(self.model, "generate_voice_clone_streaming"):
            kwargs = dict(
                text=text,
                language=language,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                chunk_size=chunk_size,
                repetition_penalty=1.12,
            )
            if voice_clone_prompt is not None:
                kwargs["voice_clone_prompt"] = voice_clone_prompt
            else:
                kwargs["ref_audio"] = ref_audio
                kwargs["ref_text"] = ref_text or ""
                kwargs["xvec_only"] = x_vector_only_mode
            try:
                yield from _iter_fast_stream(self.model.generate_voice_clone_streaming(**kwargs))
                return
            except Exception:
                traceback.print_exc()
                print("[tts_engine] streaming clone упал, пробую обычный generate")

        kwargs = dict(
            text=text,
            language=language,
            max_new_tokens=_token_budget(text, max_new_tokens),
            temperature=temperature,
            top_p=top_p,
        )
        if voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = voice_clone_prompt
        else:
            kwargs["ref_audio"] = ref_audio
            kwargs["ref_text"] = ref_text
            kwargs["x_vector_only_mode"] = x_vector_only_mode
        wavs, sr = self.inner.generate_voice_clone(**kwargs)
        yield _as_pcm(wavs[0]), int(sr)

    def stream_custom(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: Optional[str] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        chunk_size: int = 8,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        max_new_tokens = _token_budget(text, max_new_tokens)
        kwargs = dict(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if self.is_fast and hasattr(self.model, "generate_custom_voice_streaming"):
            stream_kwargs = dict(kwargs)
            stream_kwargs["chunk_size"] = chunk_size
            try:
                yield from _iter_fast_stream(self.model.generate_custom_voice_streaming(**stream_kwargs))
                return
            except TypeError:
                stream_kwargs.pop("chunk_size", None)
                try:
                    yield from _iter_fast_stream(self.model.generate_custom_voice_streaming(**stream_kwargs))
                    return
                except Exception:
                    traceback.print_exc()
            except Exception:
                traceback.print_exc()

        wavs, sr = self.inner.generate_custom_voice(**kwargs)
        yield _as_pcm(wavs[0]), int(sr)

    def stream_design(
        self,
        text: str,
        language: str,
        instruct: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        chunk_size: int = 8,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        max_new_tokens = _token_budget(text, max_new_tokens)
        kwargs = dict(
            text=text,
            language=language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if self.is_fast and hasattr(self.model, "generate_voice_design_streaming"):
            stream_kwargs = dict(kwargs)
            stream_kwargs["chunk_size"] = chunk_size
            try:
                yield from _iter_fast_stream(self.model.generate_voice_design_streaming(**stream_kwargs))
                return
            except TypeError:
                stream_kwargs.pop("chunk_size", None)
                try:
                    yield from _iter_fast_stream(self.model.generate_voice_design_streaming(**stream_kwargs))
                    return
                except Exception:
                    traceback.print_exc()
            except Exception:
                traceback.print_exc()

        wavs, sr = self.inner.generate_voice_design(**kwargs)
        yield _as_pcm(wavs[0]), int(sr)


def _load_official(model_path: str, device: str) -> Qwen3TTSModel:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn_impl = None
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("[tts_engine] official: Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            print("[tts_engine] official: SDPA")
    return Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )


def _load_fast(model_path: str, device: str) -> Any:
    cls = _try_load_fast_cls()
    if cls is None:
        raise RuntimeError(_fast_import_error or "faster-qwen3-tts не установлен")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = cls.from_pretrained(
        model_path,
        device="cuda" if device == "cuda" else "cpu",
        dtype=dtype,
        attn_implementation="sdpa",
        max_seq_len=4096,
    )
    if hasattr(model, "warmup"):
        print("[tts_engine] CUDA Graphs warmup...")
        try:
            model.warmup(prefill_len=100)
            print("[tts_engine] warmup готов")
        except Exception:
            traceback.print_exc()
            print("[tts_engine] warmup не удался — граф попробуем на первом generate")
    return model


def get_tts_backend(model_type: str, model_size: str, model_path: str) -> TtsBackend:
    """Загрузить (или взять из кэша) бэкенд. Fast путь только на CUDA."""
    device = get_device()
    prefer_fast = device == "cuda" and _try_load_fast_cls() is not None
    engine = ENGINE_FAST if prefer_fast else ENGINE_OFFICIAL
    key = (engine, model_type, model_size)
    if key in _loaded:
        return _loaded[key]

    print(f"[tts_engine] загрузка {model_type} {model_size} ({engine}) из {model_path}")
    if engine == ENGINE_FAST:
        try:
            backend = TtsBackend(_load_fast(model_path, device), ENGINE_FAST)
            _loaded[key] = backend
            print(f"[tts_engine] {model_type} {model_size}: CUDA Graphs")
            return backend
        except Exception:
            traceback.print_exc()
            print("[tts_engine] откат на официальный qwen_tts")
            engine = ENGINE_OFFICIAL
            key = (engine, model_type, model_size)
            if key in _loaded:
                return _loaded[key]

    backend = TtsBackend(_load_official(model_path, device), ENGINE_OFFICIAL)
    _loaded[key] = backend
    print(f"[tts_engine] {model_type} {model_size}: official")
    return backend


def engine_status_line() -> str:
    if get_device() != "cuda":
        return "Движок: official (CPU)"
    if _try_load_fast_cls() is not None:
        return "Движок: CUDA Graphs (faster-qwen3-tts)"
    reason = _fast_import_error or "пакет не установлен"
    return f"Движок: official ({reason})"
