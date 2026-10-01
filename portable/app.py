# coding=utf-8
"""
Qwen3-TTS Portable PRO - Русскоязычная версия со стримингом
Синтез речи с поддержкой: Дизайн голоса, Клонирование голоса, Пресеты голосов
Multi-speaker режим, профили голосов, загрузка из облака

Авторы:
@nerual_dreming - база, основной код, основатель ArtGeneration.me
"""

import os
import sys
import time
import json
import threading
import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from urllib.parse import urlparse
import pickle
import hashlib

import gradio as gr
import numpy as np
import torch
import soundfile as sf
from huggingface_hub import snapshot_download, hf_hub_download

# portable/ — tts_engine, tts_stream; родитель — вендорный qwen_tts
_PORTABLE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PORTABLE_DIR.parent
sys.path.insert(0, str(_PORTABLE_DIR))
sys.path.insert(1, str(_REPO_DIR))

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from tts_engine import engine_status_line, get_tts_backend
from tts_stream import TEXT_CHUNK_CHARS, live_stop_cmd, new_output_path, stream_pcm_to_ui

# =====================================================
# Константы и конфигурация
# =====================================================

APP_VERSION = "2.1.0"
APP_NAME = "Qwen3-TTS Portable PRO"

# Директории
SCRIPT_DIR = _PORTABLE_DIR
VOICES_DIR = SCRIPT_DIR / "voices"
PROFILES_DIR = SCRIPT_DIR / "profiles"
OUTPUT_DIR = SCRIPT_DIR / "output"
CONFIG_FILE = SCRIPT_DIR / "config.json"

# Создаем директории
VOICES_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# Глобальные переменные
# =====================================================

# Загруженные модели (кэш)
loaded_models: Dict[tuple, Qwen3TTSModel] = {}

# Кэш профилей голосов
voice_profiles_cache: Dict[str, VoiceClonePromptItem] = {}

# Флаги для стриминга
is_generating = False
stop_generation = False

# Размеры моделей
MODEL_SIZES = ["0.6B", "1.7B"]

# Типы моделей
MODEL_TYPES = {
    "Base": "Клонирование голоса",
    "CustomVoice": "Пресеты голосов",
    "VoiceDesign": "Дизайн голоса"
}

# Спикеры для CustomVoice
SPEAKERS = {
    "Aiden": "Эйден (мужской, английский)",
    "Dylan": "Дилан (мужской, английский)",
    "Eric": "Эрик (мужской, английский)",
    "Ono_anna": "Анна (женский, японский)",
    "Ryan": "Райан (мужской, английский)",
    "Serena": "Серена (женский, английский)",
    "Sohee": "Сохи (женский, корейский)",
    "Uncle_fu": "Дядя Фу (мужской, китайский)",
    "Vivian": "Вивиан (женский, английский)"
}

# Языки
LANGUAGES = {
    "Auto": "Авто (определить автоматически)",
    "Russian": "Русский",
    "English": "Английский",
    "Chinese": "Китайский",
    "Japanese": "Японский",
    "Korean": "Корейский",
    "French": "Французский",
    "German": "Немецкий",
    "Spanish": "Испанский",
    "Portuguese": "Португальский",
    "Italian": "Итальянский"
}

# =====================================================
# Конфигурация приложения
# =====================================================

@dataclass
class AppConfig:
    """Конфигурация приложения."""
    default_model_size: str = "1.7B"
    default_language: str = "Russian"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    auto_save_audio: bool = True
    theme: str = "soft"

    @classmethod
    def load(cls) -> "AppConfig":
        """Загрузка конфигурации из файла."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
        return cls()

    def save(self):
        """Сохранение конфигурации в файл."""
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")


# Глобальная конфигурация
app_config = AppConfig.load()

# =====================================================
# Профили голосов
# =====================================================

@dataclass
class VoiceProfile:
    """Профиль голоса для сохранения и загрузки."""
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    ref_text: Optional[str] = None
    x_vector_only_mode: bool = False
    audio_hash: str = ""  # хэш референсного аудио

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        return cls(**data)


def get_audio_hash(audio_data: np.ndarray) -> str:
    """Получение хэша аудио данных."""
    return hashlib.md5(audio_data.tobytes()).hexdigest()[:16]


def save_voice_profile(
    name: str,
    description: str,
    ref_audio: Tuple[np.ndarray, int],
    ref_text: Optional[str],
    x_vector_only: bool,
    model_size: str
) -> str:
    """Сохранение профиля голоса."""
    try:
        wav, sr = ref_audio
        audio_hash = get_audio_hash(wav)

        # Создаем профиль
        profile = VoiceProfile(
            name=name,
            description=description,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            audio_hash=audio_hash
        )

        # Получаем модель для создания voice clone prompt
        tts = get_model("Base", model_size)

        # Создаем VoiceClonePromptItem
        voice_prompt = tts.create_voice_clone_prompt(
            ref_audio=(wav, sr),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only
        )

        # Сохраняем данные
        profile_dir = PROFILES_DIR / name
        profile_dir.mkdir(exist_ok=True)

        # Сохраняем метаданные
        with open(profile_dir / "profile.json", "w", encoding="utf-8") as f:
            json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)

        # Сохраняем аудио
        sf.write(profile_dir / "reference.wav", wav, sr)

        # Сохраняем voice prompt (тензоры)
        torch.save({
            "ref_code": voice_prompt.ref_code,
            "ref_spk_embedding": voice_prompt.ref_spk_embedding,
            "x_vector_only_mode": voice_prompt.x_vector_only_mode,
            "icl_mode": voice_prompt.icl_mode,
            "ref_text": voice_prompt.ref_text
        }, profile_dir / "voice_prompt.pt")

        # Обновляем кэш
        voice_profiles_cache[name] = voice_prompt

        return f"Профиль '{name}' успешно сохранён!"

    except Exception as e:
        return f"Ошибка сохранения профиля: {e}"


def load_voice_profile(name: str) -> Tuple[Optional[VoiceClonePromptItem], str]:
    """Загрузка профиля голоса."""
    try:
        # Проверяем кэш
        if name in voice_profiles_cache:
            return voice_profiles_cache[name], f"Профиль '{name}' загружен из кэша."

        profile_dir = PROFILES_DIR / name
        if not profile_dir.exists():
            return None, f"Профиль '{name}' не найден."

        # Загружаем voice prompt
        data = torch.load(profile_dir / "voice_prompt.pt", map_location="cpu")

        voice_prompt = VoiceClonePromptItem(
            ref_code=data["ref_code"],
            ref_spk_embedding=data["ref_spk_embedding"],
            x_vector_only_mode=data["x_vector_only_mode"],
            icl_mode=data["icl_mode"],
            ref_text=data.get("ref_text")
        )

        # Сохраняем в кэш
        voice_profiles_cache[name] = voice_prompt

        return voice_prompt, f"Профиль '{name}' успешно загружен!"

    except Exception as e:
        return None, f"Ошибка загрузки профиля: {e}"


def list_voice_profiles() -> List[str]:
    """Получение списка сохранённых профилей."""
    profiles = []
    for path in PROFILES_DIR.iterdir():
        if path.is_dir() and (path / "profile.json").exists():
            profiles.append(path.name)
    return sorted(profiles)


def delete_voice_profile(name: str) -> str:
    """Удаление профиля голоса."""
    try:
        import shutil
        profile_dir = PROFILES_DIR / name
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
            if name in voice_profiles_cache:
                del voice_profiles_cache[name]
            return f"Профиль '{name}' удалён."
        return f"Профиль '{name}' не найден."
    except Exception as e:
        return f"Ошибка удаления профиля: {e}"


# =====================================================
# Загрузка голосов из облака
# =====================================================

CLOUD_VOICES_REPO = "Slait/russia_voices"
CLOUD_VOICES_BASE_URL = "https://huggingface.co/datasets/Slait/russia_voices/resolve/main"

# Список всех доступных голосов (обновляется при загрузке)
CLOUD_VOICES_CACHE: List[str] = []

def get_cloud_voices_list() -> Tuple[List[str], str]:
    """Получение полного списка голосов из облака."""
    global CLOUD_VOICES_CACHE
    from huggingface_hub import list_repo_files

    try:
        files = list(list_repo_files(CLOUD_VOICES_REPO, repo_type="dataset"))
        voices = [f[:-4] for f in files if f.endswith(".mp3")]
        voice_list = sorted(voices)
        CLOUD_VOICES_CACHE = voice_list
        return voice_list, f"Найдено {len(voice_list)} голосов. Репозиторий: {CLOUD_VOICES_REPO}"
    except Exception as e:
        return [], f"Ошибка загрузки списка: {e}"


def download_cloud_voice(voice_name: str) -> str:
    """Загрузка голоса из облака."""
    import requests

    try:
        # Скачиваем MP3 файл
        mp3_url = f"{CLOUD_VOICES_BASE_URL}/{voice_name}.mp3?download=true"
        txt_url = f"{CLOUD_VOICES_BASE_URL}/{voice_name}.txt?download=true"

        # Скачиваем аудио
        response = requests.get(mp3_url, timeout=60, stream=True)
        response.raise_for_status()

        mp3_path = VOICES_DIR / f"{voice_name}.mp3"
        with open(mp3_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Пробуем скачать текст
        try:
            txt_response = requests.get(txt_url, timeout=30)
            if txt_response.status_code == 200:
                txt_path = VOICES_DIR / f"{voice_name}.txt"
                txt_path.write_text(txt_response.text, encoding="utf-8")
        except:
            pass

        return f"Голос '{voice_name}' успешно загружен!"

    except Exception as e:
        return f"Ошибка загрузки голоса '{voice_name}': {e}"


# =====================================================
# Вспомогательные функции
# =====================================================

_TLD = (
    "com|org|net|edu|gov|io|ai|app|dev|me|tv|cc|co|info|biz|pro|xyz|online|"
    "site|blog|news|media|club|shop|store|live|link|top|fun|space|tech|"
    "ru|su|ua|by|kz|uz|am|ge|md|"
    "uk|de|fr|it|es|nl|pl|cz|sk|hu|ro|bg|rs|hr|lt|lv|ee|fi|se|no|dk|"
    "tr|il|in|cn|jp|kr|au|nz|br|mx|us|ca|ly|to|fm|gg|page"
)
_HOST = rf"(?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?\.)+(?:{_TLD})"
_URL_TAIL = r"(?:[^\s<>\]\)\"']*)?"
_MD_LINK_RE = re.compile(
    rf"\[([^\]]+)\]\((https?://[^\s)]+|www\.[^\s)]+|{_HOST}(?:/[^\s)]*)?)\)",
    re.IGNORECASE,
)
_BARE_URL_RE = re.compile(
    rf"(?P<url>https?://[^\s<>\]\)\"']+|www\.[^\s<>\]\)\"']+|(?<![\w./@-]){_HOST}(?::\d{{2,5}})?(?:/{_URL_TAIL})?)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(
    rf"(?<![\w./-])[\w.+-]+@{_HOST}\b",
    re.IGNORECASE,
)
_AT_HANDLE_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_]{3,32})\b")


def _host_from_url(url: str) -> str:
    raw = url.strip().rstrip(".,;:!?)»\"'")
    if not re.match(r"^https?://", raw, re.IGNORECASE):
        raw = "http://" + raw
    try:
        host = (urlparse(raw).netloc or "").lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    host = host.split(":")[0]
    return host or "сайта"


def speech_limits(model_size: str = "") -> Tuple[int, float]:
    """1.7B быстрее сыпется: короче чанк и больше живой буфер."""
    if "1.7" in str(model_size):
        return 400, 12.0
    return 550, 8.0


_ONES_M = ("ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_ONES_F = ("ноль", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")
_TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто")
_HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот")
_MONTHS_GEN = (
    "", "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
_ORD_NEUT = {
    1: "первое", 2: "второе", 3: "третье", 4: "четвёртое", 5: "пятое",
    6: "шестое", 7: "седьмое", 8: "восьмое", 9: "девятое", 10: "десятое",
    11: "одиннадцатое", 12: "двенадцатое", 13: "тринадцатое", 14: "четырнадцатое",
    15: "пятнадцатое", 16: "шестнадцатое", 17: "семнадцатое", 18: "восемнадцатое",
    19: "девятнадцатое", 20: "двадцатое", 21: "двадцать первое", 22: "двадцать второе",
    23: "двадцать третье", 24: "двадцать четвёртое", 25: "двадцать пятое",
    26: "двадцать шестое", 27: "двадцать седьмое", 28: "двадцать восьмое",
    29: "двадцать девятое", 30: "тридцатое", 31: "тридцать первое",
}


def _triad_ru(n: int, feminine: bool = False) -> str:
    n = int(n)
    if n <= 0:
        return ""
    ones = _ONES_F if feminine else _ONES_M
    parts: List[str] = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_HUNDREDS[h])
    if 10 <= rem <= 19:
        parts.append(_TEENS[rem - 10])
        return " ".join(parts)
    t, o = divmod(rem, 10)
    if t:
        parts.append(_TENS[t])
    if o:
        parts.append(ones[o])
    return " ".join(parts)


def _int_ru(n: int, feminine: bool = False) -> str:
    n = int(n)
    if n < 0:
        return "минус " + _int_ru(-n, feminine)
    if n == 0:
        return "ноль"
    if n >= 1_000_000_000:
        return " ".join(_ONES_M[int(d)] if d.isdigit() else d for d in str(n))
    parts: List[str] = []
    millions, rest = divmod(n, 1_000_000)
    thousands, rest = divmod(rest, 1000)
    if millions:
        word = _triad_ru(millions, False)
        tail = "миллионов"
        if millions % 10 == 1 and millions % 100 != 11:
            tail = "миллион"
        elif millions % 10 in (2, 3, 4) and millions % 100 not in (12, 13, 14):
            tail = "миллиона"
        parts.append(f"{word} {tail}")
    if thousands:
        word = _triad_ru(thousands, True)
        tail = "тысяч"
        if thousands % 10 == 1 and thousands % 100 != 11:
            tail = "тысяча"
        elif thousands % 10 in (2, 3, 4) and thousands % 100 not in (12, 13, 14):
            tail = "тысячи"
        parts.append(f"{word} {tail}")
    if rest or not parts:
        parts.append(_triad_ru(rest, feminine) or ("ноль" if not parts else ""))
    return " ".join(x for x in parts if x)


def _year_ru(year: int) -> str:
    y = int(year)
    if 2000 <= y <= 2099:
        rem = y - 2000
        if rem == 0:
            return "две тысячи"
        return "две тысячи " + _int_ru(rem)
    if 1900 <= y <= 1999:
        rem = y - 1900
        if rem == 0:
            return "тысяча девятьсот"
        return "тысяча девятьсот " + _int_ru(rem)
    return _int_ru(y)


def _expand_numbers_for_speech(text: str) -> str:
    """Даты и числа словами: иначе 1.7B спотыкается на 27.09.2026 и годах."""

    def _date(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y < 50 else 1900
        if not (1 <= d <= 31 and 1 <= mo <= 12 and 1800 <= y <= 2100):
            return m.group(0)
        day = _ORD_NEUT.get(d) or _int_ru(d)
        return f"{day} {_MONTHS_GEN[mo]} {_year_ru(y)} года"

    text = re.sub(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2}|\d{4})\b", _date, text)

    def _time(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 23 or mi > 59:
            return m.group(0)
        return f"{_int_ru(h)} {_hours_word(h)} {_int_ru(mi, True)} {_minutes_word(mi)}"

    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", _time, text)

    def _frac(m: re.Match) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if a > 1000 or b > 1000 or b == 0:
            return m.group(0)
        return f"{_int_ru(a)} из {_int_ru(b)}"

    text = re.sub(r"\b(\d{1,4})\s*/\s*(\d{1,4})\b", _frac, text)

    def _pct(m: re.Match) -> str:
        raw = m.group(1).replace(" ", "").replace(",", ".")
        try:
            num = float(raw)
        except ValueError:
            return m.group(0)
        if num == int(num) and abs(num) < 1_000_000:
            return f"{_int_ru(int(num))} процентов"
        whole, frac = raw.split(".", 1)
        return f"{_int_ru(int(whole or 0))} запятая {_int_ru(int(frac))} процентов"

    text = re.sub(r"\b(\d[\d\s]*,?\d*)\s*%", _pct, text)

    def _year_only(m: re.Match) -> str:
        y = int(m.group(1))
        if not (1900 <= y <= 2099):
            return m.group(0)
        tail = m.group(2) or ""
        spoken = _year_ru(y)
        if tail:
            return f"{spoken} {tail.strip()}"
        return spoken

    text = re.sub(r"\b((?:19|20)\d{2})(\s*(?:год(?:а|у|ов)?|г\.))?", _year_only, text, flags=re.IGNORECASE)

    def _dec(m: re.Match) -> str:
        if m.group(0).endswith("B") or m.group(0).endswith("b"):
            return m.group(0)
        a, b = m.group(1), m.group(2)
        if len(a) > 6 or len(b) > 4:
            return m.group(0)
        return f"{_int_ru(int(a))} запятая {_int_ru(int(b))}"

    text = re.sub(r"\b(\d{1,6})[.,](\d{1,4})(?![Bb\d])", _dec, text)

    def _plain(m: re.Match) -> str:
        raw = m.group(0).replace(" ", "").replace("\u00a0", "")
        if len(raw) > 7:
            return " ".join(_ONES_M[int(ch)] for ch in raw if ch.isdigit())
        return _int_ru(int(raw))

    text = re.sub(r"\b\d[\d\s\u00a0]{0,10}\d\b|\b\d\b", _plain, text)
    return text


def _hours_word(h: int) -> str:
    if 11 <= (h % 100) <= 14:
        return "часов"
    if h % 10 == 1:
        return "час"
    if h % 10 in (2, 3, 4):
        return "часа"
    return "часов"


def _minutes_word(m: int) -> str:
    if 11 <= (m % 100) <= 14:
        return "минут"
    if m % 10 == 1:
        return "минута"
    if m % 10 in (2, 3, 4):
        return "минуты"
    return "минут"


def _spoken_link(url: str, label: Optional[str] = None) -> str:
    host = _host_from_url(url)
    if label:
        label = re.sub(r"\s+", " ", label).strip()
        if label and not _BARE_URL_RE.fullmatch(label):
            return f"{label} (ссылка на {host})"
    return f"ссылка на {host}"


def sanitize_text_for_speech(text: str) -> str:
    """Убирает сырые URL и @ники: иначе модель зачитывает слэши и начинает галлюцинировать."""
    text = text.replace("\ufeff", "").replace("\u200b", "")
    held: List[str] = []

    def _hold(spoken: str) -> str:
        held.append(spoken)
        return f"\x00L{len(held) - 1}\x00"

    text = _MD_LINK_RE.sub(lambda m: _hold(_spoken_link(m.group(2), m.group(1))), text)
    text = _EMAIL_RE.sub(lambda _m: _hold("электронная почта"), text)
    text = _BARE_URL_RE.sub(lambda m: _hold(_spoken_link(m.group("url"))), text)
    text = _AT_HANDLE_RE.sub(lambda m: _hold(f"аккаунт {m.group(1)}"), text)
    for i, spoken in enumerate(held):
        text = text.replace(f"\x00L{i}\x00", spoken)
    text = re.sub(r"\b(\d+)[.,](\d+)[Bb]\b", r"\1 пункт \2 би", text)
    text = _expand_numbers_for_speech(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_into_chunks(text: str, max_chars: int = 1500) -> List[str]:
    """Разбивает длинный текст на части по границам предложений."""
    text = sanitize_text_for_speech(text.strip())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Разбиваем по концам предложений (. ! ? …)
    sentences = re.split(r'(?<=[.!?…])\s+', text)

    chunks = []
    current = ""
    for sentence in sentences:
        # Если предложение само длиннее лимита — режем по запятым
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            sub = ""
            for part in parts:
                if len(sub) + len(part) + 1 <= max_chars:
                    sub = (sub + " " + part).strip() if sub else part
                else:
                    if sub:
                        chunks.append(sub)
                    sub = part
            if sub:
                current = sub
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks

def get_device():
    """Определение устройства для вычислений."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model_path(model_type: str, model_size: str) -> str:
    """Получение пути к модели."""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


def get_backend(model_type: str, model_size: str):
    """Бэкенд генерации: CUDA Graphs если есть, иначе официальный qwen_tts."""
    return get_tts_backend(model_type, model_size, get_model_path(model_type, model_size))


def get_model(model_type: str, model_size: str) -> Qwen3TTSModel:
    """Официальный wrapper модели (для профилей и prompt). Не грузим второй инстанс."""
    return get_backend(model_type, model_size).inner


def resolve_language(language: str) -> str:
    for code, name in LANGUAGES.items():
        if name == language:
            return code
    return "Auto"


def resolve_speaker(speaker: str) -> str:
    for sid, sname in SPEAKERS.items():
        if sname == speaker:
            return sid.lower()
    if speaker:
        return speaker.split()[0].lower()
    return "vivian"


def make_result_audio():
    """Выходной плеер. Без Gradio streaming: в портативке нет ffmpeg, из-за него был WinError 2."""
    return gr.Audio(
        label="Результат",
        type="filepath",
        interactive=False,
        autoplay=False,
        format="wav",
    )


def normalize_audio(wav, eps=1e-12, clip=True):
    """Нормализация аудио в диапазон [-1, 1]."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Неподдерживаемый тип данных: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def audio_to_tuple(audio) -> Optional[Tuple[np.ndarray, int]]:
    """Конвертация аудио Gradio в кортеж (wav, sr)."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = normalize_audio(audio["data"])
        return wav, sr

    return None


def save_audio_file(audio_data: np.ndarray, sample_rate: int, output_dir: str = None) -> str:
    """Сохранение аудио в файл."""
    if output_dir is None:
        output_dir = str(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"qwen3_tts_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath, audio_data, sample_rate)
    return filepath


# =====================================================
# Multi-speaker парсер
# =====================================================

def parse_multi_speaker_script(script: str) -> List[Tuple[int, str]]:
    """
    Парсинг скрипта с несколькими дикторами.
    Формат: "Speaker N: текст" или "Диктор N: текст"

    Возвращает список кортежей (speaker_id, text)
    """
    lines = script.strip().split('\n')
    result = []

    # Паттерны для парсинга
    patterns = [
        r'^Speaker\s*(\d+)\s*:\s*(.+)$',
        r'^Диктор\s*(\d+)\s*:\s*(.+)$',
        r'^Голос\s*(\d+)\s*:\s*(.+)$',
        r'^\[(\d+)\]\s*(.+)$',
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                text = match.group(2).strip()
                result.append((speaker_id, text))
                matched = True
                break

        if not matched:
            # Если формат не распознан, добавляем к Speaker 0
            result.append((0, line))

    return result


# =====================================================
# Функции генерации
# =====================================================

def generate_voice_design(
    text: str,
    language: str,
    voice_description: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    autoplay: bool = True,
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Генерация речи с дизайном голоса (стриминг в плеер)."""
    global is_generating, stop_generation

    if not text or not text.strip():
        yield None, "Ошибка: Введите текст для синтеза.", ""
        return

    if not voice_description or not voice_description.strip():
        yield None, "Ошибка: Введите описание голоса.", ""
        return

    if model_size != "1.7B":
        yield None, "Ошибка: Дизайн голоса доступен только для модели 1.7B.", ""
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели VoiceDesign...", ""
        backend = get_backend("VoiceDesign", model_size)
        lang_code = resolve_language(language)
        chunk_chars, preroll = speech_limits(model_size)
        chunks = split_text_into_chunks(text.strip(), max_chars=chunk_chars)
        yield None, f"{engine_status_line()}\nТекст разбит на {len(chunks)} частей.", ""

        def generate_chunk(chunk: str):
            yield from backend.stream_design(
                text=chunk,
                language=lang_code,
                instruct=voice_description.strip(),
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        yield from stream_pcm_to_ui(
            chunks,
            generate_chunk,
            output_path=new_output_path(OUTPUT_DIR),
            autoplay=bool(autoplay),
            stop_fn=lambda: stop_generation,
            title="Дизайн голоса",
            engine_name=backend.engine,
            preroll_sec=preroll,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"Ошибка: {type(e).__name__}: {e}", ""
    finally:
        is_generating = False


def generate_voice_clone(
    ref_audio,
    ref_text: str,
    target_text: str,
    language: str,
    use_xvector_only: bool,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    autoplay: bool = True,
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Клонирование голоса с нарезкой и стримом в плеер."""
    global is_generating, stop_generation

    if not target_text or not target_text.strip():
        yield None, "Ошибка: Введите текст для синтеза.", ""
        return

    audio_tuple = audio_to_tuple(ref_audio)
    if audio_tuple is None:
        yield None, "Ошибка: Загрузите референсное аудио.", ""
        return

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        yield None, "Ошибка: Введите текст референсного аудио или включите режим 'Только x-vector'.", ""
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели Base...", ""
        backend = get_backend("Base", model_size)
        lang_code = resolve_language(language)
        chunk_chars, preroll = speech_limits(model_size)
        chunks = split_text_into_chunks(target_text.strip(), max_chars=chunk_chars)
        yield None, f"{engine_status_line()}\nСчитаю voice prompt один раз...", ""

        prompt = backend.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
        )
        yield None, f"Текст разбит на {len(chunks)} частей. Начинаю генерацию...", ""

        def generate_chunk(chunk: str):
            yield from backend.stream_clone(
                text=chunk,
                language=lang_code,
                voice_clone_prompt=prompt,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        yield from stream_pcm_to_ui(
            chunks,
            generate_chunk,
            output_path=new_output_path(OUTPUT_DIR),
            autoplay=bool(autoplay),
            stop_fn=lambda: stop_generation,
            title="Клонирование голоса",
            engine_name=backend.engine,
            preroll_sec=preroll,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"Ошибка: {type(e).__name__}: {e}", ""
    finally:
        is_generating = False


def generate_with_profile(
    profile_name: str,
    target_text: str,
    language: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    autoplay: bool = True,
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Генерация с сохранённым профилем голоса."""
    global is_generating, stop_generation

    if not target_text or not target_text.strip():
        yield None, "Ошибка: Введите текст для синтеза.", ""
        return

    if not profile_name:
        yield None, "Ошибка: Выберите профиль голоса.", ""
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, f"Загрузка профиля '{profile_name}'...", ""
        voice_prompt, load_msg = load_voice_profile(profile_name)
        if voice_prompt is None:
            yield None, load_msg, ""
            return

        yield None, "Загрузка модели Base...", ""
        backend = get_backend("Base", model_size)
        lang_code = resolve_language(language)
        chunk_chars, preroll = speech_limits(model_size)
        chunks = split_text_into_chunks(target_text.strip(), max_chars=chunk_chars)

        def generate_chunk(chunk: str):
            yield from backend.stream_clone(
                text=chunk,
                language=lang_code,
                voice_clone_prompt=voice_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        yield from stream_pcm_to_ui(
            chunks,
            generate_chunk,
            output_path=new_output_path(OUTPUT_DIR),
            autoplay=bool(autoplay),
            stop_fn=lambda: stop_generation,
            title=f"Профиль {profile_name}",
            engine_name=backend.engine,
            preroll_sec=preroll,
        )
    except Exception as e:
        yield None, f"Ошибка: {type(e).__name__}: {e}", ""
    finally:
        is_generating = False


def generate_custom_voice(
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    autoplay: bool = True,
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Пресеты голосов с нарезкой и стримом в плеер."""
    global is_generating, stop_generation

    if not text or not text.strip():
        yield None, "Ошибка: Введите текст для синтеза.", ""
        return

    if not speaker:
        yield None, "Ошибка: Выберите голос.", ""
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели CustomVoice...", ""
        backend = get_backend("CustomVoice", model_size)
        speaker_id = resolve_speaker(speaker)
        lang_code = resolve_language(language)
        chunk_chars, preroll = speech_limits(model_size)
        chunks = split_text_into_chunks(text.strip(), max_chars=chunk_chars)
        yield None, f"{engine_status_line()}\nТекст разбит на {len(chunks)} частей.", ""

        def generate_chunk(chunk: str):
            yield from backend.stream_custom(
                text=chunk,
                language=lang_code,
                speaker=speaker_id,
                instruct=instruct.strip() if instruct else None,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        yield from stream_pcm_to_ui(
            chunks,
            generate_chunk,
            output_path=new_output_path(OUTPUT_DIR),
            autoplay=bool(autoplay),
            stop_fn=lambda: stop_generation,
            title="Пресет голоса",
            engine_name=backend.engine,
            preroll_sec=preroll,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"Ошибка: {type(e).__name__}: {e}", ""
    finally:
        is_generating = False


def generate_multi_speaker(
    script: str,
    num_speakers: int,
    speaker_audios: List,
    speaker_texts: List[str],
    language: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    autoplay: bool = True,
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Диалог нескольких дикторов со стримом в плеер."""
    global is_generating, stop_generation

    if not script or not script.strip():
        yield None, "Ошибка: Введите сценарий диалога.", ""
        return

    parsed_lines = parse_multi_speaker_script(script)
    if not parsed_lines:
        yield None, "Ошибка: Не удалось распознать формат сценария.", ""
        return

    used_speakers = set(sp for sp, _ in parsed_lines)
    for sp in used_speakers:
        if sp >= num_speakers:
            yield None, f"Ошибка: В сценарии используется Диктор {sp}, но настроено только {num_speakers} дикторов.", ""
            return
        audio = speaker_audios[sp] if sp < len(speaker_audios) else None
        if audio_to_tuple(audio) is None:
            yield None, f"Ошибка: Не загружено аудио для Диктора {sp}.", ""
            return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели Base...", ""
        backend = get_backend("Base", model_size)
        lang_code = resolve_language(language)

        yield None, "Создание профилей голосов для дикторов...", ""
        voice_prompts = {}
        for sp in used_speakers:
            audio_tuple = audio_to_tuple(speaker_audios[sp])
            ref_text = speaker_texts[sp] if sp < len(speaker_texts) else None
            voice_prompts[sp] = backend.create_voice_clone_prompt(
                ref_audio=audio_tuple,
                ref_text=ref_text.strip() if ref_text else None,
                x_vector_only_mode=not bool(ref_text),
            )

        line_texts = [sanitize_text_for_speech(text) for _, text in parsed_lines]
        prompt_by_index = [voice_prompts[sp] for sp, _ in parsed_lines]

        def generate_chunk(chunk: str):
            idx = generate_chunk.i
            generate_chunk.i += 1
            yield from backend.stream_clone(
                text=chunk,
                language=lang_code,
                voice_clone_prompt=prompt_by_index[idx],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        generate_chunk.i = 0

        yield from stream_pcm_to_ui(
            line_texts,
            generate_chunk,
            output_path=new_output_path(OUTPUT_DIR),
            autoplay=bool(autoplay),
            stop_fn=lambda: stop_generation,
            title="Multi-speaker",
            engine_name=backend.engine,
            pause_sec=0.3,
            preroll_sec=speech_limits(model_size)[1],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, f"Ошибка: {type(e).__name__}: {e}", ""
    finally:
        is_generating = False


def stop_generation_fn():
    """Полный стоп: рвём GPU-поток и гасим живой плеер."""
    global stop_generation
    stop_generation = True
    return (
        gr.update(value=None, autoplay=False),
        "Остановлено. Генерация и воспроизведение прерваны.",
        live_stop_cmd(int(time.time() * 1000)),
    )


# =====================================================
# Локальные голоса
# =====================================================

def get_local_voices() -> Dict[str, str]:
    """Получение списка локальных голосов (включая подпапки)."""
    voices = {}
    supported_ext = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    # Рекурсивный поиск во всех подпапках
    for path in VOICES_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported_ext:
            voices[path.stem] = str(path)

    return dict(sorted(voices.items()))


def get_voice_text(voice_name: str) -> Optional[str]:
    """Получение текста для голоса (если есть)."""
    # Ищем в основной папке
    txt_path = VOICES_DIR / f"{voice_name}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()
    # Ищем в подпапках
    for txt_path in VOICES_DIR.rglob(f"{voice_name}.txt"):
        return txt_path.read_text(encoding="utf-8").strip()
    return None


def get_first_ru_voice(voices: Dict[str, str]) -> Optional[str]:
    """Получение первого голоса, начинающегося с RU_."""
    for name in sorted(voices.keys()):
        if name.upper().startswith("RU_"):
            return name
    return None


def get_random_ru_voices(voices: Dict[str, str], count: int = 2) -> List[str]:
    """Получение случайных голосов, начинающихся с RU_."""
    import random
    ru_voices = [name for name in voices.keys() if name.upper().startswith("RU_")]
    if len(ru_voices) < count:
        return ru_voices + [None] * (count - len(ru_voices))
    return random.sample(ru_voices, count)


# =====================================================
# Построение интерфейса
# =====================================================

def build_ui():
    """Построение интерфейса Gradio."""

    # CSS стили
    css = """
    .gradio-container {max-width: none !important;}

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
    }

    .settings-card {
        background: rgba(30, 41, 59, 0.95) !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .generation-card {
        background: rgba(30, 41, 59, 0.95) !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .speaker-block {
        background: #1e293b !important;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }

    .tab-nav button {
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
    }

    /* Исправление белых рамок */
    .examples-table, .examples-table tbody, .examples-table tr, .examples-table td {
        background: transparent !important;
        border: none !important;
    }

    .prose {
        color: #e2e8f0 !important;
    }

    /* Куски живого эфира должны оставаться в DOM: visible=False у Gradio часто без textarea. */
    #tts_live_chunk {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    """

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        primary_hue="indigo",
        secondary_hue="purple",
    )

    # Gradio 6: js/css/theme только в launch(). head — обычный <script>, иначе () => {} не вызывается.
    head_html = """
    <script>
    (function () {
        if (window.__ttsInit) return;
        window.__ttsInit = true;
        window.__tts = { ctx: null, next: 0, sources: [], last: '', paused: false };
        const AC = window.AudioContext || window.webkitAudioContext;
        function queueBuf(buf) {
            const t = window.__tts;
            const src = t.ctx.createBufferSource();
            src.buffer = buf;
            src.connect(t.ctx.destination);
            const now = t.ctx.currentTime;
            if (t.next < now + 0.03) t.next = now + 0.03;
            src.start(t.next);
            t.next += buf.duration;
            t.sources.push(src);
            src.onended = () => { t.sources = t.sources.filter((x) => x !== src); };
        }
        window.ttsArm = () => {
            const t = window.__tts;
            if (!t.ctx || t.ctx.state === 'closed') {
                t.ctx = new AC();
                t.next = 0;
                t.sources = [];
            }
            if (!t.paused) t.ctx.resume();
        };
        window.ttsTogglePause = () => {
            const t = window.__tts;
            if (!t.ctx) return;
            if (t.ctx.state === 'running') {
                t.paused = true;
                t.ctx.suspend();
            } else {
                t.paused = false;
                t.ctx.resume();
            }
        };
        window.ttsStop = () => {
            const t = window.__tts;
            (t.sources || []).forEach((s) => { try { s.stop(); } catch (e) {} });
            t.sources = [];
            t.next = 0;
            t.paused = false;
            if (t.ctx && t.ctx.state !== 'closed') {
                try { t.ctx.suspend(); } catch (e) {}
            }
        };
        window.ttsPushFile = async (name) => {
            if (!name) return;
            window.ttsArm();
            const res = await fetch('/tts_live/' + encodeURIComponent(name));
            if (!res.ok) { console.warn('[tts] live fetch', res.status, name); return; }
            const arr = await res.arrayBuffer();
            const buf = await window.__tts.ctx.decodeAudioData(arr.slice(0));
            queueBuf(buf);
        };
        window.ttsHandle = (payload) => {
            if (!payload) return;
            let msg = payload;
            try { msg = JSON.parse(payload); } catch (e) { return; }
            if (msg.cmd === 'stop') { window.ttsStop(); return; }
            if (msg.cmd === 'push' && msg.file) window.ttsPushFile(msg.file);
        };
        const watchLive = () => {
            const root = document.getElementById('tts_live_chunk');
            if (!root) return;
            const el = root.querySelector('textarea') || root.querySelector('input') || root;
            if (!el) return;
            const v = (el.value !== undefined && el.value !== '') ? el.value : (el.textContent || '');
            if (v && v !== window.__tts.last) {
                window.__tts.last = v;
                window.ttsHandle(v);
            }
        };
        setInterval(watchLive, 120);
        console.log('[tts] live player ready');
    })();
    </script>
    """

    with gr.Blocks(title=APP_NAME) as demo:
        # Заголовок
        gr.HTML(f"""
        <div class="main-header">
            <h1>{APP_NAME} v{APP_VERSION}</h1>
            <p>Синтез речи с клонированием голосов и Multi-speaker режимом</p>
            <p style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.5rem;">
                Собрал <a href="https://t.me/nerual_dreming" target="_blank" style="color: white;">Nerual Dreaming</a> — основатель <a href="https://artgeneration.me/" target="_blank" style="color: white;">ArtGeneration.me</a>, техноблогер и нейро-евангелист.
            </p>
            <p style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.3rem;">
                <a href="https://t.me/neuroport" target="_blank" style="color: white;">Нейро-Софт</a> — репаки и портативки полезных нейросетей
            </p>
        </div>
        """)

        # Глобальная настройка автовоспроизведения
        with gr.Row():
            autoplay_checkbox = gr.Checkbox(
                label="Автовоспроизведение",
                value=True,
                info="Живой эфир через Web Audio: без заиканий на стыке кусков. Выкл — только файл.",
            )
        live_chunk = gr.Textbox(
            label="live",
            visible=True,
            show_label=False,
            lines=1,
            max_lines=1,
            elem_id="tts_live_chunk",
        )
        # Gradio 6 гарантированно исполняет js= на клике. head/Blocks js могут молчать.
        TTS_ARM_JS = r"""
(...args) => {
    if (!window.__ttsInit) {
        window.__ttsInit = true;
        window.__tts = { ctx: null, next: 0, sources: [], last: '', paused: false };
        const AC = window.AudioContext || window.webkitAudioContext;
        function queueBuf(buf) {
            const t = window.__tts;
            const src = t.ctx.createBufferSource();
            src.buffer = buf;
            src.connect(t.ctx.destination);
            const now = t.ctx.currentTime;
            if (t.next < now + 0.03) t.next = now + 0.03;
            src.start(t.next);
            t.next += buf.duration;
            t.sources.push(src);
            src.onended = () => { t.sources = t.sources.filter((x) => x !== src); };
        }
        window.ttsArm = () => {
            const t = window.__tts;
            if (!t.ctx || t.ctx.state === 'closed') {
                t.ctx = new AC();
                t.next = 0;
                t.sources = [];
            }
            if (!t.paused) t.ctx.resume();
        };
        window.ttsTogglePause = () => {
            const t = window.__tts;
            if (!t.ctx) return;
            if (t.ctx.state === 'running') {
                t.paused = true;
                t.ctx.suspend();
            } else {
                t.paused = false;
                t.ctx.resume();
            }
        };
        window.ttsStop = () => {
            const t = window.__tts;
            (t.sources || []).forEach((s) => { try { s.stop(); } catch (e) {} });
            t.sources = [];
            t.next = 0;
            t.paused = false;
            if (t.ctx && t.ctx.state !== 'closed') {
                try { t.ctx.suspend(); } catch (e) {}
            }
        };
        window.ttsPushFile = async (name) => {
            if (!name) return;
            window.ttsArm();
            const res = await fetch('/tts_live/' + encodeURIComponent(name));
            if (!res.ok) { console.warn('[tts] live fetch', res.status, name); return; }
            const arr = await res.arrayBuffer();
            const buf = await window.__tts.ctx.decodeAudioData(arr.slice(0));
            queueBuf(buf);
        };
        window.ttsHandle = (payload) => {
            if (!payload) return;
            let msg = payload;
            try { msg = JSON.parse(payload); } catch (e) { return; }
            if (msg.cmd === 'stop') { window.ttsStop(); return; }
            if (msg.cmd === 'push' && msg.file) window.ttsPushFile(msg.file);
        };
        const watchLive = () => {
            const root = document.getElementById('tts_live_chunk');
            if (!root) return;
            const el = root.querySelector('textarea') || root.querySelector('input') || root;
            if (!el) return;
            const v = (el.value !== undefined && el.value !== '') ? el.value : (el.textContent || '');
            if (v && v !== window.__tts.last) {
                window.__tts.last = v;
                window.ttsHandle(v);
            }
        };
        setInterval(watchLive, 120);
        console.log('[tts] live player armed');
    }
    if (window.ttsArm) window.ttsArm();
    return args;
}
"""
        TTS_PAUSE_JS = "(...args) => { if (window.ttsTogglePause) window.ttsTogglePause(); return args; }"

        with gr.Tabs() as tabs:
            # =====================================================
            # Вкладка 1: Пресеты голосов (CustomVoice)
            # =====================================================
            with gr.Tab("Пресеты голосов", id="custom"):
                gr.Markdown("### Синтез речи с предустановленными голосами")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        cv_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить...",
                            value="Привет! Это демонстрация системы синтеза речи Qwen3-TTS, портативная версия от канала Нейро-софт. Она поддерживает русский язык и множество других языков."
                        )

                        with gr.Row():
                            cv_language = gr.Dropdown(
                                label="Язык",
                                choices=list(LANGUAGES.values()),
                                value=LANGUAGES["Russian"],
                                interactive=True,
                            )
                            cv_speaker = gr.Dropdown(
                                label="Голос",
                                choices=list(SPEAKERS.values()),
                                value=SPEAKERS["Vivian"],
                                interactive=True,
                            )

                        cv_instruct = gr.Textbox(
                            label="Стиль (опционально)",
                            lines=2,
                            placeholder="Например: Говорить радостно и энергично",
                        )

                        with gr.Row():
                            cv_model_size = gr.Dropdown(
                                label="Размер модели",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )

                        with gr.Accordion("Параметры генерации", open=False):
                            cv_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            cv_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.45, step=0.1
                            )
                            cv_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            cv_generate_btn = gr.Button("Сгенерировать", variant="primary", scale=2)
                            cv_pause_btn = gr.Button("Пауза / далее", scale=1)
                            cv_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        cv_audio_out = make_result_audio()
                        cv_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                        # Примеры стилей
                        gr.Markdown("**Примеры стилей** (кликни для применения)")
                        gr.Examples(
                            examples=[
                                ["Speak with warmth and a gentle smile"],
                                ["Speak with excitement and high energy"],
                                ["Speak slowly and calmly, relaxed tone"],
                                ["Speak with authority and confidence"],
                                ["Speak softly, like telling a secret"],
                                ["Speak with sadness in voice"],
                                ["Speak angrily with sharp emphasis"],
                                ["Speak in a playful, teasing manner"],
                            ],
                            inputs=[cv_instruct],
                            label=""
                        )

                cv_gen_evt = cv_generate_btn.click(
                    generate_custom_voice,
                    inputs=[cv_text, cv_language, cv_speaker, cv_instruct, cv_model_size, cv_max_tokens, cv_temperature, cv_top_p, autoplay_checkbox],
                    outputs=[cv_audio_out, cv_status, live_chunk],
                    js=TTS_ARM_JS,
                )
                cv_pause_btn.click(fn=None, js=TTS_PAUSE_JS)
                cv_stop_btn.click(
                    stop_generation_fn,
                    outputs=[cv_audio_out, cv_status, live_chunk],
                    cancels=[cv_gen_evt],
                )

            # =====================================================
            # Вкладка 2: Клонирование голоса (Base)
            # =====================================================
            with gr.Tab("Клонирование голоса", id="clone"):
                gr.Markdown("### Клонирование голоса из референсного аудио")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        # Выбор голоса из библиотеки
                        local_voices = get_local_voices()
                        first_ru_voice = get_first_ru_voice(local_voices)

                        # Предзагрузка аудио и текста для дефолтного голоса
                        default_vc_audio, default_vc_text = None, ""
                        if first_ru_voice and first_ru_voice in local_voices:
                            path = local_voices[first_ru_voice]
                            import soundfile as sf
                            wav, sr = sf.read(path)
                            default_vc_audio = (sr, wav)
                            default_vc_text = get_voice_text(first_ru_voice) or ""

                        vc_voice_preset = gr.Dropdown(
                            label="Выбрать голос из библиотеки",
                            choices=["-- Загрузить свой --"] + list(local_voices.keys()),
                            value=first_ru_voice if first_ru_voice else "-- Загрузить свой --",
                            interactive=True,
                        )
                        vc_refresh_voices = gr.Button("Обновить список", size="sm")

                        vc_ref_audio = gr.Audio(
                            label="Референсное аудио (голос для клонирования)",
                            type="numpy",
                            sources=["upload", "microphone"],
                            value=default_vc_audio,
                        )
                        vc_ref_text = gr.Textbox(
                            label="Текст референсного аудио",
                            lines=2,
                            placeholder="Введите текст, который произносится в референсном аудио...",
                            value=default_vc_text,
                        )

                        def load_voice_preset(voice_name):
                            if voice_name == "-- Загрузить свой --" or not voice_name:
                                return None, ""
                            voices = get_local_voices()
                            path = voices.get(voice_name)
                            if path:
                                import soundfile as sf
                                wav, sr = sf.read(path)
                                ref_text = get_voice_text(voice_name) or ""
                                return (sr, wav), ref_text
                            return None, ""

                        def refresh_voice_list():
                            voices = get_local_voices()
                            return gr.update(choices=["-- Загрузить свой --"] + list(voices.keys()))

                        vc_voice_preset.change(
                            load_voice_preset,
                            inputs=[vc_voice_preset],
                            outputs=[vc_ref_audio, vc_ref_text],
                        )
                        vc_refresh_voices.click(refresh_voice_list, outputs=[vc_voice_preset])

                        vc_xvector_only = gr.Checkbox(
                            label="Только x-vector (без текста референса, качество ниже)",
                            value=False,
                        )

                        gr.Markdown("---")

                        vc_target_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить клонированным голосом...",
                            value="Ничего себе, мой голос клонированный с помощью Qwen3-TTS звучит почти как настоящий!",
                        )

                        with gr.Row():
                            vc_language = gr.Dropdown(
                                label="Язык",
                                choices=list(LANGUAGES.values()),
                                value=LANGUAGES["Auto"],
                                interactive=True,
                            )
                            vc_model_size = gr.Dropdown(
                                label="Размер модели",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )

                        with gr.Accordion("Параметры генерации", open=False):
                            vc_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            vc_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.45, step=0.1
                            )
                            vc_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            vc_generate_btn = gr.Button("Клонировать и озвучить", variant="primary", scale=2)
                            vc_pause_btn = gr.Button("Пауза / далее", scale=1)
                            vc_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        vc_audio_out = make_result_audio()
                        vc_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                        # Загрузка голосов из облака
                        with gr.Accordion("Загрузить голоса из облака", open=False):
                            gr.Markdown(f"*Репозиторий: `{CLOUD_VOICES_REPO}`*")

                            vc_cloud_status = gr.Textbox(
                                label="Статус",
                                interactive=False,
                                value="Нажмите 'Загрузить список' для получения доступных голосов",
                            )

                            vc_load_cloud_btn = gr.Button("Загрузить список", variant="secondary")

                            vc_cloud_voices = gr.CheckboxGroup(
                                label="Доступные голоса",
                                choices=[],
                                interactive=True,
                            )

                            vc_download_btn = gr.Button("Скачать выбранные", variant="primary")
                            vc_download_status = gr.Textbox(
                                label="Результат загрузки",
                                interactive=False,
                            )

                            def load_cloud_list_vc():
                                voices, status = get_cloud_voices_list()
                                if voices:
                                    return status, gr.update(choices=voices, value=[])
                                return status, gr.update(choices=[], value=[])

                            vc_load_cloud_btn.click(
                                load_cloud_list_vc,
                                outputs=[vc_cloud_status, vc_cloud_voices],
                            )

                            def download_selected_voices_vc(selected):
                                if not selected:
                                    return "Выберите голоса для загрузки."
                                results = []
                                for voice in selected:
                                    result = download_cloud_voice(voice)
                                    results.append(result)
                                return "\n".join(results)

                            vc_download_btn.click(
                                download_selected_voices_vc,
                                inputs=[vc_cloud_voices],
                                outputs=[vc_download_status],
                            )

                vc_gen_evt = vc_generate_btn.click(
                    generate_voice_clone,
                    inputs=[vc_ref_audio, vc_ref_text, vc_target_text, vc_language, vc_xvector_only, vc_model_size, vc_max_tokens, vc_temperature, vc_top_p, autoplay_checkbox],
                    outputs=[vc_audio_out, vc_status, live_chunk],
                    js=TTS_ARM_JS,
                )
                vc_pause_btn.click(fn=None, js=TTS_PAUSE_JS)
                vc_stop_btn.click(
                    stop_generation_fn,
                    outputs=[vc_audio_out, vc_status, live_chunk],
                    cancels=[vc_gen_evt],
                )

            # =====================================================
            # Вкладка 3: Multi-speaker
            # =====================================================
            with gr.Tab("Multi-speaker", id="multi"):
                gr.Markdown("### Генерация диалога с несколькими дикторами")
                gr.Markdown("""
                **Формат сценария:**
                ```
                Speaker 0: Привет, как дела?
                Speaker 1: Отлично, спасибо! А у тебя?
                Speaker 0: Тоже хорошо!
                ```
                Также поддерживаются форматы: `Диктор N:`, `Голос N:`, `[N]`
                """)

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        ms_num_speakers = gr.Slider(
                            label="Количество дикторов",
                            minimum=2, maximum=4, value=2, step=1,
                        )

                        # Блоки дикторов
                        local_voices = get_local_voices()
                        voice_choices = ["-- Загрузить свой --"] + list(local_voices.keys())
                        # Автовыбор случайных RU_ голосов для дикторов
                        default_ru_voices = get_random_ru_voices(local_voices, 4)

                        # Предзагрузка аудио и текста для дефолтных голосов
                        def load_default_voice_data(voice_name):
                            if not voice_name or voice_name == "-- Загрузить свой --":
                                return None, ""
                            path = local_voices.get(voice_name)
                            if path:
                                import soundfile as sf
                                wav, sr = sf.read(path)
                                ref_text = get_voice_text(voice_name) or ""
                                return (sr, wav), ref_text
                            return None, ""

                        speaker_blocks = []
                        speaker_audios = []
                        speaker_texts = []
                        speaker_presets = []

                        for i in range(4):
                            with gr.Column(visible=(i < 2), elem_classes="speaker-block") as block:
                                gr.Markdown(f"**Диктор {i}**")
                                default_voice = default_ru_voices[i] if i < len(default_ru_voices) and default_ru_voices[i] else "-- Загрузить свой --"
                                default_audio, default_text = load_default_voice_data(default_voice)
                                preset = gr.Dropdown(
                                    label="Пресет голоса",
                                    choices=voice_choices,
                                    value=default_voice,
                                )
                                audio = gr.Audio(
                                    label="Аудио референса",
                                    type="numpy",
                                    sources=["upload", "microphone"],
                                    value=default_audio,
                                )
                                text = gr.Textbox(
                                    label="Текст референса (опционально)",
                                    lines=1,
                                    placeholder="Текст произносимый в аудио...",
                                    value=default_text,
                                )

                                # Обработчик выбора пресета
                                def update_from_preset(preset_name, idx=i):
                                    if preset_name == "-- Загрузить свой --":
                                        return gr.update(value=None), gr.update(value="")
                                    path = local_voices.get(preset_name)
                                    if path:
                                        import soundfile as sf
                                        wav, sr = sf.read(path)
                                        ref_text = get_voice_text(preset_name) or ""
                                        return gr.update(value=(sr, wav)), gr.update(value=ref_text)
                                    return gr.update(value=None), gr.update(value="")

                                preset.change(
                                    update_from_preset,
                                    inputs=[preset],
                                    outputs=[audio, text],
                                )

                                speaker_blocks.append(block)
                                speaker_audios.append(audio)
                                speaker_texts.append(text)
                                speaker_presets.append(preset)

                        # Обновление видимости блоков
                        def update_speaker_visibility(num):
                            return [gr.update(visible=(i < num)) for i in range(4)]

                        ms_num_speakers.change(
                            update_speaker_visibility,
                            inputs=[ms_num_speakers],
                            outputs=speaker_blocks,
                        )

                        gr.Markdown("---")

                        with gr.Row():
                            ms_language = gr.Dropdown(
                                label="Язык",
                                choices=list(LANGUAGES.values()),
                                value=LANGUAGES["Auto"],
                                interactive=True,
                            )
                            ms_model_size = gr.Dropdown(
                                label="Размер модели",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )

                        with gr.Accordion("Параметры генерации", open=False):
                            ms_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            ms_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.45, step=0.1
                            )
                            ms_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        ms_script = gr.Textbox(
                            label="Сценарий диалога",
                            lines=10,
                            placeholder="Speaker 0: Привет!\nSpeaker 1: Привет, как дела?",
                            value="Speaker 0: Привет! Ты уже подписался на канал нейро-софт?\nSpeaker 1: Да, там регулярно выходят портативные версии полезных нейросетей!\nSpeaker 0: Это точно, а еще классные мемы!",
                        )

                        with gr.Row():
                            ms_generate_btn = gr.Button("Сгенерировать диалог", variant="primary", scale=2)
                            ms_pause_btn = gr.Button("Пауза / далее", scale=1)
                            ms_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                        ms_audio_out = make_result_audio()
                        ms_status = gr.Textbox(
                            label="Статус",
                            lines=6,
                            interactive=False,
                        )

                # Wrapper для передачи аудио дикторов
                def multi_speaker_wrapper(script, num_speakers, audio0, audio1, audio2, audio3, text0, text1, text2, text3, language, model_size, max_tokens, temperature, top_p, autoplay):
                    audios = [audio0, audio1, audio2, audio3]
                    texts = [text0, text1, text2, text3]
                    yield from generate_multi_speaker(script, num_speakers, audios, texts, language, model_size, max_tokens, temperature, top_p, autoplay)

                ms_gen_evt = ms_generate_btn.click(
                    multi_speaker_wrapper,
                    inputs=[ms_script, ms_num_speakers,
                            speaker_audios[0], speaker_audios[1], speaker_audios[2], speaker_audios[3],
                            speaker_texts[0], speaker_texts[1], speaker_texts[2], speaker_texts[3],
                            ms_language, ms_model_size, ms_max_tokens, ms_temperature, ms_top_p, autoplay_checkbox],
                    outputs=[ms_audio_out, ms_status, live_chunk],
                    js=TTS_ARM_JS,
                )
                ms_pause_btn.click(fn=None, js=TTS_PAUSE_JS)
                ms_stop_btn.click(
                    stop_generation_fn,
                    outputs=[ms_audio_out, ms_status, live_chunk],
                    cancels=[ms_gen_evt],
                )

            # =====================================================
            # Вкладка 4: Дизайн голоса (VoiceDesign)
            # =====================================================
            with gr.Tab("Дизайн голоса", id="design"):
                gr.Markdown("### Создание голоса по текстовому описанию")
                gr.Markdown("*Доступно только для модели 1.7B*")
                gr.Markdown("**Примечание:** Описание голоса можно писать на русском, но на английском результат лучше.")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        vd_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить...",
                            value="Привет! Как твои дела? Это демонстрация синтеза речи Qwen3-TTS от канала Нейро-софт."
                        )

                        vd_language = gr.Dropdown(
                            label="Язык",
                            choices=list(LANGUAGES.values()),
                            value=LANGUAGES["Russian"],
                            interactive=True,
                        )

                        vd_description = gr.Textbox(
                            label="Описание голоса (лучше на английском)",
                            lines=3,
                            placeholder="Young female voice, warm and friendly...",
                            value="Young female voice, warm and friendly, speaking with enthusiasm"
                        )

                        vd_model_size = gr.Dropdown(
                            label="Размер модели",
                            choices=["1.7B"],
                            value="1.7B",
                            interactive=False,
                        )

                        with gr.Accordion("Параметры генерации", open=False):
                            vd_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            vd_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.45, step=0.1
                            )
                            vd_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            vd_generate_btn = gr.Button("Сгенерировать", variant="primary", scale=2)
                            vd_pause_btn = gr.Button("Пауза / далее", scale=1)
                            vd_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        vd_audio_out = make_result_audio()
                        vd_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                        # Примеры промптов
                        gr.Markdown("**Готовые промпты** (кликни для применения)")
                        gr.Examples(
                            examples=[
                                ["Female, 25 years old, warm soprano voice, speaking with a gentle smile and soft tone"],
                                ["Male, 35 years old, deep baritone, confident and authoritative, measured pace"],
                                ["Male, 17 years old, tenor range, gaining confidence - deeper breath support now"],
                                ["Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice"],
                                ["Elderly woman, 70 years old, soft and caring, speaking slowly with wisdom and warmth"],
                                ["Young child, 8 years old, playful and cheerful, high-pitched with innocent excitement"],
                                ["Professional news anchor, clear articulation, neutral tone, moderate pace"],
                                ["Speak with intense anger, sharp emphasis on words, aggressive tone"],
                                ["Whisper softly with mystery, secretive and intimate, barely audible"],
                                ["Exhausted and sleepy voice, slow drowsy delivery, yawning between words"],
                                ["Speak with genuine surprise and disbelief, voice rising in pitch"],
                                ["Seductive female voice, low and breathy, slow sensual pace"],
                            ],
                            inputs=[vd_description],
                            label=""
                        )

                vd_gen_evt = vd_generate_btn.click(
                    generate_voice_design,
                    inputs=[vd_text, vd_language, vd_description, vd_model_size, vd_max_tokens, vd_temperature, vd_top_p, autoplay_checkbox],
                    outputs=[vd_audio_out, vd_status, live_chunk],
                    js=TTS_ARM_JS,
                )
                vd_pause_btn.click(fn=None, js=TTS_PAUSE_JS)
                vd_stop_btn.click(
                    stop_generation_fn,
                    outputs=[vd_audio_out, vd_status, live_chunk],
                    cancels=[vd_gen_evt],
                )


    demo._portable_theme = theme
    demo._portable_css = css
    demo._portable_head = head_html
    return demo


# =====================================================
# Точка входа
# =====================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"{APP_NAME} v{APP_VERSION}")
    print("=" * 60)
    print()

    # Определяем устройство
    device = get_device()
    print(f"Устройство: {device.upper()}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print()
    print(f"Директория голосов: {VOICES_DIR}")
    print(f"Директория профилей: {PROFILES_DIR}")
    print(f"Директория вывода: {OUTPUT_DIR}")

    local_voices = get_local_voices()
    print(f"Локальных голосов: {len(local_voices)}")
    print(engine_status_line())

    profiles = list_voice_profiles()
    print(f"Сохранённых профилей: {len(profiles)}")

    print()
    print("Запуск веб-интерфейса...")
    print()

    # Строим и запускаем интерфейс
    demo = build_ui()

    from fastapi.responses import FileResponse, Response
    from gradio.routes import App as GradioApp

    _create_app = GradioApp.create_app

    def _create_app_with_live(*args, **kwargs):
        app = _create_app(*args, **kwargs)
        if getattr(app, "_tts_live_ok", False):
            return app
        app._tts_live_ok = True
        live_root = (OUTPUT_DIR / "live").resolve()

        @app.get("/tts_live/{name}")
        async def tts_live(name: str):
            if not name.endswith(".wav") or any(ch in name for ch in ("/", "\\", "..")):
                return Response(status_code=400)
            path = (live_root / name).resolve()
            if path.parent != live_root or not path.is_file():
                return Response(status_code=404)
            return FileResponse(str(path), media_type="audio/wav")

        return app

    GradioApp.create_app = staticmethod(_create_app_with_live)

    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        theme=demo._portable_theme,
        css=demo._portable_css,
        head=demo._portable_head,
        allowed_paths=[str(OUTPUT_DIR.resolve())],
    )
