# Qwen3-TTS Portable (fork)

Портативный Gradio-интерфейс для [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) под Windows: клонирование голоса, пресеты, multi-speaker и дизайн голоса. Этот репозиторий — форк с живым воспроизведением длинных текстов.

## Источники

- Модель и официальный код: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- Портативная русская оболочка: [timoncool/Qwen3-TTS_portable_rus](https://github.com/timoncool/Qwen3-TTS_portable_rus)
- Коллекция моделей: [Hugging Face — Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts)

## Что изменено в этом форке

- **CUDA Graphs** (`faster-qwen3-tts`) — быстрее официального generate на NVIDIA, с откатом на `qwen_tts`, если графы недоступны.
- **Живое воспроизведение** — чанки сразу идут в Web Audio, без пересоздания плеера на каждом куске.
- **Длинный текст** — нарезка по предложениям, очередь GPU → браузер, преролл, чтобы не заикаться на старте.
- **Пауза и стоп** — пауза не убивает генерацию, стоп гасит и GPU, и звук.
- **Русский текст** — даты, числа, URL, email и `@ники` читаются словами, а не ломают 1.7B.
- **Чанки без обрыва и без длинной дыры** — модель не бросает фразу на EOS и не тянет тишину до следующего куска.

## Запуск

1. Распакуйте архив.
2. `portable/install.bat` — зависимости (один раз).
3. `portable/run.bat` — интерфейс.

Модели качаются при первом запуске в `portable/models/`. Сами веса и встроенный Python в git не входят.

## Требования

- Windows 10/11
- NVIDIA GPU, CUDA, лучше от 8 GB VRAM
- 16 GB RAM
- интернет только на первую загрузку моделей

На RTX 3060 12 GB комфортнее **0.6B**. **1.7B** ближе к realtime: живой буфер больше, чанки короче.

## Лицензия

Код оболочки — [Apache 2.0](LICENSE). Веса Qwen3-TTS — [Qwen License](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT).
