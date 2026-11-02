# VTracker 🎭🖐️

**VTracker** — это легковесный и быстрый трекер мимики и рук для VTube Studio, работающий без iPhone. Использует мощь **MediaPipe** и **OpenCV** для захвата лица и рук через обычную веб-камеру, и передает данные в VTS по WebSocket API.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)](https://developers.google.com/mediapipe)
[![VTube Studio](https://img.shields.io/badge/VTube_Studio-API-orange.svg)](https://denchisoft.com/)

## ✨ Возможности

- **Трекинг лица**: Повороты головы (Pitch/Yaw/Roll), моргание, направление взгляда, открытие рта, улыбка, движение бровей.
- **Трекинг рук**: Позиция кистей, независимое сгибание всех 5 пальцев на каждой руке, повороты.
- **Распознавание жестов**: 12+ жестов (кулак, ладонь, «ОК», «мир», рога и др.) с защитой от дребезга (debounce).
- **Стабильный WebSocket-клиент**: Уникальные requestID и дренаж буфера для предотвращения разрывов соединения с VTS.
- **Автоматическая калибровка**: Нейтральное положение лица и рук сохраняется в `calibration.json`.
- **Утилита `get.py`**: Быстрый дамп всех доступных выражений и параметров вашей текущей модели в VTS.

## ⚙️ Требования

- Python 3.10+
- VTube Studio с включенным API (порт по умолчанию `8001`)
- Веб-камера (желательно 60 FPS)

## 📦 Установка

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/itz-Myu/vtracker.git
   cd vtracker
   ```

2. **Создайте виртуальное окружение и установите зависимости:**
   ```bash
   python -m venv .vtenv
   source .vtenv/bin/activate  # Для Linux/macOS
   # .vtenv\Scripts\activate   # Для Windows
   pip install mediapipe opencv-python websockets numpy
   ```

3. **Скачайте модели MediaPipe:**
   Поместите файлы `face_landmarker.task` и `hand_landmarker.task` в папку `tasks/`.
   (Скачать можно с [официального сайта MediaPipe](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models)).

4. **Получите токен VTube Studio:**
   Запустите скрипт `python secrets/get_token.py` и разрешите плагину доступ в VTS. Токен сохранится в `secrets/token.txt`.

5. **Настройте `config.py`:**
   Укажите IP/порт VTS и настройте алиасы жестов (`GESTURE_ALIAS`) под выражения вашей модели.

## 🚀 Запуск

```bash
./run.sh
# или
python main.py
```

> **Важно:** При первом запуске подождите пару секунд, пока пройдет калибровка. Смотрите прямо в камеру с нейтральным выражением лица.

## 🛠 Решение проблем

- **VTS отключается через минуту:** Убедитесь, что у вас актуальный `vts_client.py` (в нем реализован `asyncio.wait_for(recv())` для очистки буфера).
- **Жесты не срабатывают:** Проверьте, есть ли имена из `GESTURE_ALIAS` в вашей модели. Запустите `python get.py`, чтобы увидеть список доступных выражений.
- **Руки выворачиваются:** Удалите `calibration.json` и откалибруйте руки заново.
- **Модель не реагирует на пальцы/щёки:** Не все Live2D-модели имеют артикуляцию для `CheekPuff` или пальцев рук. Это ограничение рига модели, а не скрипта.

## 🧩 Структура проекта

```text
vtracker/
├── main.py              # Точка входа, event loop
├── face_tracker.py      # MediaPipe Face Landmarker + эвристики
├── hand_tracker.py      # MediaPipe Hand Landmarker + расчет пальцев
├── gestures.py          # Логика распознавания жестов
├── vts_client.py        # Стабильный WebSocket клиент VTS
├── get.py               # Дамп выражений и параметров
├── config.py            # Настройки
├── tasks/               # Модели .task (не коммитятся)
└── secrets/             # Токены (не коммитятся)
```

## 📄 Лицензия

Этот проект распространяется под лицензией GPL-3.0. Подробности в файле `LICENSE`.
