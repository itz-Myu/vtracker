# VTracker — трекер лица и рук для VTube Studio

Кроссплатформенный (в первую очередь Linux) мост между веб-камерой и VTube Studio.
Использует **MediaPipe** для трекинга лица и рук, **OpenCV** для захвата видео и
**WebSocket API VTube Studio** для передачи мимики и жестов. iPhone не нужен.

## ✨ Возможности

- **Трекинг лица**: углы поворота головы (pitch/yaw/roll), моргание, направление взгляда, открытие рта, улыбка, брови.
- **Трекинг рук**: позиция кисти, степень раскрытия, сгибание пальцев, повороты.
- **Распознавание жестов**: 12 статических жестов (кулак, открытая ладонь, «мир», «ОК», большой палец и т.д.), а также двухручные комбинации.
- **Интеграция с VTS**: привязка жестов к выражениям (Expressions) через алиасы в `config.py`.
- **Калибровка**: автоматическая калибровка нейтрального положения лица и рук с сохранением в `calibration.json`.
- **Утилита `get.py`**: человекочитаемый дамп доступных выражений и параметров текущей модели VTS.

## ⚙️ Требования

- Python 3.10+
- VTube Studio с включённым API (по умолчанию порт `8001`)
- Веб-камера

## 📦 Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/itz-Myu/vtracker.git
   cd vtracker
   ```

2. Создайте виртуальное окружение и установите зависимости:
   ```bash
   python -m venv .vtenv
   source .vtenv/bin/activate   # Linux/macOS
   pip install mediapipe opencv-python websockets numpy
   ```

3. Скачайте модели MediaPipe в папку `tasks/`:
   - `face_landmarker.task`
   - `hand_landmarker.task`

   Официальные ссылки для скачивания: [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models).

4. Получите токен плагина VTube Studio.
   - Запустите скрипт `python secrets/get_token.py` и следуйте инструкциям (он запросит доступ в VTS).
   - Токен сохранится в `secrets/token.txt`.

5. При необходимости отредактируйте `config.py`:
   - `VTS_IP`, `VTS_PORT` — адрес VTube Studio.
   - `GESTURE_ALIAS` — сопоставление жестов с именами выражений в вашей модели.

## 🚀 Запуск

```bash
./run.sh
# или
python main.py
```

При первом запуске появится окно калибровки. Примите нейтральное положение лица
и рук, затем нажмите пробел для фиксации.

## 🧩 Структура проекта

```text
vtracker/
├── main.py              # точка входа, event loop
├── face_tracker.py      # трекинг лица (MediaPipe Face Landmarker)
├── hand_tracker.py      # трекинг рук (MediaPipe Hand Landmarker)
├── gestures.py          # распознавание жестов
├── vts_client.py        # WebSocket-клиент VTube Studio
├── calibration.py       # калибровка и сохранение смещений
├── config.py            # настройки (IP, порт, алиасы жестов)
├── get.py               # дамп выражений и параметров модели VTS
├── tasks/               # модели MediaPipe (.task) — не коммитятся
├── secrets/             # токен и get_token.py
└── README.md
```

## 🔧 Решение проблем

- **VTS отключается через минуту-две** — убедитесь, что используете актуальную версию `vts_client.py`. В ней добавлен дренаж receive-буфера (`asyncio.wait_for(recv(), timeout=0.05)`), который предотвращает переполнение очереди и разрыв соединения.
- **Жесты не срабатывают** — проверьте, что имена в `GESTURE_ALIAS` совпадают с выражениями в вашей модели. Запустите `python get.py`, чтобы увидеть актуальный список.
- **Модель не реагирует на `CheekPuff` / `TongueOut`** — эти параметры есть в VTS API, но не все Live2D-модели имеют для них артикуляцию.

## 📄 Лицензия

См. файл `LICENSE`.
