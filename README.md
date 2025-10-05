# Zelebobiki — Shark Risk & Ocean Features

Zelebobiki — это веб‑проективная система, которая по входным координатам (широта/долгота) оценивает вероятность встречи с акулами и реконструирует базовые океанологические признаки. Проект состоит из backend (Django + DRF) и frontend (Vite + Vue), а также подпроекта `neura` с тренировочным пайплайном нейросети.

## Что решает
- Оценка риска встречи с акулами по координатам (без необходимости подгружать большой набор внешних признаков на этапе инференса).
- Восстановление полезных океанологических характеристик из координат: chlor_a_mean, sst_mean, chl_norm, sst_suit, month_sin, month_cos.
- Единая API‑точка для интеграции с картой/клиентом.

## Архитектура
- Backend: Django + DRF, эндпоинты в `backend/website/apps/api`.
- Модель: TensorFlow 2.x, артефакты в `neura/model_out` (или `model_out` в корне, если так проще деплоить).
- Frontend: Vite/Vue (см. `frontend`).

Диаграмма потоков (высокоуровнево):

1) Клиент → (lat, lon) → DRF endpoint `/api/v1/sharks/predict/`
2) DRF → загружает `final_model.keras` + `input_scaler.joblib` + `output_scaler.joblib`
3) DRF → масштабирует координаты → инференс → инвертирует реконструируемые признаки
4) DRF → отдаёт JSON: океан признаки + `shark_probability` + `shark_present`

## Быстрый старт
1. Установите зависимости Python проекта backend (см. `backend/requirements.txt`) и Node‑зависимости frontend.
2. Обучите модель или положите артефакты в `neura/model_out` (см. `neura/README.md`).
3. Запустите backend (Django) и frontend (Vite).

Пример запроса к API (PowerShell):

```powershell
$body = @{ lat = 18.0; lon = -152.0 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/sharks/predict/?lat=18&lon=-152" -Method Get
```

Пример ответа:

```json
{
  "chlor_a_mean": 0.03,
  "sst_mean": 26.8,
  "chl_norm": 0.0013,
  "sst_suit": 0.97,
  "month_sin": 0.08,
  "month_cos": -0.14,
  "shark_probability": 0.28,
  "shark_present": true
}
```

## Где лежат артефакты модели
- По умолчанию бэкенд ищет модель и скейлеры по нескольким путям (в порядке приоритета):
  - `neura/model_out`
  - `model_out` в корне
  - `backend/website/model_out`

Нужные файлы:
- `final_model.keras` — сохранённая Keras‑модель (TF2 SavedModel/Keras format)
- `input_scaler.joblib` — StandardScaler для входных координат
- `output_scaler.joblib` — StandardScaler для реконструируемых признаков

## Качество (по публичному датасету в neura/data)
- Классификация: Accuracy ≈ 0.888, ROC‑AUC ≈ 0.949
- Реконструкция: `sst_mean` RMSE ≈ 2.45°C (Corr ≈ 0.96), `sst_suit` RMSE ≈ 0.17 (Corr ≈ 0.91). Сезонные синусы/косинусы восстанавливаются слабее — из одних координат сезон извлекается ограниченно.

## Папки проекта
- `backend/` — Django сервис, REST API.
- `frontend/` — клиентская часть (Vite/Vue).
- `neura/` — обучение модели (см. отдельный README).
- `model_out/` — альтернативное место для артефактов модели (если не используете `neura/model_out`).

## Замечания по продакшену
- Регулярно дообучайте модель по свежим данным и обновляйте артефакты.
- Если точность по конкретным зонам/сезонам критична, добавляйте временные и пространственно‑временные признаки и/или включайте подачу реальных признаков в классификатор (см. флаги в `neura/train_shark_model.py`).

## Лицензия
Смотри `LICENSE`.
