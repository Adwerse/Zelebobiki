# neura — обучение и инференс модели

Этот раздел содержит тренировочный пайплайн нейросети, которая по входным координатам (lat, lon) реконструирует океанологические признаки и оценивает вероятность появления акулы.

## Что делает модель
- Вход: только координаты `lat_cell`, `lon_cell`.
- Выходы:
  - Реконструируемые признаки: `chlor_a_mean`, `sst_mean`, `chl_norm`, `sst_suit`, `month_sin`, `month_cos`.
  - Классификация: вероятность `shark_probability` (выход `label`), бинарный вывод берётся порогом 0.5.

## Файлы и структура
- `train_shark_model.py` — основной скрипт обучения/оценки/сохранения/демо-инференса.
- `data/` — CSV‑файлы для обучения (может быть несколько, скрипт соберёт все по маске).
- `model_out/` — артефакты обучения:
  - `final_model.keras` — Keras‑модель (инференс)
  - `best_model.h5` — лучшая по `val_loss` (чекпоинт)
  - `input_scaler.joblib` — StandardScaler для входных координат
  - `output_scaler.joblib` — StandardScaler для реконструируемых таргетов
  - `training_history.csv`/`png` — лог обучения
  - `logs/` — TensorBoard

## Требования к данным
Обязательные колонки (в CSV):
`label, month_idx, lat_cell, lon_cell, chlor_a_mean, sst_mean, chl_norm, sst_suit, month_sin, month_cos`

Примечания:
- Если `label` отсутствует — скрипт попытается сформировать его из `shark_count`/`count`/`total`, иначе сгенерирует вероятностно из `sst_mean` и `chlor_a_mean`.
- Если `month_sin`/`month_cos` отсутствуют — будут рассчитаны из `month_idx`.
- Пропуски в числовых столбцах заполняются медианой.

## Быстрый старт обучения
```powershell
cd C:\Zelebobiki\neura
python train_shark_model.py
```
- Если в `neura/data` нет CSV — скрипт сгенерирует синтетический датасет (2000 строк).
- Артефакты сохранятся в `neura/model_out`.

## Ключевые настройки (вверху `train_shark_model.py`)
- `TEST_SIZE = 0.2`, `RANDOM_SEED = 42`, `BATCH_SIZE = 32`, `EPOCHS = 50`
- `LOSS_WEIGHTS = {"reconstruct": 1.0, "label": 1.0}` для баланса задач
- `USE_RECONSTRUCT_FEATURES_IN_CLASSIFIER = True` — конкатенировать реконструкции с shared‑фичами в классификаторе
- `TRAIN_WITH_FULL_FEATURES = False` — при True классификатор принимает реальные признаки (только для экспериментов; в проде остаёмся на lon/lat)

## Инференс в бэкенде
Бэкенд (DRF) ожидает артефакты:
- `final_model.keras`, `input_scaler.joblib`, `output_scaler.joblib`
и по запросу `?lat=<..>&lon=<..>` возвращает:
```json
{
  "chlor_a_mean": <float>,
  "sst_mean": <float>,
  "chl_norm": <float>,
  "sst_suit": <float>,
  "month_sin": <float>,
  "month_cos": <float>,
  "shark_probability": <0..1>,
  "shark_present": true|false
}
```

## Метрики качества (пример)
На демонстрационном наборе:
- Классификация: Accuracy ~0.888, ROC‑AUC ~0.949
- Реконструкция: `sst_mean` RMSE ~2.45°C (Corr ~0.96), `sst_suit` RMSE ~0.17 (Corr ~0.91)

## Типичные проблемы и решения
- Предупреждения `protobuf`/`oneDNN` — информативны, не мешают запуску.
- Если бэкенд не находит артефакты — проверьте пути `neura/model_out` или переместите артефакты в корень `model_out`.
- Если `label` сильно несбалансирован — увеличьте `LOSS_WEIGHTS['label']` или используйте class weights.

## Лицензия
Смотри корневой `LICENSE`.
