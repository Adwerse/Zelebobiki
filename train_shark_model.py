#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Скрипт обучения модели обнаружения акул на табличных данных."""

# Импорты стандартной библиотеки
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Попытка импортировать внешние зависимости
try:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import callbacks, layers, models
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import joblib
    import matplotlib.pyplot as plt
except ImportError as exc:
    missing_module = getattr(exc, "name", None) or str(exc)
    print(f"Не удалось импортировать модуль: {missing_module}", file=sys.stderr)
    print("Установите зависимости командой:", file=sys.stderr)
    print("pip install tensorflow pandas numpy scikit-learn joblib matplotlib", file=sys.stderr)
    sys.exit(1)

# Константы конфигурации
DATA_DIR = "./Nasa_Ocean"
CSV_GLOBS = ["**/*.csv"]
LABEL_COLUMN = "label"
TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
MODEL_OUTDIR = "./model_out"
HISTORY_PLOT_NAME = "training_history.png"
SCALER_FILENAME = "scaler.joblib"
METADATA_FILENAME = "features_metadata.json"
FINAL_MODEL_FILENAME = "final_model.keras"

# Настройка генераторов случайных чисел для воспроизводимости
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def ensure_output_dir(directory: str) -> None:
    """Создать директорию вывода при необходимости."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def find_csv_files(data_dir: str, patterns: List[str]) -> List[Path]:
    """Найти все CSV-файлы по указанным шаблонам."""
    data_path = Path(data_dir)
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path for path in data_path.glob(pattern) if path.is_file())
    return sorted(files)


def create_synthetic_dataframe(rows: int = 1000) -> pd.DataFrame:
    """Создать синтетический датасет для демонстрации."""
    rng = np.random.default_rng(RANDOM_SEED)
    temperatures = rng.normal(loc=16.0, scale=4.0, size=rows)
    plankton_level = rng.gamma(shape=2.0, scale=1.5, size=rows)
    chlorophyll = rng.normal(loc=1.5, scale=0.3, size=rows)
    salinity = rng.normal(loc=35.0, scale=1.2, size=rows)
    wave_height = rng.normal(loc=1.0, scale=0.4, size=rows)
    count = rng.poisson(lam=2.0, size=rows)
    label = (count > 0).astype(int)
    df = pd.DataFrame(
        {
            "temperature": temperatures,
            "plankton_level": plankton_level,
            "chlorophyll": chlorophyll,
            "salinity": salinity,
            "wave_height": wave_height,
            "count": count,
            LABEL_COLUMN: label,
        }
    )
    return df


def load_dataset(data_dir: str, patterns: List[str]) -> pd.DataFrame:
    """Загрузить все доступные CSV-файлы или создать синтетику."""
    csv_files = find_csv_files(data_dir, patterns)
    if not csv_files:
        print("CSV-файлы не найдены, создаем синтетический датасет")
        return create_synthetic_dataframe()

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)  # Читаем таблицу из файла
            df["__source_file"] = str(csv_path)  # Сохраняем источник для отладки
            frames.append(df)
            print(f"Загружен файл: {csv_path} (строк: {len(df)})")
        except Exception as err:  # noqa: BLE001
            print(f"Ошибка чтения {csv_path}: {err}")
    if not frames:
        print("Не удалось прочитать CSV-файлы, используем синтетический датасет")
        return create_synthetic_dataframe()

    try:
        combined = pd.concat(frames, ignore_index=True, sort=False)  # Объединяем все таблицы
        print(f"Общий датафрейм собран: {combined.shape[0]} строк, {combined.shape[1]} колонок")
        return combined
    except Exception as err:  # noqa: BLE001
        print(f"Ошибка объединения CSV: {err}")
        print("Переходим на синтетический датасет для демонстрации")
        return create_synthetic_dataframe()


def ensure_label_column(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    """Обеспечить наличие бинарной метки и вернуть её вместе с описанием."""
    df = df.copy()
    mapping_info: Dict[str, str] = {}

    if label_column not in df.columns:
        generated = False
        for candidate in ("count", "total"):
            if candidate in df.columns:
                numeric_candidate = pd.to_numeric(df[candidate], errors="coerce").fillna(0)  # Переводим в числа
                df[label_column] = (numeric_candidate > 0).astype(int)  # Создаем бинарную метку
                mapping_info = {"source": candidate, "rule": "> 0 -> 1"}
                generated = True
                print(f"Создан столбец меток на основе '{candidate}'")
                break
        if not generated:
            rng = np.random.default_rng(RANDOM_SEED)
            df[label_column] = rng.integers(0, 2, size=len(df))  # Псевдослучайные метки
            mapping_info = {"source": "random", "rule": "seeded binary"}
            print("Метка не найдена, сгенерированы случайные значения")

    label_series = df[label_column]
    numeric_series = pd.to_numeric(label_series, errors="coerce")  # Пробуем трактовать метку как числа

    if numeric_series.notna().any():
        label_binary = (numeric_series > 0).astype(int)  # Преобразуем к 0/1 через порог
        mapping_info.setdefault("conversion", "numeric > 0 -> 1")
    else:
        lowered = label_series.astype(str).str.lower()
        shark_mask = lowered.str.contains("shark", case=False, na=False)  # Ищем строки с упоминанием акул
        if shark_mask.any():
            label_binary = shark_mask.astype(int)  # Метка 1, если есть "shark"
            mapping_info.setdefault("conversion", "строка содержит 'shark'")
        elif label_series.nunique(dropna=True) == 2:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(label_series.astype(str))  # Кодируем два класса как 0/1
            label_binary = pd.Series(encoded, index=label_series.index)
            mapping_info = {str(cls): str(val) for cls, val in zip(encoder.classes_, encoder.transform(encoder.classes_))}
            mapping_info.setdefault("conversion", "LabelEncoder")
        else:
            rng = np.random.default_rng(RANDOM_SEED)
            label_binary = pd.Series(rng.integers(0, 2, size=len(label_series)), index=label_series.index)
            mapping_info = {"source": "random", "rule": "seeded binary"}
            print("Не удалось создать метку из текста, использованы случайные значения")

    df[label_column] = label_binary.astype(int)
    return df, df[label_column], mapping_info


def preprocess_features(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Подготовить признаки: заполнение пропусков, кодирование категорий."""
    features = df.drop(columns=[label_column], errors="ignore").copy()
    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns.tolist()  # Определяем числовые поля
    categorical_cols = [col for col in features.columns if col not in numeric_cols]  # Остальные считаем категориальными

    if numeric_cols:
        for col in numeric_cols:
            features[col] = pd.to_numeric(features[col], errors="coerce")  # Гарантируем числовой формат
        medians = features[numeric_cols].median()
        features[numeric_cols] = features[numeric_cols].fillna(medians)  # Заполняем медианой
    if categorical_cols:
        for col in categorical_cols:
            features[col] = (
                features[col]
                .astype(str)
                .fillna("missing")
                .replace({"nan": "missing", "None": "missing"})
            )  # Подставляем заглушку для пропусков
        features = pd.get_dummies(features, columns=categorical_cols, dummy_na=False)  # One-hot кодирование

    features = features.fillna(0)  # Дублирующая защита от пропусков
    dummy_cols = [col for col in features.columns if col not in numeric_cols]  # Имена новых категориальных столбцов
    return features, numeric_cols, dummy_cols


def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделить данные на обучение и валидацию со стратификацией при возможности."""
    unique, counts = np.unique(y, return_counts=True)
    stratify_arg = y if (len(unique) >= 2 and counts.min() >= 2) else None  # Проверяем, можно ли стратифицировать
    if stratify_arg is None:
        print("Стратификация отключена из-за недостаточного количества классов")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_arg,
    )
    return X_train, X_val, y_train, y_val


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
    """Масштабировать числовые признаки и вернуть scaler."""
    if not numeric_cols:
        print("Числовые признаки отсутствуют, масштабирование пропущено")
        return X_train.copy(), X_val.copy(), None

    scaler = StandardScaler()  # Инициализируем стандартизатор
    scaler.fit(X_train[numeric_cols])  # Обучаем только на обучающей выборке
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])  # Применяем к обучению
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])  # Применяем к валидации
    return X_train_scaled, X_val_scaled, scaler


def build_model(input_dim: int) -> models.Sequential:
    """Построить простую полносвязную нейросеть."""
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def save_history_plot(history: tf.keras.callbacks.History, outdir: str) -> None:
    """Сохранить график обучения для наглядности."""
    if not history or not history.history:
        return
    ensure_output_dir(outdir)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="train_loss")  # Динамика функции потерь
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.plot(history.history.get("accuracy", []), label="train_acc")  # Динамика точности
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title("Динамика обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plot_path = Path(outdir) / HISTORY_PLOT_NAME
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"График обучения сохранен: {plot_path}")


def main() -> None:
    """Основная точка входа скрипта."""
    ensure_output_dir(MODEL_OUTDIR)  # Подготовка папки для артефактов

    print("Загрузка данных...")
    raw_df = load_dataset(DATA_DIR, CSV_GLOBS)  # Читаем данные или создаем синтетику
    print(f"Исходная форма данных: {raw_df.shape}")

    print("Проверка и подготовка меток...")
    df_with_labels, labels, label_info = ensure_label_column(raw_df, LABEL_COLUMN)  # Гарантируем бинарную метку
    print(f"Информация о метке: {label_info}")

    print("Предобработка признаков...")
    features, numeric_cols, dummy_cols = preprocess_features(df_with_labels, LABEL_COLUMN)  # Готовим матрицу признаков
    print(f"Числовых признаков: {len(numeric_cols)}, категориальных после кодирования: {len(dummy_cols)}")

    print("Разбиение на обучающую и валидационную выборки...")
    X_train, X_val, y_train, y_val = split_data(features, labels)  # Делим данные
    print(f"Размеры X_train: {X_train.shape}, X_val: {X_val.shape}")
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print(f"Распределение классов в обучении: {dict(zip(unique_classes.tolist(), class_counts.tolist()))}")

    print("Масштабирование числовых признаков...")
    X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val, numeric_cols)  # Применяем StandardScaler
    feature_columns = X_train_scaled.columns.tolist()  # Запоминаем порядок признаков

    X_train_array = X_train_scaled.astype(np.float32).to_numpy()  # Приводим к numpy-массивам
    X_val_array = X_val_scaled.astype(np.float32).to_numpy()
    y_train_array = y_train.astype(np.float32).to_numpy()
    y_val_array = y_val.astype(np.float32).to_numpy()

    print("Построение модели...")
    model = build_model(input_dim=X_train_array.shape[1])  # Конструируем нейросеть
    model.summary(print_fn=lambda line: print(line))

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)  # Останавливаем при переобучении
    checkpoint_path = Path(MODEL_OUTDIR) / "best_model.h5"
    model_ckpt = callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_best_only=True,
        monitor="val_loss",
        verbose=1,
    )

    print("Запуск обучения...")
    history = model.fit(
        X_train_array,
        y_train_array,
        validation_data=(X_val_array, y_val_array),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, model_ckpt],
        verbose=1,
    )

    print("Сохранение графика обучения...")
    save_history_plot(history, MODEL_OUTDIR)

    print("Оценка модели на валидации...")
    eval_results = model.evaluate(X_val_array, y_val_array, verbose=0)  # Считаем метрики на валидации
    for name, value in zip(model.metrics_names, eval_results):
        print(f"{name}: {value:.4f}")

    print("Генерация отчетов по качеству...")
    y_val_pred_prob = model.predict(X_val_array, batch_size=BATCH_SIZE)  # Получаем вероятности
    y_val_pred = (y_val_pred_prob >= 0.5).astype(int)  # Переводим в классы
    cm = confusion_matrix(y_val_array, y_val_pred)  # Confusion matrix
    print("Confusion matrix:")
    print(cm)
    report = classification_report(y_val_array, y_val_pred, digits=4)
    print("Classification report:")
    print(report)

    print("Сохранение артефактов модели...")
    ensure_output_dir(MODEL_OUTDIR)  # Убеждаемся, что папка существует
    model_save_path = Path(MODEL_OUTDIR) / FINAL_MODEL_FILENAME
    model.save(model_save_path)  # Сохраняем итоговую модель
    print(f"Модель сохранена: {model_save_path}")

    scaler_payload: Dict[str, object] = {
        "scaler": scaler,
        "numeric_columns": numeric_cols,
        "feature_columns": feature_columns,
        "label_info": label_info,
    }
    scaler_path = Path(MODEL_OUTDIR) / SCALER_FILENAME
    joblib.dump(scaler_payload, scaler_path)  # Сохраняем стандартизатор и метаданные
    print(f"Scaler и метаданные сохранены: {scaler_path}")

    metadata_path = Path(MODEL_OUTDIR) / METADATA_FILENAME
    metadata_payload = {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_cols,
        "categorical_expanded_columns": dummy_cols,
        "label_info": label_info,
    }
    with metadata_path.open("w", encoding="utf-8") as meta_file:
        json.dump(metadata_payload, meta_file, ensure_ascii=False, indent=2)  # Дублируем метаданные в JSON
    print(f"JSON с метаданными сохранен: {metadata_path}")

    print("Демонстрация инференса на одном примере...")
    if len(X_val) == 0:
        print("Валидационная выборка пуста, инференс пропущен")
    else:
        sample_features = X_val.iloc[[0]].copy()  # Берем первую строку валидации
        if scaler is not None and numeric_cols:
            sample_features[numeric_cols] = scaler.transform(sample_features[numeric_cols])  # Масштабируем числовые признаки
        sample_features = sample_features.reindex(columns=feature_columns, fill_value=0)  # Выравниваем порядок столбцов
        sample_array = sample_features.astype(np.float32).to_numpy()
        sample_prob = model.predict(sample_array)[0][0]
        sample_label = int(sample_prob >= 0.5)  # Пороговое решение 0.5
        print(f"Пример вероятности: {sample_prob:.4f}, предсказанный класс: {sample_label}")

    print("Работа завершена успешно")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Возникла ошибка при выполнении скрипта: {exc}", file=sys.stderr)
        sys.exit(1)
