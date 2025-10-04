"""Тренировочный скрипт для модели реконструкции океанологических признаков и вероятности появления акулы.

Скрипт устойчив к отсутствию исходных CSV: при их отсутствии генерирует синтетические данные.
Инференс модели ожидает только координаты (lat, lon) и восстанавливает остальные признаки.
"""

import os
from pathlib import Path

# --- Проверка зависимостей -------------------------------------------------
try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - критическая ошибка среды
    print("Не удалось импортировать tensorflow. Установите зависимости командой:\n"
          "    pip install tensorflow pandas scikit-learn joblib matplotlib")
    raise SystemExit(1) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    print("Не удалось импортировать numpy. Установите зависимости командой:\n"
          "    pip install tensorflow pandas scikit-learn joblib matplotlib")
    raise SystemExit(1) from exc

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    print("Не удалось импортировать pandas. Установите зависимости командой:\n"
          "    pip install tensorflow pandas scikit-learn joblib matplotlib")
    raise SystemExit(1) from exc

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError as exc:  # pragma: no cover
    print("Не удалось импортировать scikit-learn. Установите зависимости командой:\n"
          "    pip install tensorflow pandas scikit-learn joblib matplotlib")
    raise SystemExit(1) from exc

try:
    import joblib
except ImportError as exc:  # pragma: no cover
    print("Не удалось импортировать joblib. Установите зависимости командой:\n"
          "    pip install tensorflow pandas scikit-learn joblib matplotlib")
    raise SystemExit(1) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    print("Не удалось импортировать matplotlib. Установите зависимости командой:\n"
          "    pip install tensorflow pandas scikit-learn joblib matplotlib")
    raise SystemExit(1) from exc

# --- Глобальные константы --------------------------------------------------
DATA_DIR = Path("./data")
CSV_GLOBS = ["**/*.csv"]
REQUIRED_COLUMNS = [
    "label",
    "month_idx",
    "lat_cell",
    "lon_cell",
    "chlor_a_mean",
    "sst_mean",
    "chl_norm",
    "sst_suit",
    "month_sin",
    "month_cos",
]
INPUT_COORD_COLUMNS = ["lat_cell", "lon_cell"]
RECONSTRUCT_TARGETS = [
    "chlor_a_mean",
    "sst_mean",
    "chl_norm",
    "sst_suit",
    "month_sin",
    "month_cos",
]
LABEL_COLUMN = "label"
TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
MODEL_OUTDIR = Path("./model_out")
LOSS_WEIGHTS = {"reconstruct": 1.0, "label": 1.0}
USE_RECONSTRUCT_FEATURES_IN_CLASSIFIER = True
TRAIN_WITH_FULL_FEATURES = False  # Если True, классификатор получает реальные признаки при обучении
TENSORBOARD_LOG_DIR = MODEL_OUTDIR / "logs"

# --- Функции работы с данными ---------------------------------------------

def ensure_output_dir() -> None:
    """Создаёт директорию для артефактов, если не существует."""
    MODEL_OUTDIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Численно стабильный сигмоид, используется при генерации синтетических меток."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_dataframe(n_rows: int = 2000) -> pd.DataFrame:
    """Генерирует синтетический датасет с реалистичными распределениями."""
    rng = np.random.default_rng(RANDOM_SEED)

    month_idx = rng.integers(1, 13, size=n_rows)
    lat_cell = rng.uniform(-60.0, 60.0, size=n_rows)
    lon_cell = rng.uniform(-180.0, 180.0, size=n_rows)
    chlor_a_mean = np.exp(rng.normal(np.log(1.5), 0.4, size=n_rows))
    sst_mean = rng.normal(18.0, 4.0, size=n_rows)
    chl_norm = (chlor_a_mean - np.mean(chlor_a_mean)) / (np.std(chlor_a_mean) + 1e-6)
    sst_suit = np.clip(sigmoid(0.6 * (sst_mean - 16.0)) + rng.normal(0.0, 0.05, size=n_rows), 0.0, 1.0)

    radians = 2 * np.pi * (month_idx - 1) / 12.0
    month_sin = np.sin(radians)
    month_cos = np.cos(radians)

    logits = 0.18 * (sst_mean - 17.5) + 0.45 * (chlor_a_mean - chlor_a_mean.mean())
    base_prob = sigmoid(logits)
    label = (base_prob > rng.random(n_rows)).astype(int)

    df = pd.DataFrame({
        "label": label,
        "month_idx": month_idx,
        "lat_cell": lat_cell,
        "lon_cell": lon_cell,
        "chlor_a_mean": chlor_a_mean,
        "sst_mean": sst_mean,
        "chl_norm": chl_norm,
        "sst_suit": sst_suit,
        "month_sin": month_sin,
        "month_cos": month_cos,
    })

    print("Данные CSV не найдены. Сгенерирован синтетический датасет на 2000 строк.")
    return df


def find_csv_files() -> list:
    """Находит все CSV-файлы в DATA_DIR согласно шаблонам."""
    files = []
    if not DATA_DIR.exists():
        return files

    for pattern in CSV_GLOBS:
        files.extend(DATA_DIR.glob(pattern))
    return [f for f in files if f.is_file()]


def read_csv_safely(file_path: Path) -> pd.DataFrame:
    """Читает CSV с обработкой ошибок."""
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Загружен файл {file_path} (строк: {len(df)})")
        return df
    except Exception as exc:  # pragma: no cover - зависит от данных
        print(f"✗ Ошибка чтения {file_path}: {exc}")
        return pd.DataFrame()


def combine_datasets(frames: list) -> pd.DataFrame:
    """Объединяет список DataFrame в один."""
    if not frames:
        return generate_synthetic_dataframe()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)
    return combined


def create_label_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантирует наличие столбца LABEL_COLUMN."""
    if LABEL_COLUMN in df.columns:
        return df

    candidate_cols = [col for col in df.columns if col.lower() in {"shark_count", "count", "total"}]
    for col in candidate_cols:
        try:
            df[LABEL_COLUMN] = (df[col].astype(float) > 0).astype(int)
            print(f"✓ Сформирован столбец label из {col}.")
            return df
        except Exception as exc:  # pragma: no cover - зависит от данных
            print(f"✗ Не удалось преобразовать {col} в label: {exc}")

    rng = np.random.default_rng(RANDOM_SEED)
    if "sst_mean" in df.columns and "chlor_a_mean" in df.columns:
        logits = 0.18 * (df["sst_mean"].astype(float) - df["sst_mean"].mean())
        logits += 0.35 * (df["chlor_a_mean"].astype(float) - df["chlor_a_mean"].mean())
        probs = sigmoid(logits.to_numpy())
        df[LABEL_COLUMN] = (probs > rng.random(len(df))).astype(int)
        print("✓ Сформирован вероятностный label из признаков sst_mean и chlor_a_mean.")
    else:
        df[LABEL_COLUMN] = rng.integers(0, 2, size=len(df))
        print("⚠️ Признаки для логической метки не найдены, создан случайный label.")
    return df


def ensure_month_trigonometry(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантирует наличие синус/косинус месяцей на основе month_idx."""
    if "month_idx" in df.columns:
        radians = 2 * np.pi * (df["month_idx"].astype(float) - 1) / 12.0
        if "month_sin" not in df.columns:
            df["month_sin"] = np.sin(radians)
            print("✓ Вычислен month_sin из month_idx.")
        if "month_cos" not in df.columns:
            df["month_cos"] = np.cos(radians)
            print("✓ Вычислен month_cos из month_idx.")
    return df


def filter_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только необходимые столбцы, добавляя отсутствующие при необходимости."""
    df = ensure_month_trigonometry(df)
    df = create_label_if_missing(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"⚠️ Обнаружены отсутствующие столбцы: {missing}. Будут заполнены NaN.")
        for col in missing:
            df[col] = np.nan

    filtered = df[INPUT_COORD_COLUMNS + RECONSTRUCT_TARGETS + [LABEL_COLUMN]].copy()
    return filtered


def summarize_dataset(df: pd.DataFrame) -> None:
    """Печатает краткую сводку по датасету."""
    total_nan = int(df.isna().sum().sum())
    label_counts = df[LABEL_COLUMN].value_counts(dropna=False)

    print("\n===== Сводка данных =====")
    print(f"Строк: {len(df)}, столбцов: {df.shape[1]}")
    print(f"Всего пропусков: {total_nan}")
    print("Распределение классов:")
    print(label_counts.to_string())
    print("========================\n")


def preprocess_dataframe(df: pd.DataFrame):
    """Выполняет заполнение пропусков и масштабирование."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        df[col] = df[col].fillna("missing")

    for col in ["month_sin", "month_cos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df


# --- Подготовка обучающих выборок -----------------------------------------

def prepare_datasets(df: pd.DataFrame):
    """Формирует тренировочную и валидационную выборки."""
    features = df[INPUT_COORD_COLUMNS].to_numpy(dtype=np.float32)
    targets_reconstruct = df[RECONSTRUCT_TARGETS].to_numpy(dtype=np.float32)
    labels = df[LABEL_COLUMN].astype(int).to_numpy()

    stratify = labels if len(np.unique(labels)) >= 2 else None

    X_train, X_val, recon_train, recon_val, y_train, y_val = train_test_split(
        features,
        targets_reconstruct,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    print("Размеры выборок:")
    print(f"  Train: {X_train.shape[0]} образцов")
    print(f"  Val  : {X_val.shape[0]} образцов")
    if stratify is not None:
        unique, counts = np.unique(y_train, return_counts=True)
        print("Распределение классов в train:")
        for cls, cnt in zip(unique, counts):
            print(f"  Класс {cls}: {cnt}")
    print()

    input_scaler = StandardScaler()
    input_scaler.fit(X_train)
    X_train_scaled = input_scaler.transform(X_train)
    X_val_scaled = input_scaler.transform(X_val)

    output_scaler = StandardScaler()
    output_scaler.fit(recon_train)
    recon_train_scaled = output_scaler.transform(recon_train)
    recon_val_scaled = output_scaler.transform(recon_val)

    ensure_output_dir()
    joblib.dump(input_scaler, MODEL_OUTDIR / "input_scaler.joblib")
    joblib.dump(output_scaler, MODEL_OUTDIR / "output_scaler.joblib")
    print("✓ Сохранены scaler'ы в MODEL_OUTDIR.")

    # Разделяем реконструкции по таргетам
    y_train_dict = {
        name: recon_train_scaled[:, idx][:, np.newaxis]
        for idx, name in enumerate(RECONSTRUCT_TARGETS)
    }
    y_val_dict = {
        name: recon_val_scaled[:, idx][:, np.newaxis]
        for idx, name in enumerate(RECONSTRUCT_TARGETS)
    }
    y_train_dict[LABEL_COLUMN] = y_train.astype(np.float32)[:, np.newaxis]
    y_val_dict[LABEL_COLUMN] = y_val.astype(np.float32)[:, np.newaxis]

    if TRAIN_WITH_FULL_FEATURES:
        train_inputs = [X_train_scaled, recon_train_scaled]
        val_inputs = [X_val_scaled, recon_val_scaled]
    else:
        train_inputs = X_train_scaled
        val_inputs = X_val_scaled

    aux_data = {
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "recon_train_raw": recon_train,
        "recon_val_raw": recon_val,
        "y_train": y_train,
        "y_val": y_val,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "train_inputs": train_inputs,
        "val_inputs": val_inputs,
        "recon_train_scaled": recon_train_scaled,
        "recon_val_scaled": recon_val_scaled,
    }

    return train_inputs, val_inputs, y_train_dict, y_val_dict, aux_data


# --- Архитектура модели ----------------------------------------------------

def build_model():
    """Создаёт и компилирует многоцелевую модель."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    inputs_coords = tf.keras.layers.Input(shape=(len(INPUT_COORD_COLUMNS),), name="coords_input")
    additional_inputs = []

    shared = tf.keras.layers.Dense(128, activation="relu", name="shared_dense_1")(inputs_coords)
    shared = tf.keras.layers.Dropout(0.2, name="shared_dropout_1")(shared)
    shared = tf.keras.layers.Dense(64, activation="relu", name="shared_dense_2")(shared)
    shared = tf.keras.layers.Dense(32, activation="relu", name="shared_dense_3")(shared)

    x_recon = tf.keras.layers.Dense(64, activation="relu", name="recon_dense")(shared)

    recon_outputs = {}
    for target in RECONSTRUCT_TARGETS:
        recon_outputs[target] = tf.keras.layers.Dense(1, name=target)(x_recon)

    classifier_inputs_list = [shared]
    if USE_RECONSTRUCT_FEATURES_IN_CLASSIFIER:
        classifier_inputs_list.append(x_recon)

    real_features_input = None
    if TRAIN_WITH_FULL_FEATURES:
        real_features_input = tf.keras.layers.Input(shape=(len(RECONSTRUCT_TARGETS),), name="real_features_input")
        classifier_inputs_list.append(real_features_input)
        additional_inputs.append(real_features_input)

    if len(classifier_inputs_list) > 1:
        classifier_input = tf.keras.layers.Concatenate(name="classifier_concat")(classifier_inputs_list)
    else:
        classifier_input = classifier_inputs_list[0]

    classifier_branch = tf.keras.layers.Dense(32, activation="relu", name="cls_dense_1")(classifier_input)
    classifier_branch = tf.keras.layers.Dropout(0.2, name="cls_dropout")(classifier_branch)
    classifier_branch = tf.keras.layers.Dense(16, activation="relu", name="cls_dense_2")(classifier_branch)
    label_output = tf.keras.layers.Dense(1, activation="sigmoid", name=LABEL_COLUMN)(classifier_branch)

    outputs = {**recon_outputs, LABEL_COLUMN: label_output}

    model_inputs = [inputs_coords] + additional_inputs
    model = tf.keras.Model(inputs=model_inputs, outputs=outputs, name="shark_multi_task_model")

    losses = {target: "mse" for target in RECONSTRUCT_TARGETS}
    losses[LABEL_COLUMN] = "binary_crossentropy"

    num_recon = max(len(RECONSTRUCT_TARGETS), 1)
    loss_weights = {target: LOSS_WEIGHTS["reconstruct"] / num_recon for target in RECONSTRUCT_TARGETS}
    loss_weights[LABEL_COLUMN] = LOSS_WEIGHTS["label"]

    metrics = {LABEL_COLUMN: ["accuracy", tf.keras.metrics.AUC(name="auc")]}  # реконструкции без метрик

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    model.summary(print_fn=lambda x: print(x))
    return model


# --- Обучение --------------------------------------------------------------

def train_model(model, train_inputs, y_train_dict, val_inputs, y_val_dict):
    """Запускает процесс обучения модели."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_OUTDIR / "best_model.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    if TENSORBOARD_LOG_DIR.exists():
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(TENSORBOARD_LOG_DIR),
                histogram_freq=0,
                write_graph=True,
            )
        )

    history = model.fit(
        train_inputs,
        y_train_dict,
        validation_data=(val_inputs, y_val_dict),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        callbacks=callbacks,
    )
    return history


# --- Оценка ----------------------------------------------------------------

def evaluate_model(model, val_inputs, y_val_dict, aux_data):
    """Выполняет оценку модели на валидации."""
    print("\n===== Оценка модели (val) =====")
    results = model.evaluate(val_inputs, y_val_dict, verbose=0, return_dict=True)
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    y_val_true = aux_data["y_val"]
    y_val_pred_prob = model.predict(val_inputs, verbose=0)[LABEL_COLUMN].ravel()
    y_val_pred_bin = (y_val_pred_prob >= 0.5).astype(int)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val_true, y_val_pred_bin))
    print("\nClassification report:")
    print(classification_report(y_val_true, y_val_pred_bin, digits=3))

    output_scaler = aux_data["output_scaler"]
    recon_val_raw = aux_data["recon_val_raw"]
    predictions = model.predict(val_inputs, verbose=0)

    recon_preds_scaled = np.hstack([
        predictions[target] for target in RECONSTRUCT_TARGETS
    ])
    recon_preds = output_scaler.inverse_transform(recon_preds_scaled)

    print("\nМетрики реконструкции (RMSE / MAE):")
    for idx, target in enumerate(RECONSTRUCT_TARGETS):
        diffs = recon_preds[:, idx] - recon_val_raw[:, idx]
        rmse = np.sqrt(np.mean(diffs ** 2))
        mae = np.mean(np.abs(diffs))
        print(f"  {target}: RMSE={rmse:.4f}, MAE={mae:.4f}")

    print("===============================\n")


# --- Сохранение ------------------------------------------------------------

def save_model_and_history(model, history):
    """Сохраняет модель и график обучения."""
    model.save(MODEL_OUTDIR / "final_model.keras")
    print(f"✓ Модель сохранена в {MODEL_OUTDIR / 'final_model.keras'}")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(MODEL_OUTDIR / "training_history.csv", index=False)

    plt.figure(figsize=(10, 4))
    for column in history_df.columns:
        plt.plot(history_df[column], label=column)
    plt.title("Кривые обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("Значение")
    plt.legend(loc="upper right")
    plt.tight_layout()
    history_plot_path = MODEL_OUTDIR / "training_history.png"
    plt.savefig(history_plot_path)
    plt.close()
    print(f"✓ История обучения сохранена ({history_plot_path}).")


# --- Демонстрация инференса -----------------------------------------------

def demonstrate_inference(model, aux_data, sample_index: int = 0) -> None:
    """Показывает пример инференса модели для координат."""
    input_scaler = aux_data["input_scaler"]
    output_scaler = aux_data["output_scaler"]
    X_val_raw = aux_data["X_val_raw"]

    if X_val_raw.shape[0] == 0:
        print("⚠️ Недостаточно данных для демонстрации инференса.")
        return

    sample_index = int(np.clip(sample_index, 0, X_val_raw.shape[0] - 1))
    coords = X_val_raw[sample_index]
    coords_scaled = input_scaler.transform(coords.reshape(1, -1))

    if TRAIN_WITH_FULL_FEATURES:
        recon_scaled = aux_data["recon_val_scaled"][sample_index].reshape(1, -1)
        model_inputs = [coords_scaled, recon_scaled]
    else:
        model_inputs = coords_scaled

    prediction = model.predict(model_inputs, verbose=0)

    recon_scaled = np.hstack([prediction[target] for target in RECONSTRUCT_TARGETS])
    recon_values = output_scaler.inverse_transform(recon_scaled)[0]

    shark_prob = float(prediction[LABEL_COLUMN].ravel()[0])
    shark_label = int(shark_prob >= 0.5)

    result = {
        target: float(value)
        for target, value in zip(RECONSTRUCT_TARGETS, recon_values)
    }
    result["shark_probability"] = shark_prob
    result["shark_present"] = shark_label

    print("\n===== Демонстрация инференса =====")
    print(f"Координаты (lat, lon): {coords}")
    print("Предсказанные признаки и вероятность:")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")
    print("JSON-вывод:")
    print(result)
    print("==================================\n")


# --- Основной сценарий -----------------------------------------------------

def main() -> None:
    """Основная точка входа скрипта."""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    csv_files = find_csv_files()
    data_frames = [read_csv_safely(path) for path in csv_files]
    data_frames = [df for df in data_frames if not df.empty]

    if not data_frames:
        df = generate_synthetic_dataframe()
    else:
        df = combine_datasets(data_frames)
        print(f"Общий датасет после объединения: {len(df)} строк.")

    df = filter_required_columns(df)
    df = preprocess_dataframe(df)
    summarize_dataset(df)

    train_inputs, val_inputs, y_train_dict, y_val_dict, aux_data = prepare_datasets(df)

    model = build_model()
    history = train_model(model, train_inputs, y_train_dict, val_inputs, y_val_dict)
    evaluate_model(model, val_inputs, y_val_dict, aux_data)
    save_model_and_history(model, history)
    demonstrate_inference(model, aux_data)

    print("Совет: для повышения качества можно добавить временные срезы и пространственно-временные признаки.")
    if TRAIN_WITH_FULL_FEATURES:
        print("⚠️ Включён режим TRAIN_WITH_FULL_FEATURES: потребуется подать реальные признаки в классификатор.")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # pragma: no cover - общая защита
        print(f"Необработанная ошибка: {err}")
        raise
