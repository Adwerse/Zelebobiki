import os
import numpy as np
import pandas as pd
import pydeck as pdk
import joblib
from pathlib import Path
from django.conf import settings
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema, OpenApiParameter

from apps.api.mixins import VersioningAPIViewMixin
from apps.core.mixins import OceanGeoPointAPIViewMixin


# Глобальные переменные для хранения модели и scaler'ов
_MODEL = None
_INPUT_SCALER = None
_OUTPUT_SCALER = None
_MODEL_DIR = None

# Константы, соответствующие обучающему скрипту в neura
RECONSTRUCT_TARGETS = [
    "chlor_a_mean",
    "sst_mean",
    "chl_norm",
    "sst_suit",
    "month_sin",
    "month_cos",
]
LABEL_OUTPUT = "label"
CLASS_THRESHOLD = 0.5


def _resolve_model_dir() -> Path:
    """Возвращает путь к директории с артефактами модели."""
    candidates = [
        settings.BASE_DIR / "model_out",
        settings.BASE_DIR.parent / "model_out",
        settings.BASE_DIR.parent.parent / "neura" / "model_out",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Директория с моделью не найдена. Проверены пути:\n - " + joined
    )


def load_model_artifacts():
    """
    Загружает модель и scaler при старте приложения.
    Вызывается один раз при первом обращении к predict.
    """
    global _MODEL, _INPUT_SCALER, _OUTPUT_SCALER, _MODEL_DIR
    
    if _MODEL is not None and _INPUT_SCALER is not None and _OUTPUT_SCALER is not None:
        return  # Уже загружено
    
    try:
        import tensorflow as tf
        
        _MODEL_DIR = _resolve_model_dir()

        model_path = _MODEL_DIR / "final_model.keras"
        input_scaler_path = _MODEL_DIR / "input_scaler.joblib"
        output_scaler_path = _MODEL_DIR / "output_scaler.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not input_scaler_path.exists():
            raise FileNotFoundError(f"Scaler координат не найден: {input_scaler_path}")
        if not output_scaler_path.exists():
            raise FileNotFoundError(f"Scaler реконструкции не найден: {output_scaler_path}")

        _MODEL = tf.keras.models.load_model(str(model_path))
        _INPUT_SCALER = joblib.load(input_scaler_path)
        _OUTPUT_SCALER = joblib.load(output_scaler_path)

        print(f"✓ Модель загружена из {model_path}")
        print(f"✓ Scaler координат загружен из {input_scaler_path}")
        print(f"✓ Scaler реконструируемых признаков загружен из {output_scaler_path}")
        
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        _MODEL = None
        _INPUT_SCALER = None
        _OUTPUT_SCALER = None
        raise


def predict_shark_risk(latitude: float, longitude: float):
    """
    Выполняет инференс модели для заданных координат.
    
    Args:
        latitude: Широта (float)
        longitude: Долгота (float)
    
    Returns:
        dict: Словарь с предсказаниями:
            - chlor_a_mean, sst_mean, chl_norm, sst_suit, month_sin, month_cos
            - shark_probability: Вероятность присутствия акул (float, 0.0-1.0)
            - shark_present: Бинарный вывод (bool)
    
    Raises:
        Exception: Если модель не загружена или произошла ошибка инференса
    """
    global _MODEL, _INPUT_SCALER, _OUTPUT_SCALER
    
    # Проверяем, что модель загружена
    if _MODEL is None or _INPUT_SCALER is None or _OUTPUT_SCALER is None:
        load_model_artifacts()
    
    if _MODEL is None:
        raise Exception("Модель не загружена")
    
    try:
        coords = np.array([[latitude, longitude]], dtype=np.float32)
        coords_scaled = _INPUT_SCALER.transform(coords)

        raw_predictions = _MODEL.predict(coords_scaled, verbose=0)

        if isinstance(raw_predictions, dict):
            label_probs = raw_predictions[LABEL_OUTPUT].ravel()
            recon_scaled = np.column_stack(
                [raw_predictions[target].ravel() for target in RECONSTRUCT_TARGETS]
            )
        else:
            # Если модель вернула список, предполагаем порядок как в RECONSTRUCT_TARGETS + label
            label_probs = raw_predictions[-1].ravel()
            recon_scaled = np.column_stack(
                [pred.ravel() for pred in raw_predictions[:-1]]
            )

        reconstructed = _OUTPUT_SCALER.inverse_transform(recon_scaled)[0]
        shark_probability = float(label_probs[0])
        shark_present = bool(shark_probability >= CLASS_THRESHOLD)

        response_payload = {
            target: float(value)
            for target, value in zip(RECONSTRUCT_TARGETS, reconstructed)
        }
        response_payload["shark_probability"] = shark_probability
        response_payload["shark_present"] = shark_present

        return response_payload
        
    except Exception as e:
        print(f"✗ Ошибка инференса: {e}")
        raise


class SharkPredictionAPIView(OceanGeoPointAPIViewMixin, VersioningAPIViewMixin, GenericAPIView):
    """
    Эндпоинт для предсказания вероятности встречи с акулами.
    
    POST /api/v1/sharks/predict/
    
    Пример запроса:
    {
        "latitude": 55.75,
        "longitude": 37.61
    }
    
    Пример ответа:
    {
        "chlor_a_mean": 0.32,
        "sst_mean": 24.1,
        "chl_norm": 0.01,
        "sst_suit": 0.82,
        "month_sin": 0.50,
        "month_cos": 0.87,
        "shark_probability": 0.75,
        "shark_present": 1
    }
    
    Пример curl-запроса (Linux/Mac):
    ```bash
    curl -X POST http://localhost:8000/api/v1/sharks/predict/ \
      -H "Content-Type: application/json" \
      -d '{"latitude": 55.75, "longitude": 37.61}'
    ```
    
    Пример для Windows PowerShell:
    ```powershell
    $body = @{
        latitude = 55.75
        longitude = 37.61
    } | ConvertTo-Json
    
    Invoke-RestMethod -Uri "http://localhost:8000/api/v1/sharks/predict/" `
      -Method Post `
      -Body $body `
      -ContentType "application/json"
    ```
    """
    @extend_schema(
        parameters=[
            OpenApiParameter(name="lon", type=OpenApiTypes.FLOAT, description="Longitude"),
            OpenApiParameter(name="lat", type=OpenApiTypes.FLOAT, description="Latitude"),
        ]
    )
    def get(self, request, *args, **kwargs):
        """Обработчик POST-запроса для предсказания."""

        latitude = request.GET.get("lat")
        longitude = request.GET.get("lon")

        data = {
            "lon": request.GET.get("lon"),
            "lat": request.GET.get("lat"),
        }

        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)

        latitude = serializer.validated_data["lat"]
        longitude = serializer.validated_data["lon"]

        try:
            # Выполняем предсказание
            result = predict_shark_risk(latitude, longitude)
            
            return Response(result, status=status.HTTP_200_OK)
            
        except FileNotFoundError as e:
            return Response(
                {"error": f"Файлы модели не найдены: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        # except Exception as e:
        #     return Response(
        #         {"error": f"Ошибка при выполнении предсказания: {str(e)}"},
        #         status=status.HTTP_500_INTERNAL_SERVER_ERROR
        #     )


class SharkHeatmapAPIView(VersioningAPIViewMixin, OceanGeoPointAPIViewMixin, GenericAPIView):
    def get(self, request, *args, **kwargs):
        super().get(request, *args, **kwargs)
        data = pd.DataFrame({
            'lat': [55.75, 55.76, 55.77, 55.78, 55.79],
            'lon': [37.61, 37.62, 37.63, 37.64, 37.65],
            'weight': [1, 5, 10, 3, 7]
        })

        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data,
        )

        data = heatmap_layer.data
        return Response(data)
    
