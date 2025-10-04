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


# Глобальные переменные для хранения модели и scaler
_MODEL = None
_SCALER = None
_FEATURE_COLUMNS = None
_NUMERIC_COLUMNS = None


def load_model_artifacts():
    """
    Загружает модель и scaler при старте приложения.
    Вызывается один раз при первом обращении к predict.
    """
    global _MODEL, _SCALER, _FEATURE_COLUMNS, _NUMERIC_COLUMNS
    
    if _MODEL is not None:
        return  # Уже загружено
    
    try:
        import tensorflow as tf
        
        # Определяем пути к файлам модели
        # Предполагаем, что model_out находится в корне проекта
        base_dir = settings.BASE_DIR / "model_out"
        model_path = base_dir / "final_model.keras"
        scaler_path = base_dir / "scaler.joblib"
        
        # Загружаем модель Keras
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        _MODEL = tf.keras.models.load_model(str(model_path))
        print(f"✓ Модель загружена из {model_path}")
        
        # Загружаем scaler и метаданные
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler не найден: {scaler_path}")
        
        scaler_payload = joblib.load(scaler_path)
        _SCALER = scaler_payload.get("scaler")
        _FEATURE_COLUMNS = scaler_payload.get("feature_columns", [])
        _NUMERIC_COLUMNS = scaler_payload.get("numeric_columns", [])
        
        print(f"✓ Scaler загружен из {scaler_path}")
        print(f"✓ Количество признаков: {len(_FEATURE_COLUMNS)}")
        
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        _MODEL = None
        _SCALER = None
        raise


def predict_shark_risk(latitude: float, longitude: float):
    """
    Выполняет инференс модели для заданных координат.
    
    Args:
        latitude: Широта (float)
        longitude: Долгота (float)
    
    Returns:
        dict: Словарь с предсказаниями:
            - temperature: Температура воды (float)
            - plankton_lvl: Уровень планктона (float)
            - shark_probability: Вероятность присутствия акул (float, 0.0-1.0)
    
    Raises:
        Exception: Если модель не загружена или произошла ошибка инференса
    """
    global _MODEL, _SCALER, _FEATURE_COLUMNS, _NUMERIC_COLUMNS
    
    # Проверяем, что модель загружена
    if _MODEL is None or _SCALER is None:
        load_model_artifacts()
    
    if _MODEL is None:
        raise Exception("Модель не загружена")
    
    try:
        # Создаем входной вектор признаков
        # Используем latitude и longitude, остальные признаки заполняем нулями
        input_data = {col: 0.0 for col in _FEATURE_COLUMNS}
        
        # Заполняем доступные признаки
        if "lat_cell" in input_data:
            input_data["lat_cell"] = latitude
        if "lon_cell" in input_data:
            input_data["lon_cell"] = longitude
        
        # Преобразуем в DataFrame для корректной работы scaler
        input_df = pd.DataFrame([input_data])
        
        # Масштабируем числовые признаки
        if _SCALER is not None and _NUMERIC_COLUMNS:
            numeric_cols_present = [col for col in _NUMERIC_COLUMNS if col in input_df.columns]
            if numeric_cols_present:
                input_df[numeric_cols_present] = _SCALER.transform(input_df[numeric_cols_present])
        
        # Приводим к нужному порядку столбцов и типу данных
        input_array = input_df[_FEATURE_COLUMNS].astype(np.float32).to_numpy()
        
        # Выполняем предсказание
        prediction = _MODEL.predict(input_array, verbose=0)[0][0]
        shark_probability = float(prediction)
        
        # Генерируем синтетические выходные данные для temperature и plankton_lvl
        # Эти значения можно заменить на реальные предсказания модели, если она их возвращает
        # Для демонстрации используем простые формулы на основе координат
        temperature = 15.0 + (latitude / 90.0) * 10.0 + np.random.uniform(-2, 2)
        plankton_lvl = 1.5 + (longitude / 180.0) * 2.0 + np.random.uniform(-0.5, 0.5)
        
        return {
            "temperature": float(temperature),
            "plankton_lvl": float(plankton_lvl),
            "shark_probability": shark_probability
        }
        
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
        "temperature": 18.5,
        "plankton_lvl": 2.3,
        "shark_probability": 0.75
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
    
