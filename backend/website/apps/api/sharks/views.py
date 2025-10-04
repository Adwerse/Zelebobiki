import pandas as pd
import pydeck as pdk
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response

from apps.api.mixins import VersioningAPIViewMixin
from apps.core.mixins import OceanGeoPointAPIViewMixin


class SharkPredictionAPIView(VersioningAPIViewMixin, OceanGeoPointAPIViewMixin, GenericAPIView):
    pass


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
