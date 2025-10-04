from django.urls import path

from apps.api.sharks.views import SharkPredictionAPIView, SharkHeatmapAPIView

app_name = "sharks"

urlpatterns = [
    path("predict/", SharkPredictionAPIView.as_view(), name="predict"),
    path("heatmap/", SharkHeatmapAPIView.as_view(), name="heatmap"),
]
