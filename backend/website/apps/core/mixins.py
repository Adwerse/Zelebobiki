from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema, OpenApiParameter

from apps.core.v1.serializers import OceanGeoPointSerializer


class OceanGeoPointAPIViewMixin:
    @extend_schema(
        parameters=[
            OpenApiParameter(name="lon", type=OpenApiTypes.FLOAT, description="Longitude"),
            OpenApiParameter(name="lat", type=OpenApiTypes.FLOAT, description="Latitude"),
        ]
    )
    def get(self, request, *args, **kwargs):
        lat = request.GET.get("lat")
        lon = request.GET.get("lon")

        serializer = OceanGeoPointSerializer(data={
            "lon": lon,
            "lat": lat,
        })
        serializer.is_valid(raise_exception=True)

        return super().get(request, *args, **kwargs)
