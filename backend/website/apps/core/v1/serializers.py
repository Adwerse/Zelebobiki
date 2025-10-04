from django.utils.translation import gettext_lazy as _
from global_land_mask import globe
from rest_framework import serializers
from rest_framework.exceptions import ValidationError


class OceanGeoPointSerializer(serializers.Serializer):
    lat = serializers.FloatField(min_value=-90, max_value=90)
    lon = serializers.FloatField(min_value=-180, max_value=180)

    def validate(self, attrs):
        attrs = super().validate(attrs)

        lon = attrs["lon"]
        lat = attrs["lat"]

        if not globe.is_ocean(lat, lon):
            raise ValidationError({"coordinates": _("Given coordinates is not in ocean")})

        return attrs
