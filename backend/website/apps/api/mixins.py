from django.core.exceptions import ImproperlyConfigured


class VersioningAPIViewMixin:
    """
    Versioning mixin class for api views.

    version_map = {
        'v1': {
            'serializer_class': Serializer,
        },
        'v2': {
            'serializer_class': Serializer,
        }
    }
    """
    version_map = None

    def get_version_map(self):
        if not self.version_map:
            return ImproperlyConfigured('Version map was not provided')
        return self.version_map

    def get_versioned_serializer_class(self, version):
        version_map = self.get_version_map()
        return version_map[version]['serializer_class']

    def get_serializer_class(self):
        version = self.request.version
        return self.get_versioned_serializer_class(version)
