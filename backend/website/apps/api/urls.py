from django.urls import path, include

app_name = "api"

urlpatterns = [
    path("sharks/", include("apps.api.sharks.urls")),
]
