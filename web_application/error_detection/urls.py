from django.urls import path
from .views import error_detection_view

urlpatterns = [
    path('', error_detection_view, name='error_detection'),
]