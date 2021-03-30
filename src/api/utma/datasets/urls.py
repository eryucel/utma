from django.urls import path

from utma.datasets.views import DatasetListAPIView

urlpatterns = [
    path('list', DatasetListAPIView.as_view(), name='list')
]
