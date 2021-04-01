from django.urls import path

from utma.datasets.views import (DatasetListAPIView,
                                 DatasetDetailAPIView,
                                 DatasetDeleteAPIView,
                                 DatasetUpdateAPIView,
                                 DatasetCreateAPIView)

urlpatterns = [
    path('list', DatasetListAPIView.as_view(), name='list'),
    path('detail/<slug>', DatasetDetailAPIView.as_view(), name='detail'),
    path('delete/<slug>', DatasetDeleteAPIView.as_view(), name='delete'),
    path('update/<slug>', DatasetUpdateAPIView.as_view(), name='update'),
    path('create', DatasetCreateAPIView.as_view(), name='create')
]
