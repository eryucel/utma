from django.conf.urls import url
from django.urls import include, path, re_path
from rest_framework import routers
from .views import *

router = routers.DefaultRouter()

urlpatterns = [
    path('', include(router.urls)),
    url(r'^upload/$', FileUploadView.as_view(), name='file-upload'),
    path('task/run/<int:pk>', RunAlgorithm.as_view()),
    path('task/create', TaskCreateApi.as_view()),
    path('task', TaskListApi.as_view()),
    path('dataset/create', DatasetCreateApi.as_view()),
    path('dataset/<int:pk>', DatasetUpdateApi.as_view()),
    path('dataset/delete/<int:pk>/', DatasetDeleteApi.as_view()),
    path('dataset', DatasetListApi.as_view()),
    # path('result', ResultListApi.as_view()),
    # path('result/<int:task>', ResultListApi.as_view()),
    re_path('^result/(?P<task>.+)/$', ResultRetrieveApi.as_view()),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
