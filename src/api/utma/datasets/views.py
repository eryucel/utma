from rest_framework.generics import ListAPIView, RetrieveAPIView, DestroyAPIView, UpdateAPIView, CreateAPIView

from utma.datasets.serializers import DatasetSerializer
from utma.datasets.models import Dataset
from rest_framework.permissions import (
    IsAuthenticated,
    IsAdminUser
)
from utma.datasets.permissions import IsOwnerOrAdmin


# Create your views here.
class DatasetListAPIView(ListAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    permission_classes = [IsAdminUser]


class DatasetDetailAPIView(RetrieveAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    lookup_field = 'slug'
    permission_classes = [IsOwnerOrAdmin]


class DatasetDeleteAPIView(DestroyAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    lookup_field = 'slug'
    permission_classes = [IsOwnerOrAdmin]


class DatasetUpdateAPIView(RetrieveAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    lookup_field = 'slug'
    permission_classes = [IsOwnerOrAdmin]
    # def perform_update(self, serializer):
    #     serializer.save(modified_by=self.request.user)


class DatasetCreateAPIView(CreateAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
