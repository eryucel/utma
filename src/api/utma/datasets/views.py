from rest_framework.generics import ListAPIView

from utma.datasets.serializers import DatasetSerializer
from utma.datasets.models import Dataset


# Create your views here.
class DatasetListAPIView(ListAPIView):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
