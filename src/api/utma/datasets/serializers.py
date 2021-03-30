from rest_framework import serializers
from utma.datasets.models import Dataset


# Create your serializers here.
class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = [
            'title',
            'user'
        ]
