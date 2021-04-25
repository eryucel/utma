from rest_framework import serializers
from utma.dataset.models import Dataset


# Create your serializers here.
class DatasetSerializer(serializers.ModelSerializer):
    url = serializers.HyperlinkedIdentityField(
        view_name='dataset:detail',
        lookup_field='slug'
    )
    # username = serializers.SerializerMethodField(method_name='username_new')
    username = serializers.SerializerMethodField()

    class Meta:
        model = Dataset
        fields = [
            'title',
            'user',
            'username',
            'slug',
            'url',
            'created',
            'modified'
        ]
        # exclude = [
        #     'created',
        #     'modified'
        # ]

    def get_username(self, obj):
        return str(obj.user.username)


class DatasetCreateUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = [
            'title',
            'user',
        ]

    # def save(self, **kwargs):
    #     return kwargs
    #
    # def create(self, validated_data):
    #     return validated_data
    #
    # def update(self, instance, validated_data):
    #     return instance
    #
    # def validate(self, attrs):
    #     return attrs
    #
    # def validate_title(self, value):
    #     return value
