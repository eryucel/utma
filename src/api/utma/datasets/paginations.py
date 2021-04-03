from rest_framework.pagination import PageNumberPagination


class DatasetPagination(PageNumberPagination):
    page_size = 5
