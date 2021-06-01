from django.db import models


# Create your models here.

class File(models.Model):
    file = models.FileField(blank=False, null=False)


class Dataset(models.Model):
    name = models.CharField(max_length=200)
    col = models.IntegerField(default=0)
    row = models.IntegerField(default=0)
    categoricalAttributes = models.CharField(max_length=1000, blank=True, null=True)
    numericAttributes = models.CharField(max_length=1000, blank=True, null=True)
    data = models.CharField(max_length=200)


class Task(models.Model):
    algorithmName = models.CharField(max_length=200, blank=True, null=True)
    algorithm = models.IntegerField()
    datasetName = models.CharField(max_length=200, blank=True, null=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    status = models.IntegerField()
    parameters = models.TextField()
    completed_date = models.DateTimeField(blank=True, null=True)
    create_date = models.DateTimeField(auto_now=True)


class Result(models.Model):
    algorithmName = models.CharField(max_length=200, blank=True, null=True)
    datasetName = models.CharField(max_length=200, blank=True, null=True)
    task = models.ForeignKey(Task, on_delete=models.CASCADE)
    data = models.TextField()
