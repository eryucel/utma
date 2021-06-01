# Generated by Django 3.2 on 2021-05-31 13:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0006_auto_20210531_0103'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='result',
            name='accuracy',
        ),
        migrations.AddField(
            model_name='dataset',
            name='col',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='dataset',
            name='row',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='task',
            name='algorithm',
            field=models.IntegerField(),
        ),
        migrations.DeleteModel(
            name='Algorithm',
        ),
        migrations.DeleteModel(
            name='AlgorithmType',
        ),
    ]
