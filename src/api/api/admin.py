from django.contrib import admin
from .models import *

admin.site.register(File)
admin.site.register(Dataset)
admin.site.register(Task)
admin.site.register(Result)
