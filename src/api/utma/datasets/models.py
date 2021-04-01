from django.db import models
from django.contrib.auth.models import User

# Create your models here.
from django.utils import timezone
from django.utils.text import slugify


class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=1)
    title = models.CharField(max_length=120)
    # image = models.ImageField(upload_to='media/dataset')
    modified = models.DateTimeField(editable=False)
    created = models.DateTimeField(editable=False)
    slug = models.SlugField(unique=True, max_length=150, editable=False)

    def get_slug(self):
        slug = slugify(self.user.username + '-' +
                       self.title.replace("ı", "i").replace("ş", "s").replace("ü", "u")
                       .replace("ç", "c").replace("ö", "o").replace("ğ", "g"))
        unique = slug
        number = 1
        while Dataset.objects.filter(slug=unique).exists():
            unique = '{}-{}'.format(slug, number)
            number += 1
        return unique

    def save(self, *args, **kwargs):
        if not self.id:
            self.created = timezone.now()
        self.modified = timezone.now()
        self.slug = self.get_slug()
        return super(Dataset, self).save(*args, **kwargs)

    def __str__(self):
        return self.slug
