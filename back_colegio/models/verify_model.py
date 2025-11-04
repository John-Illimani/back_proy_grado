from django.db import models


class Token(models.Model):
    cambios = models.CharField(max_length=2)
