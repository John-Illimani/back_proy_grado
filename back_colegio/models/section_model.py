from django.db import models
from .teacher_model import Profesor

# --------------------------
# Modelo Paralelo
# --------------------------
class Paralelo(models.Model):
    nombre = models.CharField(max_length=50)  
    profesor = models.OneToOneField(
        Profesor,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="paralelo"
    )

    def __str__(self):
        return f"Paralelo {self.nombre}"



