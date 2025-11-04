from django.db import models
from .teacher_model import Profesor
# --------------------------
# Modelo Materia
# --------------------------
class Materia(models.Model):
    nombre = models.CharField(max_length=100)
    descripcion = models.TextField(blank=True, null=True)
    codigo = models.CharField(max_length=20, unique=True)
    profesor = models.ForeignKey(
        Profesor,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="materia"
    )

    def __str__(self):
        return self.nombre