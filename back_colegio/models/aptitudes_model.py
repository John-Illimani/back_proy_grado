from django.db import models
from .student_model import Estudiante


class Aptitudes(models.Model):
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="aptitudes_Estudiante"
    )
    # Guarda todas las aptitudes como un diccionario o lista
    aptitudes = models.JSONField(default=dict)
    
    def __str__(self):
        return f"Aptitudes de {self.estudiante}"
