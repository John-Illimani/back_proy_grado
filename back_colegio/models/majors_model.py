from django.db import models
from .student_model import Estudiante



class Carreras(models.Model):
    carreras = models.TextField()
   
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.SET_NULL,
        null=True,    
        blank=True, 
        related_name="estudiante_carreras"
    )

