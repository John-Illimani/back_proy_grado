from django.db import models
from django.contrib.auth.models import AbstractUser

#aqui el modelo de AbstractUser me crea campos por defecto
class Usuario(AbstractUser):
    ROLE_CHOICES = (
        ('estudiante', 'Estudiante'),
        ('docente', 'Docente'),
        ('admin', 'Administrador'),
    )
    rol = models.CharField(max_length=30, choices=ROLE_CHOICES, default='estudiante')

  

    def __str__(self):
        return f"{self.username} ({self.rol})"
