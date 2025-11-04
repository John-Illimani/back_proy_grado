from django.db import models
from .usuario_model import Usuario


class Profesor(models.Model):
    usuario = models.OneToOneField(
        Usuario,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="perfil_profesor_usuario"
    )

 