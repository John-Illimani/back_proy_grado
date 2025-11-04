from django.db import models
from .usuario_model import Usuario
from .section_model import Paralelo


class Estudiante(models.Model):
    usuario = models.OneToOneField(
        Usuario,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="perfil_estudiante_usuario"
    )
    paralelo = models.ForeignKey(
        Paralelo,
        on_delete=models.SET_NULL,
        null=True,    #esto puede guardar un null en la base de datos
        blank=True,  #esto para formularios de django, que significa que se puede mandar fomularios sin este argumento, pero si esta en false se exige al front
        related_name="paralelo_estudiante_usuario"
    )

