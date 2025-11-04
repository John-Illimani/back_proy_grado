from django.db import models
from .teacher_model import Profesor
from .student_model import Estudiante
from .course_model import Materia

# --------------------------
# MODELO DE CALIFICACIONES
# --------------------------


class Calificaciones(models.Model):
    notas1 = models.FloatField(default=0)
    notas2 = models.FloatField(default=0)
    notas3 = models.FloatField(default=0)
    profesor = models.ForeignKey(
        Profesor,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="calificaciones_profesor"
    )
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="calificaciones_estudiante"
    )

    materia = models.ForeignKey(
        Materia,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="calificaciones_materia"
    )

    def __str__(self):
        return f"{self.estudiante} - {self.materia} : {self.nota}"
