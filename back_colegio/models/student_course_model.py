from django.db import models
from .course_model import Materia
from .student_model import Estudiante
class EstudianteMateria(models.Model):
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.CASCADE,
        related_name="materias"
    )
    materia = models.ForeignKey(
        Materia,
        on_delete=models.CASCADE,
        related_name="estudiantes"
    )

    class Meta:
        unique_together = ("estudiante", "materia")  # equivalente a PK compuesta

    def __str__(self):
        return f"{self.estudiante} - {self.materia}"