from django.db import models
from .student_model import Estudiante
from .tests_model import TestVocacional

class Estudiante_test(models.Model):
    estudiante = models.ForeignKey(
        Estudiante,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="estudiante_take_test"
    )
    testvocational = models.ForeignKey(
        TestVocacional,
        on_delete=models.SET_NULL,
        null=True,   
        blank=True,  
        related_name="test_vocational"
    )
    completo = models.FloatField(
        default=0.0,
        help_text="Porcentaje completado del test (0 a 100)"
    )

    def __str__(self):
        return f"{self.estudiante} - {self.testvocational} - {self.completo}% completado"
