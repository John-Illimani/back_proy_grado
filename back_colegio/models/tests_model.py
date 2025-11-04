from django.db import models
from django.contrib.auth.models import AbstractUser
from .usuario_model import Usuario
from .student_model import Estudiante

# --------------------------
# TEST VOCACIONAL
# --------------------------
class TestVocacional(models.Model):
    nombre = models.CharField(max_length=200)
    descripcion = models.TextField(blank=True, null=True)
    total_preguntas = models.IntegerField()

    def __str__(self):
        return self.nombre


# --------------------------
# PREGUNTAS
# --------------------------
class Pregunta(models.Model):

    test = models.ForeignKey(
        TestVocacional, on_delete=models.CASCADE, related_name="preguntas_test_chaside")
    texto = models.TextField()
    numero = models.IntegerField()

    def __str__(self):
        return f"Pregunta {self.numero}: {self.texto[:50]}"


# --------------------------
# OPCIONES DE RESPUESTA
# --------------------------
class OpcionRespuesta(models.Model):
    pregunta = models.ForeignKey(
        Pregunta, on_delete=models.CASCADE, related_name="opciones")
    texto = models.CharField(max_length=400)
    valor = models.IntegerField()
    estudiante = models.ForeignKey(
        Estudiante, on_delete=models.CASCADE, related_name="estudiante")

    def __str__(self):
        return f"{self.texto} (Pregunta {self.pregunta.numero})"
    class Meta:
        # ✅ SOLUCIÓN CRÍTICA PARA EL ERROR 400 EN EL POST MASIVO:
        # La combinación de pregunta y estudiante debe ser única en la DB.
        unique_together = ('pregunta', 'estudiante')



# --------------------------
# RESPUESTAS DE ESTUDIANTES
# --------------------------
class RespuestaEstudiante(models.Model):
    estudiante = models.ForeignKey(
        Usuario, on_delete=models.CASCADE, related_name="respuestas_es")
    pregunta = models.ForeignKey(
        Pregunta, on_delete=models.CASCADE, related_name="respuestas_pr")
    opcion = models.ForeignKey(
        OpcionRespuesta, on_delete=models.CASCADE, related_name="respuestas_op")
    fecha = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.estudiante.username} → {self.pregunta.numero}: {self.opcion.texto}"
