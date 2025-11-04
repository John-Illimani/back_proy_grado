# en tu archivo signals.py

from django.db.models.signals import post_save
from django.dispatch import receiver

# Importa todos los modelos necesarios
from .models.usuario_model import Usuario
from .models.student_model import Estudiante
from .models.teacher_model import Profesor
from .models.majors_model import Carreras
# Importa la función de la IA
from .services import ejecutar_prediccion_y_guardar
# Importa tus otros modelos y la función de servicio
from back_colegio.models import Estudiante, OpcionRespuesta
from .models.student_model import Estudiante
from .models.tests_model import OpcionRespuesta
from .models.verify_model import Token

from .services import ejecutar_prediccion_y_guardar


# --- SEÑAL 1: SE DISPARA AL CREAR UN USUARIO ---
@receiver(post_save, sender=Usuario)
def crear_perfiles_iniciales(sender, instance, created, **kwargs):
    """
    Si se crea un nuevo Usuario, esta señal crea los perfiles asociados
    (Estudiante o Profesor) y un registro inicial en Carreras.
    """
    if created:  # Solo se ejecuta cuando el usuario es nuevo
        if instance.rol == "estudiante":
            # Crea el perfil de Estudiante
            estudiante_creado = Estudiante.objects.create(usuario=instance)
            
            # Crea el registro inicial en la tabla Carreras
            Carreras.objects.create(
                estudiante=estudiante_creado,
                carreras="sin carreras"  # Valor por defecto
            )
            print(f"Perfil de Estudiante y registro de Carreras creados para {instance.username}")

        elif instance.rol == "docente":
            # Crea el perfil de Profesor
            Profesor.objects.create(usuario=instance)
            print(f"Perfil de Docente creado para {instance.username}")




# --- Constante para el número total de preguntas ---
TOTAL_PREGUNTAS = 793

@receiver(post_save, sender=Token)
def disparar_prediccion_por_token_personalizado(sender, instance, **kwargs):
    """
    Se ejecuta después de que un registro en la tabla Token es guardado/actualizado.
    Verifica si el estudiante asociado ha completado el test para iniciar la predicción.
    """
    try:
        # 1. OBTENEMOS AL ESTUDIANTE A PARTIR DEL TOKEN
        # Asumo que tu modelo 'Token' tiene una relación con el 'Usuario'.
        # Si la relación es directa con 'Estudiante', cambia 'instance.usuario' por 'instance.estudiante'.
        usuario = instance.usuario 
        estudiante = Estudiante.objects.get(usuario=usuario)

    except (AttributeError, Estudiante.DoesNotExist):
        # Si el token no tiene el campo 'usuario' o no pertenece a un estudiante, no hacemos nada.
        return

    # 2. CONDICIÓN DE SEGURIDAD (¡LA CLAVE!)
    # Contamos cuántas respuestas tiene este estudiante en la base de datos.
    num_respuestas = OpcionRespuesta.objects.filter(estudiante=estudiante).count()

    # 3. EJECUTAR LA IA (SOLO SI EL TEST ESTÁ COMPLETO)
    # Si el número de respuestas es igual o mayor al total, ejecutamos la predicción.
    if num_respuestas >= TOTAL_PREGUNTAS:
        print(f"✅ Señal de Token (Personalizado) recibida: El estudiante {estudiante.id} ha completado el test. Iniciando predicción...")
        try:
            # Llamamos a la función que hace todo el trabajo pesado.
            ejecutar_prediccion_y_guardar(estudiante.id)
        except Exception as e:
            print(f"❌ Error al ejecutar la predicción automática para el estudiante {estudiante.id}: {e}")
    else:
        # Si aún no ha terminado, simplemente lo registramos y no hacemos nada más.
        print(f"ℹ️ Señal de Token (Personalizado) recibida para estudiante {estudiante.id}. Aún no ha completado el test ({num_respuestas}/{TOTAL_PREGUNTAS} respuestas). No se ejecuta la IA.")