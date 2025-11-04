from rest_framework import generics
from ..serializers.serializer_tests import *
from ..models.tests_model import *
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import generics, status
# --- ¡IMPORTA LA PAGINACIÓN! ---
from rest_framework.pagination import PageNumberPagination

from rest_framework.views import APIView

# --- (OPCIONAL PERO RECOMENDADO) DEFINE UNA CLASE DE PAGINACIÓN ESTÁNDAR ---
# Puedes ajustar 'page_size' al número de items que quieras por página
class StandardResultsSetPagination(PageNumberPagination):
    page_size = 50 # Ejemplo: 50 items por página
    page_size_query_param = 'page_size' # Permite al frontend cambiar el tamaño (?page_size=100)
    max_page_size = 200 # Límite máximo que el frontend puede pedir

#------------------------------------------
#TEST_VOCACIONAL
#------------------------------------------
class TestVocationalListView(generics.ListCreateAPIView):
    authentication_classes=[JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset =  TestVocacional.objects.all()
    serializer_class  = TestVocacionalSerilizer
    # --- ¡MEJORA AÑADIDA! ---
    pagination_class = StandardResultsSetPagination # Activa la paginación para esta lista

# (DetailView no necesita paginación)
class TestVocationalDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes=[JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset =  TestVocacional.objects.all()
    serializer_class  = TestVocacionalSerilizer

#------------------------------------------
#PREGUNTA
#------------------------------------------
class PreguntaListView(generics.ListCreateAPIView):
    authentication_classes=[JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset =  Pregunta.objects.all() # Considera usar .select_related() u .only() si es necesario
    serializer_class  = PreguntaSerilizer
    # --- ¡MEJORA AÑADIDA! ---
    pagination_class = StandardResultsSetPagination # Activa la paginación

# (DetailView no necesita paginación)
class PreguntaDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes=[JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset =  Pregunta.objects.all()
    serializer_class  = PreguntaSerilizer

#------------------------------------------
#OPCION_RESPUESTA
#------------------------------------------
class OpcionRespuestaListView(generics.ListCreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    serializer_class = OpcionRespuestaSerilizer
    pagination_class = StandardResultsSetPagination 
    
    # 1. MÉTODO GET (Consulta y Paginación)
    # ------------------------------------
    def get_queryset(self):
        queryset = OpcionRespuesta.objects.all()
        
        # Filtrado por estudiante (como ya lo tenías)
        estudiante_id = self.request.query_params.get('estudiante')
        if estudiante_id:
            queryset = queryset.filter(estudiante_id=estudiante_id)
        
        # Filtrado por rangos de preguntas (como ya lo tenías)
        pregunta_gte = self.request.query_params.get('pregunta__gte')
        pregunta_lte = self.request.query_params.get('pregunta__lte')

        if pregunta_gte and pregunta_lte:
            queryset = queryset.filter(
                pregunta__id__gte=pregunta_gte, 
                pregunta__id__lte=pregunta_lte
            )
        
        # ✅ CORRECCIÓN 1: Añadir un orden explícito
        # Esto elimina el "UnorderedObjectListWarning" del log y garantiza 
        # que la paginación sea consistente.
        return queryset.order_by('pregunta__id', 'id') 

    # 2. MÉTODO POST (Creación Masiva/Bulk Create)
    # -------------------------------------------
    def create(self, request, *args, **kwargs):
        is_many = isinstance(request.data, list)
        if not is_many:
            # Si es un solo objeto, usar el flujo estándar de DRF
            return super().create(request, *args, **kwargs)

        # Si es una lista, usar la lógica de bulk create
        serializer = self.get_serializer(data=request.data, many=True)
        serializer.is_valid(raise_exception=True)
        
        # self.perform_create llama al create() de BulkCreateListSerializer,
        # el cual usa Model.objects.bulk_create(..., ignore_conflicts=True)
        self.perform_create(serializer) 
        
        headers = self.get_success_headers(serializer.data)
        
        # DRF requiere un Response, pero el resultado de bulk_create a veces es vacío
        return Response(
            serializer.data or {"status": f"{len(request.data)} creados o actualizados"}, 
            status=status.HTTP_201_CREATED, 
            headers=headers
        )

# (DetailView no necesita paginación)
class OpcionRespuestaDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes=[JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset =  OpcionRespuesta.objects.all()
    serializer_class  = OpcionRespuestaSerilizer


class OpcionRespuestaDeleteByStudentView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def delete(self, request, estudiante_id):
        respuestas = OpcionRespuesta.objects.filter(estudiante_id=estudiante_id)
        count = respuestas.count()

        if count == 0:
            return Response(
                {"detail": "No se encontraron respuestas para este estudiante."},
                status=status.HTTP_404_NOT_FOUND
            )

        # Borrado masivo
        respuestas.delete()

        return Response(
            {"deleted": count, "detail": f"{count} respuestas eliminadas correctamente."},
            status=status.HTTP_200_OK
        )

#------------------------------------------
#RESPUESTA_ESTUDIANTE
#------------------------------------------
class RespuestaEstudianteListView(generics.ListCreateAPIView):
    authentication_classes=[JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset =  RespuestaEstudiante.objects.all()
    serializer_class  = RespuestaEstudianteSerilizer
    # --- ¡MEJORA AÑADIDA! ---
    pagination_class = StandardResultsSetPagination # Activa la paginación

    # --- ¡CONSIDERACIÓN IMPORTANTE! ---
    # Si también envías RespuestaEstudiante en 'bulk' (listas),
    # deberás añadir el método 'create' aquí también (como en OpcionRespuestaListView)
    # y configurar el 'RespuestaEstudianteSerilizer' con el 'list_serializer_class'.
    # Si solo creas una a la vez, no necesitas cambiar nada aquí.

# (DetailView no necesita paginación)
class RespuestaEstudianteDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes=[JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset =  RespuestaEstudiante.objects.all()
    serializer_class  = RespuestaEstudianteSerilizer



from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from ..models.tests_model import OpcionRespuesta


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
# Asegúrate de importar tus modelos y autenticación
# from .models import Estudiante, OpcionRespuesta 
# from rest_framework_simplejwt.authentication import JWTAuthentication
# from rest_framework.permissions import IsAuthenticated

class OpcionRespuestaBulkUpdateView(APIView):
    """
    Permite actualizar múltiples respuestas (texto o valor) en una sola llamada PATCH.
    Solo actualiza las respuestas del estudiante autenticado.
    
    Optimización: Evita el problema de N+1 queries al usar un solo filtro inicial.
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def patch(self, request):
        data = request.data

        if not isinstance(data, list):
            return Response(
                {"error": "Debe enviar una lista de objetos"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 🔹 Paso 1: Buscar el estudiante asociado al usuario autenticado
        try:
            estudiante = Estudiante.objects.get(usuario=request.user)
        except Estudiante.DoesNotExist:
            return Response(
                {"error": "No se encontró un perfil de estudiante asociado a este usuario."},
                status=status.HTTP_404_NOT_FOUND
            )

        # 🔹 Paso 2: Optimización de la búsqueda (Evitar N+1 queries)
        # Extraer los IDs de las respuestas que se quieren actualizar del payload
        ids_a_actualizar = [item.get("id") for item in data if item.get("id")]
        
        if not ids_a_actualizar:
             return Response(
                {"updated": 0, "detail": "El payload no contiene IDs válidas para actualizar."},
                status=status.HTTP_200_OK
            )
        
        # Obtener todos los objetos OpcionRespuesta en UNA SOLA consulta a la DB
        respuestas_existentes = OpcionRespuesta.objects.filter(
            id__in=ids_a_actualizar,
            estudiante=estudiante
        )
        
        # Crear un mapa (diccionario) para búsqueda O(1) en memoria
        respuestas_map = {obj.id: obj for obj in respuestas_existentes}

        updated_objs = []
        
        # 🔹 Paso 3: Iterar, aplicar cambios en memoria y preparar la lista para bulk_update
        for item in data:
            resp_id = item.get("id")
            obj = respuestas_map.get(resp_id)
            
            # Verificar que el objeto existe y pertenece al estudiante
            if obj:
                # Aplicar los cambios solo si los campos están presentes en el payload
                if "texto" in item:
                    obj.texto = item["texto"]
                if "valor" in item:
                    # Se asume que el campo 'valor' puede necesitar conversión si se recibe como string
                    try:
                        obj.valor = float(item["valor"]) 
                    except (ValueError, TypeError):
                        obj.valor = item["valor"] # Mantener el valor original o el que viene si falla la conversión
                        
                updated_objs.append(obj)

        # 🔹 Paso 4: Sincronizar con la DB usando bulk_update
        if updated_objs:
            # Solo actualiza los campos 'texto' y 'valor' en la DB
            OpcionRespuesta.objects.bulk_update(updated_objs, ["texto", "valor"])
            return Response(
                {"updated": len(updated_objs)},
                status=status.HTTP_200_OK
            )

        return Response(
            {"updated": 0, "detail": "No se encontraron respuestas válidas para actualizar."},
            status=status.HTTP_200_OK
        )