# api/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import traceback  # <--- ESTA ES LA PARTE IMPORTANTE PARA DEPURAR

# Asegúrate de que la ruta a 'services' sea correcta
from ..services import ejecutar_prediccion_y_guardar 

class PredecirCarreraView(APIView):
    """
    Vista de depuración para encontrar el error 'None'.
    """
    def post(self, request, estudiante_id, *args, **kwargs):
        
        print("\n--- EJECUTANDO VISTA DE DEPURACIÓN (LA VERSIÓN CORRECTA) ---\n")
        
        print(f"✅ Petición recibida para INICIAR la predicción del estudiante {estudiante_id}.")
        try:
            ejecutar_prediccion_y_guardar(estudiante_id)
            
            return Response(
                {"mensaje": f"Procesamiento completado para el estudiante {estudiante_id}."},
                status=status.HTTP_200_OK
            )

        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        
        # --- ESTE BLOQUE ATRAPARÁ Y MOSTRARÁ EL ERROR REAL ---
        except Exception as e:
            print("\n--- INICIO DEL TRACEBACK DETALLADO ---")
            traceback.print_exc()  # Imprime el historial completo del error
            print("--- FIN DEL TRACEBACK DETALLADO ---\n")
            
            error_type = type(e).__name__
            return Response(
                {"error": f"Error inesperado. Revisa la consola del servidor. Tipo: {error_type}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )






            # hola