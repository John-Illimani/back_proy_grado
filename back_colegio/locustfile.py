import random
from locust import HttpUser, task, between

# =============================================================================
# TOKEN DE AUTENTICACIÓN FIJO
# =============================================================================
# Token del usuario con ID 3.
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzYxMzExOTQ1LCJpYXQiOjE3NjEyMjU1NDUsImp0aSI6ImRmNzJkNjk5ZTc1YTQ3Y2E4YWU2NGMzMWQ5NmMzMTQ5IiwidXNlcl9pZCI6IjMifQ.ykDUVXgOswAkiqrYzKXzIExgGQCwTcms4FOprfp2XbU"


# =============================================================================
# PERSONA: USUARIO ADMINISTRADOR (Prueba Final)
# =============================================================================
class AdminUser(HttpUser):
    wait_time = between(1, 3)
    
    # Esta lista guardará los IDs de los NUEVOS usuarios
    # que este script cree.
    created_user_ids = [] 

    def on_start(self):
        """
        Configura la cabecera con el token.
        """
        self.client.headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}"
        }

    @task(5) 
    def view_user_list(self):
        """
        Prueba de LECTURA (GET) a la lista (paginada).
        Pide solo la página 1, que sabemos que existe.
        """
        self.client.get("/api/add_user/?page=1")

    @task(2) 
    def create_user_and_save_id(self):
        i = random.randint(1000, 9999)
        payload = {
            "first_name": f"UsuarioLocust_{i}",
            "last_name": f"Apellido_{i}",
            "rol": "docente" 
        }
        
        # --- ¡ARREGLO AQUÍ! ---
        # Añade 'catch_response=True' dentro del post()
        with self.client.post("/api/add_user/", json=payload, name="/api/add_user/", catch_response=True) as response:
            if response.status_code == 201: # 201 Created
                try:
                    # Guardamos el ID del usuario recién creado
                    new_id = response.json().get("id")
                    if new_id:
                        self.created_user_ids.append(new_id)
                except Exception:
                    pass

    @task(3) 
    def view_user_detail(self):
        """
        Prueba de LECTURA (GET) a un detalle.
        Ahora solo pide IDs que sabemos que existen.
        """
        if not self.created_user_ids:
            # Si la lista está vacía, pide el ID 3 (nuestro ID)
            user_id_to_get = 3
        else:
            # Elige un ID al azar de la lista (los nuevos o el ID 3)
            # (Agregamos el 3 para más variedad)
            user_id_to_get = random.choice(self.created_user_ids + [3])
        
        self.client.get(
            f"/api/add_user/{user_id_to_get}/", 
            name="/api/add_user/[pk]"
        )



# import random
# from locust import HttpUser, task, between

# # =============================================================================
# # TOKEN DE AUTENTICACIÓN FIJO
# # =============================================================================
# # Este es el token JWT que me diste (pertenece al user_id: 3).
# # ADVERTENCIA: Este token caduca hoy, 23 de Octubre de 2025,
# # aproximadamente a las 14:39 (hora de Bolivia).
# AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzYxMzExOTQ1LCJpYXQiOjE3NjEyMjU1NDUsImp0aSI6ImRmNzJkNjk5ZTc1YTQ3Y2E4YWU2NGMzMWQ5NmMzMTQ5IiwidXNlcl9pZCI6IjMifQ.ykDUVXgOswAkiqrYzKXzIExgGQCwTcms4FOprfp2XbU"

# # =============================================================================
# # ID DEL ESTUDIANTE DE PRUEBA
# # =============================================================================
# # El ID del estudiante para el que enviaremos las respuestas
# # (basado en tu ejemplo: {"estudiante": 171}).
# # Asegúrate de que el estudiante con ID 171 exista en tu base de datos.
# TARGET_STUDENT_ID = 171


# # =============================================================================
# # PERSONA: EL "ESTUDIANTE" (Simulado por un Admin)
# # =============================================================================
# class EstudianteUser(HttpUser):
#     # Tiempo de espera entre que un usuario virtual termina de enviar
#     # las 793 respuestas y empieza de nuevo.
#     wait_time = between(10, 30)
    
#     # Atributos para este usuario
#     auth_token = AUTH_TOKEN
#     student_id = TARGET_STUDENT_ID

#     def on_start(self):
#         """
#         Se ejecuta 1 vez por usuario virtual al iniciar.
#         No inicia sesión, solo configura la cabecera de autenticación.
#         """
#         # 1. Configurar el token en todas las peticiones futuras
#         self.client.headers = {"Authorization": f"Bearer {self.auth_token}"}
        
#         # 2. Confirmar el ID del estudiante (ya está puesto arriba)
#         if not self.student_id:
#              print("ERROR: TARGET_STUDENT_ID no está configurado.")
#              self.environment.runner.quit()

#     @task # Esta es la tarea principal
#     def submit_full_test_individually(self):
#         """
#         Simula a un estudiante enviando sus 793 respuestas,
#         UNA POR UNA, lo más rápido posible.
#         """
#         print(f"Usuario (con token de admin) iniciando envío de 793 respuestas para Estudiante ID: {self.student_id}...")

#         # --- Bucle de 793 peticiones POST ---
#         for pregunta_id in range(1, 794):
            
#             # Basado en tu ejemplo: {"texto": "vacio","valor": 0,"pregunta": 6,"estudiante": 171}
#             payload = {
#                 "pregunta": pregunta_id,
#                 "estudiante": self.student_id, # Usará el ID 171
#                 "texto": "respuesta_locust", # Valor simulado
#                 "valor": random.randint(1, 5)  # Valor simulado
#             }
            
#             # Endpoint: /api/student_response/
#             self.client.post(
#                 "/api/option_response/", 
#                 json=payload,
#                 name="/api/option_response/" # Agrupa todas las 793 peticiones en una sola fila
#             )

#         print(f"Envío completo para Estudiante ID: {self.student_id}.")