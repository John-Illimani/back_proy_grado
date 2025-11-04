

from rest_framework import generics

# from ..serializers.serializer import UsuarioSerilizer
# from rest_framework_simplejwt.authentication import JWTAuthentication
# from rest_framework.permissions import IsAuthenticated



from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ..serializers.serializer_login import LoginSerializer

# class UsuarioListView(generics.ListCreateAPIView):
#     # jwt que esta lista solo puedan ver los usuarios autenticados
#     authentication_classes=[JWTAuthentication]
#     permission_classes = [IsAuthenticated]
#     serializer_class  = UsuarioSerilizer

#     def get(self, request):
#         if request.user.rol == "estudiante":
#             return Response({"message": "Videos para estudiantes"})
#         elif request.user.rol == "docente":
#             return Response({"message": "Videos para docentes"})
#         else:
#             return Response({"message": "Acceso admin"})




class LoginView(APIView):
    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return Response(serializer.validated_data, status=status.HTTP_200_OK)