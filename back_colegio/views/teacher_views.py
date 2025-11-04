

from rest_framework import generics
from ..serializers.serializer_teacher import TeacherSerilizer
from ..models.teacher_model import Profesor
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated


# ------------------------------------------
# TEST_VOCACIONAL
# ------------------------------------------
class TeacherListView(generics.ListCreateAPIView):
    # jwt que esta lista solo puedan ver los usuarios autenticados
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    queryset = Profesor.objects.all()
    serializer_class = TeacherSerilizer

# para el manejo de los metodos


class TeacherDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Profesor.objects.all()
    serializer_class = TeacherSerilizer
