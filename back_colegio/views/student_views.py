

from rest_framework import generics
from ..serializers.serializer_student import StudentSerilizer
from ..models.student_model import Estudiante
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated


class StudentListView(generics.ListCreateAPIView):

    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    queryset = Estudiante.objects.all()
    serializer_class = StudentSerilizer


class StudentDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Estudiante.objects.all()
    serializer_class = StudentSerilizer
