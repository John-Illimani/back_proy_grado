from rest_framework import generics
from ..serializers.serializer_grades import GradesSerilizer
from ..models.grades_model import Calificaciones
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated


class GradesListView(generics.ListCreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Calificaciones.objects.all()
    serializer_class = GradesSerilizer


class GradesDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Calificaciones.objects.all()
    serializer_class = GradesSerilizer
