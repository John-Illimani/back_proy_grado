from rest_framework import generics
from ..serializers.serializer_section import ParaleloSerilizer
from ..models.section_model import Paralelo
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated


class ParaleloListView(generics.ListCreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Paralelo.objects.all()
    serializer_class = ParaleloSerilizer


class ParaleloDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Paralelo.objects.all()
    serializer_class = ParaleloSerilizer
