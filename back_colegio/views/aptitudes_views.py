from rest_framework import generics
from ..serializers.serializer_aptitudes import AptitudesSerilizer
from ..models.aptitudes_model import Aptitudes 
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated


class AptitudesListView(generics.ListCreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Aptitudes.objects.all()
    serializer_class = AptitudesSerilizer


class AptitudesDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Aptitudes.objects.all()
    serializer_class = AptitudesSerilizer
