

from rest_framework import generics
from ..serializers.serializer_verify import TokenSerilizer
from ..models.verify_model import Token
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated


class TokenListView(generics.ListCreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Token.objects.all()
    serializer_class = TokenSerilizer


class TokenDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Token.objects.all()
    serializer_class = TokenSerilizer
