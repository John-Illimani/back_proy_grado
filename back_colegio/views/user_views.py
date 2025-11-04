

from rest_framework import generics
from ..serializers.serializer_user import UserSerializer
from ..models.usuario_model import Usuario
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated

# from rest_framework.pagination import PageNumberPagination




# class StandardResultsSetPagination(PageNumberPagination):
#     page_size = 25  # Número de usuarios por página
#     page_size_query_param = 'page_size'
#     max_page_size = 100


# ------------------------------------------
# USUARIO
# ------------------------------------------
class UserlListView(generics.ListCreateAPIView):
    # jwt que esta lista solo puedan ver los usuarios autenticados
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    queryset = Usuario.objects.all()
    serializer_class = UserSerializer

    # pagination_class = StandardResultsSetPagination

# para el manejo de los metodos


class UserDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Usuario.objects.all()
    serializer_class = UserSerializer






