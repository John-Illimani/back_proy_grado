from rest_framework import serializers

from django.contrib.auth import authenticate
# from ..models.usuario_model import Usuario
from rest_framework_simplejwt.tokens import RefreshToken


# # para pasar de objeto a json

# class UsuarioSerilizer(serializers.ModelSerializer):
#     class Meta:
#         model = Usuario
        


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)
    
    def validate(self, data):
        username = data.get('username')
        password = data.get('password')
        
        user = authenticate(username=username, password=password)
        if user is None:
            raise serializers.ValidationError('Usuario o contraseña incorrectos')
        
        # generar token JWT
        refresh = RefreshToken.for_user(user)
        
        return {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'rol': user.rol,
            'username': user.username
        }