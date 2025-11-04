from rest_framework import serializers
from ..models.usuario_model import Usuario
from django.contrib.auth.hashers import make_password

def generar_username_unico(base_username):
    """
    Genera un username único agregando un número al final si ya existe.
    """
    username = base_username.lower()
    contador = 1
    while Usuario.objects.filter(username=username).exists():
        username = f"{base_username}{contador}".lower()
        contador += 1
    return username

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = Usuario
        # Puedes poner "__all__" o listar los campos que quieras exponer
        fields = "__all__"
        extra_kwargs = {
            "username": {"required": False},
            "password": {"required": False,"write_only": True},
            "email": {"required": False},
        }

    def create(self, validated_data):
        # Generar username y password si no vienen
        primer_nombre = validated_data['first_name'].split(" ")[0]
        primer_apellido = validated_data['last_name'].split(" ")[0]

        if not validated_data.get('username'):
            base_username = f"{primer_nombre[0]}{primer_apellido}"
            validated_data['username'] = generar_username_unico(base_username)

        if not validated_data.get('password'):
            password_generada = f"{primer_nombre}_{primer_apellido}".lower()
            validated_data['password'] = make_password(password_generada)

        if not validated_data.get('email'):
            validated_data['email'] = f"{validated_data['username']}@colegio.edu"

        # Ajustar permisos según rol
        rol = validated_data.get('rol', '')
        if rol == "estudiante":
            validated_data['is_staff'] = False
            validated_data['is_superuser'] = False
            validated_data['is_active'] = True
        elif rol == "docente":
            validated_data['is_staff'] = True
            validated_data['is_superuser'] = False
            validated_data['is_active'] = True
        elif rol == "admin":
            validated_data['is_staff'] = True
            validated_data['is_superuser'] = True
            validated_data['is_active'] = True

        return super().create(validated_data)

    def update(self, instance, validated_data):
        # Actualizar campos normalmente
        password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance
