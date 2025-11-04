from rest_framework import serializers
from ..models.tests_model import TestVocacional, Pregunta, OpcionRespuesta, RespuestaEstudiante
# --- ¡IMPORTA ESTO! ---
from rest_framework.serializers import ListSerializer
from rest_framework.validators import UniqueTogetherValidator

# --- CLASE DE AYUDA PARA BULK CREATE ---
# (La misma que usamos antes)
class BulkCreateListSerializer(ListSerializer):
    """
    Sobrescribe el método 'create' del ListSerializer
    para usar 'bulk_create' de Django.
    """
    def create(self, validated_data):
        ModelClass = self.child.Meta.model
        objs = [
            ModelClass(**item) for item in validated_data
        ]
        try:
            return ModelClass.objects.bulk_create(objs, ignore_conflicts=True)
        except Exception as e:
            print(f"Error en bulk_create: {e}")
            raise serializers.ValidationError("Error durante la creación masiva.")
            

# --------------------------------------------------
# SERIALIZERS
# --------------------------------------------------

class TestVocacionalSerilizer(serializers.ModelSerializer):
    class Meta:
        model = TestVocacional
        fields = "__all__"

class PreguntaSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Pregunta
        fields = "__all__"

class OpcionRespuestaSerilizer(serializers.ModelSerializer):
    class Meta:
        model = OpcionRespuesta
        fields = "__all__"
        list_serializer_class = BulkCreateListSerializer
        
        # 🚨 PASO 2: AÑADIR EL VALIDADOR DE UNICIDAD COMPUESTA
        validators = [
            UniqueTogetherValidator(
                queryset=OpcionRespuesta.objects.all(),
                fields=['pregunta', 'estudiante'], # <-- La clave única es esta combinación
                message="Ya existe una respuesta de este estudiante para esta pregunta."
            )
        ]

class RespuestaEstudianteSerilizer(serializers.ModelSerializer):
    class Meta:
        model = RespuestaEstudiante
        fields = "__all__"
        list_serializer_class = BulkCreateListSerializer