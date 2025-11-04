from rest_framework import serializers
from ..models.grades_model import Calificaciones

# para pasar de objeto a json
class GradesSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Calificaciones
        fields = "__all__"
        
