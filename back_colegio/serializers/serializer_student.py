from rest_framework import serializers
from ..models.student_model import Estudiante

# para pasar de objeto a json
class StudentSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Estudiante
        fields = "__all__"
    
        
