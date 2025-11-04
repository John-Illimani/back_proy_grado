from rest_framework import serializers
from ..models.student_takes_test_model import Estudiante_test

# para pasar de objeto a json
class StudentTestSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Estudiante_test
        fields = "__all__"
        
