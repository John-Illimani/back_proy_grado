from rest_framework import serializers
from ..models.teacher_model import Profesor

# para pasar de objeto a json
class TeacherSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Profesor
        fields = "__all__"
        
