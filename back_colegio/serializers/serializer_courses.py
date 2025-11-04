from rest_framework import serializers
from ..models.course_model import Materia

# para pasar de objeto a json
class CourseSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Materia
        fields = "__all__"
        
