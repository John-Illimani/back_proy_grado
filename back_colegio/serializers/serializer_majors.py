from rest_framework import serializers
from ..models.majors_model import Carreras


class MajorsSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Carreras
        fields = "__all__"
        
