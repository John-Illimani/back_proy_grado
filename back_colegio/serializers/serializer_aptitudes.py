from rest_framework import serializers
from ..models.aptitudes_model import Aptitudes

class AptitudesSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Aptitudes
        fields = "__all__"
        
