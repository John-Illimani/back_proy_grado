from rest_framework import serializers
from ..models.section_model import Paralelo

# para pasar de objeto a json
class ParaleloSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Paralelo
        fields = "__all__"
        
