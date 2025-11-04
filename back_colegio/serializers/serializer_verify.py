from rest_framework import serializers
from ..models.verify_model import Token


class TokenSerilizer(serializers.ModelSerializer):
    class Meta:
        model = Token
        fields = "__all__"
        
