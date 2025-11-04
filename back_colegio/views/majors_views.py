from rest_framework import generics
from ..serializers.serializer_majors import MajorsSerilizer
from ..models.majors_model import Carreras
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class MajorsListView(generics.ListCreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Carreras.objects.all()
    serializer_class = MajorsSerilizer


class MajorsDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Carreras.objects.all()
    serializer_class = MajorsSerilizer


class MajorsDeleteByStudentView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def delete(self, request, estudiante_id):
        """
        Elimina TODAS las carreras asociadas a un estudiante específico.
        """
        carreras = Carreras.objects.filter(estudiante_id=estudiante_id)
        count = carreras.count()

        if count == 0:
            return Response(
                {"detail": "No se encontraron carreras para este estudiante."},
                status=status.HTTP_404_NOT_FOUND
            )

        carreras.delete()

        return Response(
            {"deleted": count, "detail": f"{count} carreras eliminadas correctamente."},
            status=status.HTTP_200_OK
        )