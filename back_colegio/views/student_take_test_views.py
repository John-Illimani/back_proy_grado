from rest_framework import generics
from ..serializers.serializer_student_take_test import StudentTestSerilizer 
from ..models.student_takes_test_model import Estudiante_test 
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class StudentTestListView(generics.ListCreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Estudiante_test.objects.all()
    serializer_class = StudentTestSerilizer


class StudentTestDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Estudiante_test.objects.all()
    serializer_class = StudentTestSerilizer

class StudentTestDeleteByStudentView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def delete(self, request, estudiante_id):
        """
        Elimina TODAS las pruebas (tests) asociadas a un estudiante específico.
        """
        tests = Estudiante_test.objects.filter(estudiante_id=estudiante_id)
        count = tests.count()

        if count == 0:
            return Response(
                {"detail": "No se encontraron tests para este estudiante."},
                status=status.HTTP_404_NOT_FOUND
            )

        tests.delete()

        return Response(
            {"deleted": count, "detail": f"{count} tests eliminados correctamente."},
            status=status.HTTP_200_OK
        )