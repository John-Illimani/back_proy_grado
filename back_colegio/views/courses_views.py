from rest_framework import generics
from ..serializers.serializer_courses import CourseSerilizer
from ..models.course_model import Materia
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated


class CourseListView(generics.ListCreateAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Materia.objects.all()
    serializer_class = CourseSerilizer


class CourseDetailView(generics.RetrieveUpdateDestroyAPIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = Materia.objects.all()
    serializer_class = CourseSerilizer
