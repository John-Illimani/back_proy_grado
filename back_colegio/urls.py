from django.urls import path
from .views.login_views import LoginView
from .views.test_views import *
from .views.teacher_views import *
from .views.student_views import *
from .views.user_views import *
from .views.section_views import *
from .views.grades_views import *
from .views.courses_views import *
from .views.student_take_test_views import *
from .views.majors_views import *
from .views.verify_view import *
from .views.buzon_views import *
from .views.aptitudes_views import *
from .views.proxy_views import *


# aqui van los endpoints de la aplicacion 
urlpatterns = [
    # PARA EL LOGIN
    path('login/', LoginView.as_view(), name='login'),
    # PARA LA VISTA DE TEST VOCACIONAL 
    path("tests_vocational/", TestVocationalListView.as_view(), name="test_vocacional_list_view"),
    path('tests_vocational/<int:pk>/', TestVocationalDetailView.as_view(), name='test_vocational_detail_view'),

    path('query/', PreguntaListView.as_view(), name='query_list_view'),
    path('query/<int:pk>/', PreguntaDetailView.as_view(), name='query_detail_view'),

    path('option_response/', OpcionRespuestaListView.as_view(), name='option_response_list_view'),
    path('option_response/<int:pk>/', OpcionRespuestaDetailView.as_view(), name='option_response_detail_view'),
    path('delete_response/<int:estudiante_id>/', OpcionRespuestaDeleteByStudentView.as_view(), name='delete_response'),
    path('update/', OpcionRespuestaBulkUpdateView.as_view(), name='update'),


    path('student_response/', RespuestaEstudianteListView.as_view(), name='student_response_list_view'),
    path('student_response/<int:pk>/', RespuestaEstudianteDetailView.as_view(), name='student_response_detail_view'),

    # PARA LA VISTA DE TEACHER 
    path('add_teacher/', TeacherListView.as_view(), name='teacher_add_list_view'),
    path('add_teacher/<int:pk>/', TeacherDetailView.as_view(), name='teacher_add_detail_view'),

    # PARA LA VISTA DE STUDENT 
    path('add_student/', StudentListView.as_view(), name='student_add_list_view'),
    path('add_student/<int:pk>/', StudentDetailView.as_view(), name='student_add_detail_view'),

    # PARA LA VISTA DE USUARIO 
    path('add_user/', UserlListView.as_view(), name='user_add_list_view'),
    path('add_user/<int:pk>/', UserDetailView.as_view(), name='user_add_detail_view'),

    # PARA LA VISTA DE PARALELO
    path('add_section/', ParaleloListView.as_view(), name='section_add_list_view'),
    path('add_section/<int:pk>/', ParaleloDetailView.as_view(), name='section_add_detail_view'),

    # PARA LA VISTA DE CALIFICACIONES
    path('add_grades/', GradesListView.as_view(), name='grades_add_list_view'),
    path('add_grades/<int:pk>/', GradesDetailView.as_view(), name='grades_add_detail_view'),

    # PARA LA VISTA DE MATERIAS
    path('add_courses/', CourseListView.as_view(), name='course_add_list_view'),
    path('add_courses/<int:pk>/', CourseDetailView.as_view(), name='course_add_detail_view'),

    # PARA LA VISTA DE ESTUDIANTE REALIZA TEST
    path('add_student_test/', StudentTestListView.as_view(), name='student_test_add_list_view'),
    path('add_student_test/<int:pk>/', StudentTestDetailView.as_view(), name='student_test_add_detail_view'),
    path('delete/<int:estudiante_id>/', StudentTestDeleteByStudentView.as_view(), name='delete_student_test_add_detail_view'),

    # PARA LA VISTA CARRERAS
    path('majors/', MajorsListView.as_view(), name='majors_list_view'),
    path('majors/<int:pk>/', MajorsDetailView.as_view(), name='majors_detail_view'),
    path('delete_majors/<int:estudiante_id>/', MajorsDeleteByStudentView.as_view(), name='delete_majors_detail_view'),

    
    # PARA LA VISTA DEL TOKEN
    path('token/', TokenListView.as_view(), name='token_list_view'),
    path('token/<int:pk>/', TokenDetailView.as_view(), name='token_detail_view'),
    
    #PARA LA VISTA DEL PREDICT PARA SACAR LAS CARRERAS
    path('predict/<int:estudiante_id>/', PredecirCarreraView.as_view(), name='predecir-por-estudiante'),

    # PARA LA VISTA DE APTITUDES
    path('aptitudes/', AptitudesListView.as_view(), name='aptitudes_list_view'),
    path('aptitudes/<int:pk>/', AptitudesDetailView.as_view(), name='aptitudes_detail_view'),
    

 # PARA LA VISTA DEL PROXY
    path('proxy/', proxy_convocatorias, name='proxy_view'),
    


]
