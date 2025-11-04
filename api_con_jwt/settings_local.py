from datetime import timedelta

# parametrizacion de contraseñas

SECRET_KEY = 'illimani'
DEBUG = True

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),  #ESTE ES CUANDO EL ANTERIOR SE A CADUCADO
}