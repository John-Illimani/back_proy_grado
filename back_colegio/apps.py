from django.apps import AppConfig


class Back_colegioConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'back_colegio'
    def ready(self):
        # Importar señales al arrancar la app
        import back_colegio.signals



