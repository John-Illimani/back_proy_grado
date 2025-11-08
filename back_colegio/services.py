from django.db import transaction
import os
import requests
import joblib
import pandas as pd
from pathlib import Path
from django.conf import settings
import gdown

# Importaciones de tus modelos de Django
from .models.tests_model import OpcionRespuesta
from .models.student_model import Estudiante
from .models.majors_model import Carreras
from .models.aptitudes_model import Aptitudes  # ✅ se importa el modelo de aptitudes

# =============================================================================
# 0. FUNCIÓN PARA DESCARGAR LOS MODELOS DESDE GOOGLE DRIVE
# =============================================================================

# Helper: descarga un archivo de Google Drive dado su file_id
def descargar_desde_drive(file_id: str, destino: Path):
    """
    Descarga un archivo público de Google Drive y lo guarda en `destino`
    usando gdown (maneja archivos grandes automáticamente).
    El archivo debe estar con permiso 'Cualquiera con el enlace'.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)

    # URL estándar para gdown
    url = f"https://drive.google.com/uc?id={file_id}"

    print(f"⬇️ Descargando {destino.name} desde Google Drive con gdown...")
    # quiet=False para que muestre progreso en los logs de Render
    gdown.download(url, str(destino), quiet=False)

    # Verificación rápida de que no se haya guardado HTML por algún motivo raro
    with open(destino, "rb") as f:
        head = f.read(200).lower()
        if b"<html" in head or b"<!doctype html" in head:
            raise RuntimeError(
                f"ERROR: Google Drive devolvió HTML en lugar del archivo binario "
                f"para {destino.name}. Revisa el ID de archivo y los permisos "
                f"(tiene que estar en 'Cualquiera con el enlace')."
            )

    print(f"✅ Descarga completada: {destino}")






def asegurar_modelos_descargados():
    """
    Verifica si los .pkl existen en disco.
    Si falta alguno, lo descarga desde Google Drive usando los IDs de archivo.
    Los IDs se leen de variables de entorno (Render → Environment).
    """
    # Carpeta local donde guardaremos los modelos
    models_dir = Path(settings.BASE_DIR) / "back_colegio" / "random forest"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Mapeo nombre_archivo -> ID de Google Drive (variables de entorno)
    # Mapeo nombre_archivo -> ID de Google Drive (IDs "hardcodeados" para prueba)
    # Mapeo nombre_archivo -> ID de Google Drive (IDs "hardcodeados" para prueba)
    archivos_drive = {
        "modelo_areas_elite.pkl": "1fJMLzbCDNQABU24h0oIdJgqmvNfFqyR-",
        "modelo_carreras_elite.pkl": "1k6tPzvDJtWhoaar2wnGDUMBV_Q0GqNa0",
        "codificadores_features.pkl": "1D0x-9eIi5ujuZn_7esvdUV6qF8_ODEAL",
        "codificador_objetivo_area.pkl": "1gTTIqcdQk6Oxxbf5TIWo_ay_OisbryZC",
        "codificador_objetivo_carrera.pkl": "1qawqksC29GSWNTcFhzAg4t7a-Kgvx8FS",
    }
    for nombre_archivo, file_id in archivos_drive.items():
        ruta_archivo = models_dir / nombre_archivo

        if ruta_archivo.exists():
            # Ya está en disco
            continue

        if not file_id:
            raise RuntimeError(
                f"No se encontró la variable de entorno con el ID de Drive "
                f"para {nombre_archivo}. Revisa tu config en Render."
            )

        # Descargar desde Drive
        descargar_desde_drive(file_id, ruta_archivo)

    return models_dir


# =============================================================================
# 1. CARGA DE TODOS LOS COMPONENTES DEL SISTEMA DE IA ÉLITE
#    (se ejecuta al iniciar Django)
# =============================================================================
# Inicializamos variables para evitar NameError si algo falla
MODELS_DIR = None
modelo_areas = None
modelo_carreras = None
feature_encoders = None
le_area = None
le_carrera = None
carreras_bolivia = {}

try:
    # Asegurarse de que todos los modelos estén descargados desde Drive
    MODELS_DIR = asegurar_modelos_descargados()

    # Carga los 2 modelos entrenados
    modelo_areas = joblib.load(MODELS_DIR / "modelo_areas_elite.pkl")
    modelo_carreras = joblib.load(MODELS_DIR / "modelo_carreras_elite.pkl")

    # Carga el codificador para las respuestas (features)
    feature_encoders = joblib.load(MODELS_DIR / "codificadores_features.pkl")

    # Carga los codificadores para los resultados (objetivos)
    le_area = joblib.load(MODELS_DIR / "codificador_objetivo_area.pkl")
    le_carrera = joblib.load(MODELS_DIR / "codificador_objetivo_carrera.pkl")

    # El diccionario de carreras es necesario para el filtrado final de las recomendaciones
    carreras_bolivia = {
        "Ingeniería y Tecnología": ["Ingeniería Civil", "Ingeniería de Sistemas", "Ingeniería Industrial", "Ingeniería Mecánica", "Ingeniería Electrónica", "Ingeniería Química", "Arquitectura", "Ingeniería Petrolera"],
        "Ciencias de la Salud": ["Medicina", "Enfermería", "Odontología", "Bioquímica y Farmacia", "Fisioterapia y Kinesiología", "Nutrición y Dietética"],
        "Ciencias Sociales y Humanidades": ["Derecho", "Psicología", "Comunicación Social", "Trabajo Social", "Sociología", "Ciencias de la Educación", "Historia", "Turismo"],
        "Ciencias Económicas y Financieras": ["Administración de Empresas", "Contaduría Pública", "Ingeniería Comercial", "Economía", "Ingeniería Financiera"],
        "Ciencias Puras y Naturales": ["Biología", "Física", "Química", "Matemáticas", "Informática", "Ingeniería Ambiental"],
        "Artes y Diseño": ["Artes Plásticas", "Música", "Diseño Gráfico y Comunicación Visual", "Diseño de Modas"],
        "Ciencias Agrícolas": ["Ingeniería Agronómica", "Medicina Veterinaria y Zootecnia"],
        "Fuerzas de Seguridad y Defensa": ["Colegio Militar del Ejército", "Colegio Militar de Aviación", "Escuela Naval Militar", "Academia Nacional de Policías"]
    }

    print("✅ Servicios de IA: Pipeline Élite y todos los componentes cargados correctamente.")

except Exception as e:
    # Capturamos cualquier error (RuntimeError, KeyError, FileNotFoundError, etc.)
    print(f"❌ ERROR cargando modelos de IA: {e}")
    modelo_areas = None
    modelo_carreras = None
    feature_encoders = None
    le_area = None
    le_carrera = None
    carreras_bolivia = {}

# =============================================================================
# FUNCIÓN PRINCIPAL DEL SERVICIO DE PREDICCIÓN
# =============================================================================

@transaction.atomic
def ejecutar_prediccion_y_guardar(estudiante_id: int):
    """
    Función de servicio final que:
    1. Obtiene las respuestas de un estudiante de la base de datos.
    2. Las procesa a través del pipeline de IA de dos etapas.
    3. Guarda las 3 mejores recomendaciones y las aptitudes en la base de datos.
    4. Devuelve el resultado.
    """
    # Verificación de que todos los modelos estén cargados antes de proceder
    if not all([modelo_areas, modelo_carreras, feature_encoders, le_area, le_carrera]):
        raise RuntimeError("Los modelos de IA del pipeline Élite no están completamente cargados.")

    # --- 1. OBTENCIÓN Y TRANSFORMACIÓN DE DATOS DESDE LA DB ---
    respuestas_qs = OpcionRespuesta.objects.filter(
        estudiante_id=estudiante_id).values('estudiante_id', 'pregunta_id', 'texto')
    
    if not respuestas_qs.exists():
        raise ValueError(f"No se encontraron respuestas para el estudiante {estudiante_id}.")

    respuestas_df_largo = pd.DataFrame.from_records(respuestas_qs)
    
    respuestas_df_ancho = respuestas_df_largo.pivot(
        index='estudiante_id', columns='pregunta_id', values='texto')
    respuestas_df_ancho.reset_index(drop=True, inplace=True)

    columnas_esperadas = [f'pregunta_{i}' for i in range(1, 824)]
    respuestas_df_ancho.rename(columns=lambda c: f"pregunta_{c}", inplace=True)
    respuestas_df_ancho = respuestas_df_ancho.reindex(columns=columnas_esperadas, fill_value='SIN_RESPUESTA')

    # ===================================================================
    # --- INICIO DEL BLOQUE DE PREDICCIÓN AVANZADO (PIPELINE ÉLITE) ---
    # ===================================================================
    
    # PASO A: Codificar respuestas de texto a números
    for col, le in feature_encoders.items():
        if col in respuestas_df_ancho.columns:
            valor = respuestas_df_ancho.loc[0, col]
            if valor in le.classes_:
                respuestas_df_ancho.loc[0, col] = le.transform([valor])[0]
            else:
                respuestas_df_ancho.loc[0, col] = -1  # Valor por defecto

    respuestas_df_ancho = respuestas_df_ancho.apply(pd.to_numeric, errors='coerce').fillna(0)

    # PASO B: Aplicar Ingeniería de Características para obtener "scores de aptitud"
    map_aptitud_a_preguntas = {
        'verbal': list(range(418, 465)) + list(range(568, 594)),
        'calculo': list(range(465, 505)),
        'logica_abstracta': list(range(505, 553)) + list(range(794, 824)),
        'mecanico': list(range(553, 568)) + list(range(794, 824)),
        'disciplina_organizacion': list(range(99, 183)) + list(range(232, 255)),
        'liderazgo_social': list(range(183, 232)) + list(range(255, 417))
    }
    scores_aptitud = pd.DataFrame(index=respuestas_df_ancho.index)
    for aptitud, ids_preguntas in map_aptitud_a_preguntas.items():
        columnas = [f'pregunta_{i}' for i in ids_preguntas]
        scores_aptitud[aptitud] = respuestas_df_ancho[columnas].mean(axis=1)

    # ✅ NUEVO BLOQUE: convertir aptitudes a JSON
    aptitudes_resultado = []
    for aptitud in scores_aptitud.columns:
        valor = float(scores_aptitud.loc[0, aptitud])
        porcentaje = f"{valor * 100:.2f}%"
        aptitudes_resultado.append({
            "aptitud": aptitud.replace("_", " ").capitalize(),
            "porcentaje": porcentaje
        })

    # PASO C: Predecir el área principal con el modelo general
    pred_area_num = modelo_areas.predict(scores_aptitud)
    area_principal = le_area.inverse_transform(pred_area_num)[0]

    # PASO D: Obtener el ranking global de carreras con el modelo especialista
    probabilidades_carreras = modelo_carreras.predict_proba(scores_aptitud)[0]
    ranking_global = sorted(
        zip(le_carrera.classes_, probabilidades_carreras),
        key=lambda item: item[1],
        reverse=True
    )

    # PASO E: Filtrar el ranking para mostrar solo carreras del área principal
    carreras_del_area_predicha = carreras_bolivia.get(area_principal, [])
    recomendaciones_filtradas = [
        (carrera, prob) for carrera, prob in ranking_global if carrera in carreras_del_area_predicha
    ]
    
    # PASO F: Lógica de Fallback para garantizar siempre 3 recomendaciones
    recomendaciones_finales = recomendaciones_filtradas[:3]
    if len(recomendaciones_finales) < 3:
        carreras_ya_incluidas = [carrera for carrera, prob in recomendaciones_finales]
        for carrera, prob in ranking_global:
            if len(recomendaciones_finales) >= 3:
                break
            if carrera not in carreras_ya_incluidas:
                recomendaciones_finales.append((carrera, prob))

    # ===================================================================
    # --- FIN DEL BLOQUE DE PREDICCIÓN ---
    # ===================================================================

    # Formatear el resultado
    top_3_recomendaciones = []
    for carrera, prob in recomendaciones_finales:
        top_3_recomendaciones.append({
            "carrera": carrera,
            "probabilidad": f"{prob:.2%}"
        })

    # --- 3. GUARDADO EN BASE DE DATOS ---
    try:
        estudiante_obj = Estudiante.objects.get(pk=estudiante_id)
        
        obj, created = Carreras.objects.update_or_create(
            estudiante=estudiante_obj,
            defaults={'carreras': top_3_recomendaciones}
        )
        print(f"Recomendación {'creada' if created else 'actualizada'} para el estudiante {estudiante_id}.")

        # ✅ GUARDADO DE APTITUDES EN JSON
        Aptitudes.objects.update_or_create(
            estudiante=estudiante_obj,
            defaults={'aptitudes': aptitudes_resultado}
        )
        print(f"Aptitudes guardadas para el estudiante {estudiante_id}.")

    except Estudiante.DoesNotExist:
        raise ValueError(f"El estudiante con ID {estudiante_id} no existe.")

    return top_3_recomendaciones
