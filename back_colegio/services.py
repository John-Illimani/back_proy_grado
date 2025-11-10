from django.db import transaction
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings

# Importaciones de tus modelos de Django
from .models.tests_model import OpcionRespuesta
from .models.student_model import Estudiante
from .models.majors_model import Carreras
from .models.aptitudes_model import Aptitudes  # ✅ se importa el modelo de aptitudes

# =============================================================================
# 1. CARGA DE TODOS LOS COMPONENTES DEL SISTEMA DE IA ÉLITE
# Este bloque se ejecuta una sola vez cuando Django inicia el servidor.
# =============================================================================
try:
    # Define la ruta base donde guardaste los archivos .pkl
    MODELS_DIR = Path(settings.BASE_DIR) / "back_colegio" / "random forest"
    
    # Carga los 2 modelos entrenados
    modelo_areas = joblib.load(MODELS_DIR / "modelo_areas_elite.pkl")
    modelo_carreras = joblib.load(MODELS_DIR / "modelo_carreras_elite.pkl")  # dict por área
    
    # Carga el codificador para las respuestas (features)
    feature_encoders = joblib.load(MODELS_DIR / "codificadores_features.pkl")
    
    # Carga los codificadores para los resultados (objetivos)
    le_area = joblib.load(MODELS_DIR / "codificador_objetivo_area.pkl")
    le_carrera = joblib.load(MODELS_DIR / "codificador_objetivo_carrera.pkl")  # se mantiene aunque no lo usemos aquí
    
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

except FileNotFoundError as e:
    print(f"❌ ADVERTENCIA: No se encontró un archivo del modelo de IA: {e.filename}")
    modelo_areas = None
    modelo_carreras = None
    feature_encoders = None
    le_area = None
    le_carrera = None

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
                # Valor no visto durante el entrenamiento
                respuestas_df_ancho.loc[0, col] = -1

    respuestas_df_ancho = respuestas_df_ancho.apply(pd.to_numeric, errors='coerce').fillna(0)

    # PASO B: Ingeniería de Características EXACTAMENTE como en entrenamiento
    map_aptitud_a_preguntas = {
        'verbal': list(range(418, 465)) + list(range(568, 594)),
        'calculo': list(range(465, 505)),
        'logica_abstracta': list(range(505, 553)) + list(range(794, 824)),
        'mecanico': list(range(553, 568)) + list(range(794, 824)),
        'disciplina_organizacion': list(range(99, 183)) + list(range(232, 255)),
        'liderazgo_social': list(range(183, 232)) + list(range(255, 418)),  # 418 para incluir hasta 417
    }

    # X_aptitudes tendrá columnas: verbal_mean, verbal_std, verbal_max, etc.
    X_aptitudes = pd.DataFrame(index=respuestas_df_ancho.index)
    for aptitud, ids_preguntas in map_aptitud_a_preguntas.items():
        columnas = [f'pregunta_{i}' for i in ids_preguntas if f'pregunta_{i}' in respuestas_df_ancho.columns]
        if len(columnas) == 0:
            X_aptitudes[f'{aptitud}_mean'] = 0.0
            X_aptitudes[f'{aptitud}_std'] = 0.0
            X_aptitudes[f'{aptitud}_max'] = 0.0
        else:
            X_aptitudes[f'{aptitud}_mean'] = respuestas_df_ancho[columnas].mean(axis=1)
            X_aptitudes[f'{aptitud}_std'] = respuestas_df_ancho[columnas].std(axis=1)
            X_aptitudes[f'{aptitud}_max'] = respuestas_df_ancho[columnas].max(axis=1)

    # ✅ NUEVO BLOQUE: convertir aptitudes a JSON usando el "mean"
    aptitudes_resultado = []
    for aptitud in map_aptitud_a_preguntas.keys():
        col_mean = f'{aptitud}_mean'
        valor = float(X_aptitudes.loc[0, col_mean]) if col_mean in X_aptitudes.columns else 0.0
        porcentaje = f"{valor * 100:.2f}%"
        aptitudes_resultado.append({
            "aptitud": aptitud.replace("_", " ").capitalize(),
            "porcentaje": porcentaje
        })

    # PASO C: Predecir el área principal con el modelo general (usa X_aptitudes)
    pred_area_num = modelo_areas.predict(X_aptitudes)[0]
    area_principal = le_area.inverse_transform([pred_area_num])[0]

    # PASO D: Construir el feature set COMPLETO para carreras (preguntas + aptitudes)
    X_full = pd.concat(
        [respuestas_df_ancho.reset_index(drop=True),
         X_aptitudes.reset_index(drop=True)],
        axis=1
    )

    # Seleccionar el modelo especialista de carreras para el área predicha
    info_area = modelo_carreras.get(area_principal)
    if info_area is None:
        raise ValueError(f"No se encontró modelo de carreras para el área predicha: {area_principal}")

    modelo_carrera_area = info_area["modelo"]
    encoder_carrera_area = info_area["encoder"]
    columnas_modelo = info_area.get("columnas", X_full.columns.tolist())

    # Asegurar que las columnas coinciden con las usadas en entrenamiento
    X_input_area = X_full[columnas_modelo]

    # PASO E: Obtener el ranking de carreras dentro del área usando el modelo especialista
    probabilidades_carreras = modelo_carrera_area.predict_proba(X_input_area)[0]
    clases_idx = np.arange(len(probabilidades_carreras))
    carreras_area = encoder_carrera_area.inverse_transform(clases_idx)

    ranking_global = sorted(
        zip(carreras_area, probabilidades_carreras),
        key=lambda item: item[1],
        reverse=True
    )

    # PASO F: Filtrar el ranking para mostrar solo carreras del área principal (por seguridad, aunque ya son del área)
    carreras_del_area_predicha = carreras_bolivia.get(area_principal, [])
    recomendaciones_filtradas = [
        (carrera, prob) for carrera, prob in ranking_global if carrera in carreras_del_area_predicha
    ]

    # PASO G: Lógica de Fallback para garantizar siempre hasta 3 recomendaciones
    recomendaciones_finales = recomendaciones_filtradas[:3]

    # Si el área tiene menos de 3 carreras (p.ej. Ciencias Agrícolas tiene 2),
    # intentamos completar con otras carreras del área (si existieran) o dejamos las que haya.
    if len(recomendaciones_finales) < 3:
        carreras_ya_incluidas = {carrera for carrera, _ in recomendaciones_finales}
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
