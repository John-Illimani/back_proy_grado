import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 0. RUTAS Y CARGA DE MODELOS / ENCODERS
# ============================================================

# Este script debe estar en la misma carpeta que los .pkl
MODELS_DIR = Path(__file__).resolve().parent

modelo_areas = joblib.load(MODELS_DIR / "modelo_areas_elite.pkl")
modelos_carrera_por_area = joblib.load(MODELS_DIR / "modelo_carreras_elite.pkl")
feature_encoders = joblib.load(MODELS_DIR / "codificadores_features.pkl")
le_area = joblib.load(MODELS_DIR / "codificador_objetivo_area.pkl")
le_carrera_global = joblib.load(MODELS_DIR / "codificador_objetivo_carrera.pkl")

print("✅ Modelos y encoders cargados correctamente.\n")

# ============================================================
# 1. DEFINICIÓN DE CARRERAS, PERFILES Y CONFIGURACIÓN DEL TEST
#    (resumen de tu generador original)
# ============================================================

carreras_bolivia = {
    "Ingeniería y Tecnología": [
        "Ingeniería Civil", "Ingeniería de Sistemas", "Ingeniería Industrial",
        "Ingeniería Mecánica", "Ingeniería Electrónica", "Ingeniería Química",
        "Arquitectura", "Ingeniería Petrolera"
    ],
    "Ciencias de la Salud": [
        "Medicina", "Enfermería", "Odontología", "Bioquímica y Farmacia",
        "Fisioterapia y Kinesiología", "Nutrición y Dietética"
    ],
    "Ciencias Sociales y Humanidades": [
        "Derecho", "Psicología", "Comunicación Social", "Trabajo Social",
        "Sociología", "Ciencias de la Educación", "Historia", "Turismo"
    ],
    "Ciencias Económicas y Financieras": [
        "Administración de Empresas", "Contaduría Pública", "Ingeniería Comercial",
        "Economía", "Ingeniería Financiera"
    ],
    "Ciencias Puras y Naturales": [
        "Biología", "Física", "Química", "Matemáticas", "Informática",
        "Ingeniería Ambiental"
    ],
    "Artes y Diseño": [
        "Artes Plásticas", "Música",
        "Diseño Gráfico y Comunicación Visual", "Diseño de Modas"
    ],
    "Ciencias Agrícolas": [
        "Ingeniería Agronómica", "Medicina Veterinaria y Zootecnia"
    ],
    "Fuerzas de Seguridad y Defensa": [
        "Colegio Militar del Ejército", "Colegio Militar de Aviación",
        "Escuela Naval Militar", "Academia Nacional de Policías"
    ]
}

mapa_carrera_a_area = {
    carrera: area
    for area, carreras in carreras_bolivia.items()
    for carrera in carreras
}

# Perfiles base hiper-especializados
perfiles_carrera = {
    "Ingeniería de Sistemas": {'calculo': 0.95, 'logica_abstracta': 0.98, 'mecanico': 0.4, 'verbal': 0.5, 'organizacion': 0.8, 'liderazgo': 0.5,
                               'social_empatia': 0.2, 'artistico_creatividad': 0.4, 'disciplina': 0.7, 'investigacion_cientifica': 0.8,
                               'salud_servicio': 0.1, 'persuasion_negocios': 0.4},
    "Arquitectura": {'calculo': 0.7, 'logica_abstracta': 0.8, 'mecanico': 0.6, 'verbal': 0.6, 'organizacion': 0.7, 'liderazgo': 0.6,
                     'social_empatia': 0.5, 'artistico_creatividad': 0.98, 'disciplina': 0.6, 'investigacion_cientifica': 0.3, 'salud_servicio': 0.1, 'persuasion_negocios': 0.5},
    "Medicina": {'calculo': 0.6, 'logica_abstracta': 0.7, 'mecanico': 0.2, 'verbal': 0.8, 'organizacion': 0.85, 'liderazgo': 0.6,
                 'social_empatia': 0.9, 'artistico_creatividad': 0.1, 'disciplina': 0.95, 'investigacion_cientifica': 0.9, 'salud_servicio': 0.98, 'persuasion_negocios': 0.2},
    "Derecho": {'calculo': 0.3, 'logica_abstracta': 0.6, 'mecanico': 0.1, 'verbal': 0.98, 'organizacion': 0.8, 'liderazgo': 0.9,
                'social_empatia': 0.8, 'artistico_creatividad': 0.2, 'disciplina': 0.8, 'investigacion_cientifica': 0.3, 'salud_servicio': 0.1, 'persuasion_negocios': 0.95},
    "Administración de Empresas": {'calculo': 0.6, 'logica_abstracta': 0.5, 'mecanico': 0.1, 'verbal': 0.8, 'organizacion': 0.95,
                                   'liderazgo': 0.95, 'social_empatia': 0.7, 'artistico_creatividad': 0.3, 'disciplina': 0.7, 'investigacion_cientifica': 0.2, 'salud_servicio': 0.1,
                                   'persuasion_negocios': 0.98},
    "Biología": {'calculo': 0.7, 'logica_abstracta': 0.75, 'mecanico': 0.3, 'verbal': 0.7, 'organizacion': 0.7, 'liderazgo': 0.4,
                 'social_empatia': 0.6, 'artistico_creatividad': 0.3, 'disciplina': 0.8, 'investigacion_cientifica': 0.98, 'salud_servicio': 0.5, 'persuasion_negocios': 0.2},
    "Artes Plásticas": {'calculo': 0.2, 'logica_abstracta': 0.6, 'mecanico': 0.3, 'verbal': 0.6, 'organizacion': 0.3, 'liderazgo': 0.3,
                        'social_empatia': 0.7, 'artistico_creatividad': 0.99, 'disciplina': 0.3, 'investigacion_cientifica': 0.1, 'salud_servicio': 0.1, 'persuasion_negocios': 0.4},
    "Ingeniería Agronómica": {'calculo': 0.6, 'logica_abstracta': 0.6, 'mecanico': 0.6, 'verbal': 0.5, 'organizacion': 0.7, 'liderazgo': 0.5,
                              'social_empatia': 0.3, 'artistico_creatividad': 0.1, 'disciplina': 0.7, 'investigacion_cientifica': 0.9, 'salud_servicio': 0.3, 'persuasion_negocios': 0.3},
    "Colegio Militar del Ejército": {'calculo': 0.7, 'logica_abstracta': 0.7, 'mecanico': 0.8, 'verbal': 0.6, 'organizacion': 0.9, 'liderazgo': 0.95,
                                     'social_empatia': 0.4, 'artistico_creatividad': 0.1, 'disciplina': 0.99, 'investigacion_cientifica': 0.2, 'salud_servicio': 0.3, 'persuasion_negocios': 0.2}
}

# Completar perfiles faltantes por área
for area, carreras in carreras_bolivia.items():
    for carrera in carreras:
        if carrera not in perfiles_carrera:
            if area == "Ingeniería y Tecnología":
                perfiles_carrera[carrera] = perfiles_carrera["Ingeniería de Sistemas"]
            elif area == "Ciencias de la Salud":
                perfiles_carrera[carrera] = perfiles_carrera["Medicina"]
            elif area == "Ciencias Sociales y Humanidades":
                perfiles_carrera[carrera] = perfiles_carrera["Derecho"]
            elif area == "Ciencias Económicas y Financieras":
                perfiles_carrera[carrera] = perfiles_carrera["Administración de Empresas"]
            elif area == "Ciencias Puras y Naturales":
                perfiles_carrera[carrera] = perfiles_carrera["Biología"]
            elif area == "Artes y Diseño":
                perfiles_carrera[carrera] = perfiles_carrera["Artes Plásticas"]
            elif area == "Ciencias Agrícolas":
                perfiles_carrera[carrera] = perfiles_carrera["Ingeniería Agronómica"]
            elif area == "Fuerzas de Seguridad y Defensa":
                perfiles_carrera[carrera] = perfiles_carrera["Colegio Militar del Ejército"]

# Mapeo pregunta → aptitud (como en tu generador)
map_pregunta_aptitud = {
    (1, 98): 'variado', (99, 157): ['disciplina', 'social_empatia'],
    (158, 182): ['disciplina', 'organizacion'], (183, 231): ['liderazgo', 'social_empatia'],
    (232, 254): ['disciplina', 'social_empatia'], (255, 417): ['social_empatia', 'liderazgo', 'artistico_creatividad'],
    (418, 464): 'verbal', (465, 504): 'calculo', (505, 552): 'logica_abstracta',
    (553, 567): 'mecanico', (568, 593): 'verbal', (594, 693): ['logica_abstracta', 'disciplina'],
    (694, 793): ['logica_abstracta', 'disciplina'],
    (794, 823): ['logica_abstracta', 'mecanico', 'artistico_creatividad']
}

config_tests = [
    {"id_inicio": 1, "id_fin": 98, "tipo_respuesta": "SI/NO"},
    {"id_inicio": 99, "id_fin": 157, "tipo_respuesta": "V/F"},
    {"id_inicio": 158, "id_fin": 182, "tipo_respuesta": "SI/NO"},
    {"id_inicio": 183, "id_fin": 231, "tipo_respuesta": "SI/MEDIO/NO"},
    {"id_inicio": 232, "id_fin": 254, "tipo_respuesta": "V/F"},
    {"id_inicio": 255, "id_fin": 417, "tipo_respuesta": "1-5"},
    {"id_inicio": 418, "id_fin": 464, "tipo_respuesta": "1/0"},
    {"id_inicio": 465, "id_fin": 504, "tipo_respuesta": "1/0"},
    {"id_inicio": 505, "id_fin": 552, "tipo_respuesta": "1/0"},
    {"id_inicio": 553, "id_fin": 567, "tipo_respuesta": "1/0"},
    {"id_inicio": 568, "id_fin": 593, "tipo_respuesta": "1/0"},
    {"id_inicio": 594, "id_fin": 693, "tipo_respuesta": "1/0"},
    {"id_inicio": 694, "id_fin": 793, "tipo_respuesta": "1/0"},
    {"id_inicio": 794, "id_fin": 823, "tipo_respuesta": "1/0"}
]

# ============================================================
# 2. GENERAR RESPUESTAS SIMULADAS PARA UNA CARRERA
# ============================================================

def generar_respuestas_estudiante_avanzado(carrera):
    perfil_base_carrera = perfiles_carrera[carrera]
    perfil_individual = {}
    for aptitud, score_base in perfil_base_carrera.items():
        score_individual = np.random.normal(score_base, 0.1)
        perfil_individual[aptitud] = np.clip(score_individual, 0.05, 0.95)

    respuestas = {}
    for test in config_tests:
        for q_id in range(test["id_inicio"], test["id_fin"] + 1):
            aptitud_evaluada = next(
                (apt for rango, apt in map_pregunta_aptitud.items()
                 if rango[0] <= q_id <= rango[1]),
                None
            )

            prob_positiva = 0.5
            if aptitud_evaluada:
                if isinstance(aptitud_evaluada, list):
                    prob_positiva = np.mean(
                        [perfil_individual[a] for a in aptitud_evaluada]
                    )
                elif aptitud_evaluada != 'variado':
                    prob_positiva = perfil_individual[aptitud_evaluada]
                else:
                    prob_positiva = np.mean([
                        perfil_individual['verbal'],
                        perfil_individual['social_empatia'],
                        perfil_individual['organizacion'],
                        perfil_individual['calculo']
                    ])

            if 418 <= q_id <= 823:
                prob_positiva = np.clip(prob_positiva * 1.05, 0.05, 0.95)

            prob_final = np.clip(np.random.normal(prob_positiva, 0.1), 0, 1)
            tipo_respuesta = test['tipo_respuesta']

            if tipo_respuesta in ["SI/NO", "V/F", "1/0"]:
                opciones = {
                    'SI/NO': ['SI', 'NO'],
                    'V/F': ['V', 'F'],
                    '1/0': [1, 0]
                }[tipo_respuesta]
                respuesta = np.random.choice(opciones, p=[prob_final, 1 - prob_final])
            elif tipo_respuesta == "SI/MEDIO/NO":
                p_si, p_no = prob_final * 0.8, (1 - prob_final) * 0.8
                p_medio = max(0, 1 - p_si - p_no)
                norm = p_si + p_no + p_medio
                respuesta = np.random.choice(
                    ['SI', 'TERMINO MEDIO', 'NO'],
                    p=[p_si/norm, p_medio/norm, p_no/norm]
                )
            elif tipo_respuesta == "1-5":
                centro = 1 + prob_final * 4
                pesos = np.exp(-((np.arange(1, 6) - centro) ** 2) / 4)
                respuesta = np.random.choice(
                    [1, 2, 3, 4, 5], p=pesos / pesos.sum()
                )
            else:
                respuesta = 0

            respuestas[f'pregunta_{q_id}'] = respuesta

    return respuestas

# ============================================================
# 3. PIPELINE DE PREDICCIÓN (ÁREA + CARRERA POR ÁREA)
# ============================================================

map_aptitud_a_preguntas_feat = {
    'verbal': list(range(418, 465)) + list(range(568, 594)),
    'calculo': list(range(465, 505)),
    'logica_abstracta': list(range(505, 553)) + list(range(794, 824)),
    'mecanico': list(range(553, 568)) + list(range(794, 824)),
    'disciplina_organizacion': list(range(99, 183)) + list(range(232, 255)),
    'liderazgo_social': list(range(183, 232)) + list(range(255, 418))
}

def preparar_features_para_modelo(respuestas_dict):
    df = pd.DataFrame([respuestas_dict])

    # Asegurar todas las preguntas
    columnas_esperadas = [f'pregunta_{i}' for i in range(1, 824)]
    df = df.reindex(columns=columnas_esperadas, fill_value='SIN_RESPUESTA')

    # Codificar con los mismos encoders del entrenamiento
    for col, le in feature_encoders.items():
        if col in df.columns:
            valor = df.loc[0, col]
            if valor in le.classes_:
                df.loc[0, col] = le.transform([valor])[0]
            else:
                df.loc[0, col] = -1

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Construir aptitudes (mean, std, max)
    X_aptitudes = pd.DataFrame(index=df.index)
    for aptitud, ids_preguntas in map_aptitud_a_preguntas_feat.items():
        cols = [f'pregunta_{i}' for i in ids_preguntas if f'pregunta_{i}' in df.columns]
        if len(cols) == 0:
            X_aptitudes[f'{aptitud}_mean'] = 0.0
            X_aptitudes[f'{aptitud}_std'] = 0.0
            X_aptitudes[f'{aptitud}_max'] = 0.0
        else:
            X_aptitudes[f'{aptitud}_mean'] = df[cols].mean(axis=1)
            X_aptitudes[f'{aptitud}_std'] = df[cols].std(axis=1)
            X_aptitudes[f'{aptitud}_max'] = df[cols].max(axis=1)

    X_full = pd.concat(
        [df.reset_index(drop=True), X_aptitudes.reset_index(drop=True)],
        axis=1
    )
    return X_aptitudes, X_full

def predecir_area_y_carrera(respuestas_dict):
    X_aptitudes, X_full = preparar_features_para_modelo(respuestas_dict)

    # 1) Área
    pred_area_num = modelo_areas.predict(X_aptitudes)[0]
    area_pred = le_area.inverse_transform([pred_area_num])[0]

    # 2) Modelo de carreras para esa área
    info_area = modelos_carrera_por_area[area_pred]
    modelo = info_area["modelo"]
    encoder_carrera_area = info_area["encoder"]
    columnas_modelo = info_area["columnas"]

    X_input = X_full[columnas_modelo]

    proba = modelo.predict_proba(X_input)[0]
    idx_clases = np.arange(len(proba))
    carreras_area = encoder_carrera_area.inverse_transform(idx_clases)

    ranking = sorted(zip(carreras_area, proba), key=lambda x: x[1], reverse=True)
    carrera_top1 = ranking[0][0]
    top3 = ranking[:3]

    return area_pred, carrera_top1, top3, ranking

# ============================================================
# 4. PRUEBA ESPECÍFICA: INGENIERÍA DE SISTEMAS
# ============================================================

def probar_ingenieria_sistemas(n=10):
    carrera_real = "Ingeniería de Sistemas"
    area_real = mapa_carrera_a_area[carrera_real]

    print("\n==============================")
    print(f" PRUEBAS PARA PERFIL REAL: {carrera_real}")
    print(f" Área real: {area_real}")
    print(f" Generando {n} estudiantes simulados...\n")

    aciertos_top1 = 0
    aciertos_top3 = 0

    for i in range(n):
        respuestas = generar_respuestas_estudiante_avanzado(carrera_real)
        area_pred, carrera_top1, top3, ranking = predecir_area_y_carrera(respuestas)

        en_top3 = any(c == carrera_real for c, _ in top3)
        if carrera_top1 == carrera_real:
            aciertos_top1 += 1
        if en_top3:
            aciertos_top3 += 1

        print(f" Estudiante #{i+1}:")
        print(f"   Área predicha: {area_pred}")
        print(f"   Carrera TOP-1: {carrera_top1}")
        print("   TOP-3:")
        for c_nom, p in top3:
            print(f"     - {c_nom}: {p:.2%}")
        print(f"   ¿Ingeniería de Sistemas en TOP-3? {'✅' if en_top3 else '❌'}")
        print("")

    print("Resumen para Ingeniería de Sistemas:")
    print(f"  Accuracy TOP-1 (simulado): {aciertos_top1}/{n} = {aciertos_top1/n:.2%}")
    print(f"  Accuracy TOP-3 (simulado): {aciertos_top3}/{n} = {aciertos_top3/n:.2%}")
    print("==============================\n")

def main():
    probar_ingenieria_sistemas(n=10)

if __name__ == "__main__":
    main()
