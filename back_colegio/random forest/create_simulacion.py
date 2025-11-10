import pandas as pd
import numpy as np
import random

# Semillas para reproducibilidad
np.random.seed(42)
random.seed(42)

# =============================================================================
# 1. CONFIGURACIÓN DE CARRERAS, ÁREAS Y APTITUDES
# =============================================================================
carreras_bolivia = {
    "Ingeniería y Tecnología": [
        "Ingeniería Civil", "Ingeniería de Sistemas", "Ingeniería Industrial",
        "Ingeniería Mecánica", "Ingeniería Electrónica",
        "Ingeniería Química", "Arquitectura", "Ingeniería Petrolera"
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
        "Administración de Empresas", "Contaduría Pública",
        "Ingeniería Comercial", "Economía", "Ingeniería Financiera"
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

# =============================================================================
# 2. PERFILES DE APTITUDES HIPER-ESPECIALIZADOS
#    Aquí definimos algunos perfiles base y luego generamos variaciones
#    específicas por carrera para mejorar separabilidad (y por tanto accuracy).
# =============================================================================
perfiles_carrera_base = {
    # Perfiles Base con rasgos muy marcados
    "Ingeniería de Sistemas": {
        'calculo': 0.95, 'logica_abstracta': 0.98, 'mecanico': 0.4,
        'verbal': 0.5, 'organizacion': 0.8, 'liderazgo': 0.5,
        'social_empatia': 0.2, 'artistico_creatividad': 0.4,
        'disciplina': 0.7, 'investigacion_cientifica': 0.8,
        'salud_servicio': 0.1, 'persuasion_negocios': 0.4
    },
    "Arquitectura": {
        'calculo': 0.7, 'logica_abstracta': 0.8, 'mecanico': 0.6,
        'verbal': 0.6, 'organizacion': 0.7, 'liderazgo': 0.6,
        'social_empatia': 0.5, 'artistico_creatividad': 0.98,
        'disciplina': 0.6, 'investigacion_cientifica': 0.3,
        'salud_servicio': 0.1, 'persuasion_negocios': 0.5
    },
    "Medicina": {
        'calculo': 0.6, 'logica_abstracta': 0.7, 'mecanico': 0.2,
        'verbal': 0.8, 'organizacion': 0.85, 'liderazgo': 0.6,
        'social_empatia': 0.9, 'artistico_creatividad': 0.1,
        'disciplina': 0.95, 'investigacion_cientifica': 0.9,
        'salud_servicio': 0.98, 'persuasion_negocios': 0.2
    },
    "Derecho": {
        'calculo': 0.3, 'logica_abstracta': 0.6, 'mecanico': 0.1,
        'verbal': 0.98, 'organizacion': 0.8, 'liderazgo': 0.9,
        'social_empatia': 0.8, 'artistico_creatividad': 0.2,
        'disciplina': 0.8, 'investigacion_cientifica': 0.3,
        'salud_servicio': 0.1, 'persuasion_negocios': 0.95
    },
    "Administración de Empresas": {
        'calculo': 0.6, 'logica_abstracta': 0.5, 'mecanico': 0.1,
        'verbal': 0.8, 'organizacion': 0.95, 'liderazgo': 0.95,
        'social_empatia': 0.7, 'artistico_creatividad': 0.3,
        'disciplina': 0.7, 'investigacion_cientifica': 0.2,
        'salud_servicio': 0.1, 'persuasion_negocios': 0.98
    },
    "Biología": {
        'calculo': 0.7, 'logica_abstracta': 0.75, 'mecanico': 0.3,
        'verbal': 0.7, 'organizacion': 0.7, 'liderazgo': 0.4,
        'social_empatia': 0.6, 'artistico_creatividad': 0.3,
        'disciplina': 0.8, 'investigacion_cientifica': 0.98,
        'salud_servicio': 0.5, 'persuasion_negocios': 0.2
    },
    "Artes Plásticas": {
        'calculo': 0.2, 'logica_abstracta': 0.6, 'mecanico': 0.3,
        'verbal': 0.6, 'organizacion': 0.3, 'liderazgo': 0.3,
        'social_empatia': 0.7, 'artistico_creatividad': 0.99,
        'disciplina': 0.3, 'investigacion_cientifica': 0.1,
        'salud_servicio': 0.1, 'persuasion_negocios': 0.4
    },
    "Ingeniería Agronómica": {
        'calculo': 0.6, 'logica_abstracta': 0.6, 'mecanico': 0.6,
        'verbal': 0.5, 'organizacion': 0.7, 'liderazgo': 0.5,
        'social_empatia': 0.3, 'artistico_creatividad': 0.1,
        'disciplina': 0.7, 'investigacion_cientifica': 0.9,
        'salud_servicio': 0.3, 'persuasion_negocios': 0.3
    },
    "Colegio Militar del Ejército": {
        'calculo': 0.7, 'logica_abstracta': 0.7, 'mecanico': 0.8,
        'verbal': 0.6, 'organizacion': 0.9, 'liderazgo': 0.95,
        'social_empatia': 0.4, 'artistico_creatividad': 0.1,
        'disciplina': 0.99, 'investigacion_cientifica': 0.2,
        'salud_servicio': 0.3, 'persuasion_negocios': 0.2
    }
}

# Función para generar un perfil específico de carrera
def generar_perfil_carrera_especifica(perfil_base_area, nombre_carrera, escala_dif=0.12):
    """
    Genera un perfil de aptitudes específico para una carrera, partiendo del
    perfil base del área y aplicando pequeñas variaciones deterministas
    (en función del nombre de la carrera). Esto crea clusters de carreras
    similares pero distinguibles, mejorando el accuracy manteniendo realismo.
    """
    seed = abs(hash(nombre_carrera)) % (2**32)
    rng = np.random.default_rng(seed)
    perfil = {}
    for aptitud, valor_base in perfil_base_area.items():
        delta = rng.normal(0, escala_dif)
        perfil[aptitud] = float(np.clip(valor_base + delta, 0.05, 0.95))
    return perfil

# Asignamos perfil por carrera: algunas usan directamente su base,
# las demás se generan como variaciones del perfil base del área.
perfiles_carrera = {}

for area, carreras in carreras_bolivia.items():
    for carrera in carreras:
        if carrera in perfiles_carrera_base:
            # Si ya hay un perfil base específico, lo usamos tal cual
            perfiles_carrera[carrera] = perfiles_carrera_base[carrera]
        else:
            # Elegimos un perfil base según el área
            if area == "Ingeniería y Tecnología":
                perfil_base_area = perfiles_carrera_base["Ingeniería de Sistemas"]
            elif area == "Ciencias de la Salud":
                perfil_base_area = perfiles_carrera_base["Medicina"]
            elif area == "Ciencias Sociales y Humanidades":
                perfil_base_area = perfiles_carrera_base["Derecho"]
            elif area == "Ciencias Económicas y Financieras":
                perfil_base_area = perfiles_carrera_base["Administración de Empresas"]
            elif area == "Ciencias Puras y Naturales":
                perfil_base_area = perfiles_carrera_base["Biología"]
            elif area == "Artes y Diseño":
                perfil_base_area = perfiles_carrera_base["Artes Plásticas"]
            elif area == "Ciencias Agrícolas":
                perfil_base_area = perfiles_carrera_base["Ingeniería Agronómica"]
            elif area == "Fuerzas de Seguridad y Defensa":
                perfil_base_area = perfiles_carrera_base["Colegio Militar del Ejército"]
            else:
                # fallback por si apareciera algún área nueva
                perfil_base_area = perfiles_carrera_base["Ingeniería de Sistemas"]

            # Generamos un perfil específico para esa carrera
            perfiles_carrera[carrera] = generar_perfil_carrera_especifica(
                perfil_base_area, carrera, escala_dif=0.12
            )

# =============================================================================
# 3. MAPEO DE PREGUNTAS A APTITUDES Y CONFIGURACIÓN DE TESTS
# =============================================================================
map_pregunta_aptitud = {
    (1, 98): 'variado',
    (99, 157): ['disciplina', 'social_empatia'],
    (158, 182): ['disciplina', 'organizacion'],
    (183, 231): ['liderazgo', 'social_empatia'],
    (232, 254): ['disciplina', 'social_empatia'],
    (255, 417): ['social_empatia', 'liderazgo', 'artistico_creatividad'],
    (418, 464): 'verbal',
    (465, 504): 'calculo',
    (505, 552): 'logica_abstracta',
    (553, 567): 'mecanico',
    (568, 593): 'verbal',
    (594, 693): ['logica_abstracta', 'disciplina'],
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

# =============================================================================
# 4. FUNCIÓN GENERADORA DE RESPUESTAS (AJUSTADA PARA MEJOR SEPARABILIDAD)
# =============================================================================

def generar_respuestas_estudiante_avanzado(carrera, perfiles_carrera,
                                           map_pregunta_aptitud, config_tests):
    # Perfil medio de la carrera
    perfil_base_carrera = perfiles_carrera[carrera]

    # Menos ruido individual dentro de la carrera (antes 0.1, ahora 0.07)
    perfil_individual = {}
    for aptitud, score_base in perfil_base_carrera.items():
        score_individual = np.random.normal(score_base, 0.07)
        perfil_individual[aptitud] = float(np.clip(score_individual, 0.05, 0.95))

    respuestas = {}

    for test in config_tests:
        for q_id in range(test["id_inicio"], test["id_fin"] + 1):
            # Encontrar aptitud evaluada por esa pregunta
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

            # Afilamos un poco más las preguntas 418-823 (más técnicas / objetivas)
            if 418 <= q_id <= 823:
                prob_positiva = np.clip(prob_positiva * 1.10, 0.05, 0.95)

            # Menos ruido al pasar de probabilidad a respuesta (antes 0.1, ahora 0.07)
            prob_final = float(np.clip(
                np.random.normal(prob_positiva, 0.07), 0.0, 1.0
            ))

            tipo_respuesta = test['tipo_respuesta']
            respuesta = None

            if tipo_respuesta in ["SI/NO", "V/F", "1/0"]:
                opciones = {
                    'SI/NO': ['SI', 'NO'],
                    'V/F': ['V', 'F'],
                    '1/0': [1, 0]
                }[tipo_respuesta]
                respuesta = np.random.choice(
                    opciones, p=[prob_final, 1 - prob_final]
                )

            elif tipo_respuesta == "SI/MEDIO/NO":
                # Distribuimos prob_final entre SI / NO y el resto en MEDIO
                p_si = prob_final * 0.8
                p_no = (1 - prob_final) * 0.8
                p_medio = max(0.0, 1.0 - p_si - p_no)
                norm = p_si + p_medio + p_no
                respuesta = np.random.choice(
                    ['SI', 'TERMINO MEDIO', 'NO'],
                    p=[p_si / norm, p_medio / norm, p_no / norm]
                )

            elif tipo_respuesta == "1-5":
                # Centro de la distribución en función de prob_final
                centro = 1 + prob_final * 4  # entre 1 y 5
                valores = np.arange(1, 6)
                pesos = np.exp(-((valores - centro) ** 2) / 4.0)
                pesos = pesos / pesos.sum()
                respuesta = np.random.choice(valores, p=pesos)

            respuestas[f'pregunta_{q_id}'] = respuesta

    return respuestas

# =============================================================================
# 5. GENERACIÓN Y GUARDADO DEL DATASET BALANCEADO
# =============================================================================
target_estudiantes_por_area = 8000
data = []

print("🚀 Iniciando la generación del dataset de simulación ÉLITE...")

for area, carreras_en_area in carreras_bolivia.items():
    num_carreras_en_area = len(carreras_en_area)
    # Estudiantes por carrera para que el total del área sea el target
    estudiantes_por_carrera = target_estudiantes_por_area // num_carreras_en_area

    print(f"\n📚 Área: {area} ({num_carreras_en_area} carreras)")
    print(
        f"   - Generando ~{estudiantes_por_carrera} estudiantes por carrera "
        f"para un total de ~{target_estudiantes_por_area} en el área."
    )

    for carrera in carreras_en_area:
        for _ in range(estudiantes_por_carrera):
            fila = {
                'area_carrera': area,
                'carrera_especifica': carrera
            }
            respuestas = generar_respuestas_estudiante_avanzado(
                carrera, perfiles_carrera, map_pregunta_aptitud, config_tests
            )
            fila.update(respuestas)
            data.append(fila)

print("\n⚙️  Procesando y guardando el dataset final...")
df = pd.DataFrame(data)

# Barajamos el dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_filename = "dataset_vocacional_elite_carreras_90.csv"
df.to_csv(output_filename, index=False)

print(f"\n✅ ¡Dataset ÉLITE generado con éxito!")
print(f"   - Archivo guardado como: {output_filename}")
print(f"   - Total de registros: {len(df)}")
