import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================
# 1. CARGAR Y PREPARAR EL DATASET ÉLITE
# =========================================
DATASET_FILENAME = "dataset_vocacional_elite_carreras_90.csv"

try:
    df = pd.read_csv(DATASET_FILENAME)
    print(f"✅ Dataset cargado: {DATASET_FILENAME}")
except FileNotFoundError:
    print("❌ Error: Ejecuta primero el script que genera 'dataset_vocacional_elite_carreras_90.csv'")
    exit()

# Features originales (preguntas) y targets
X_raw = df.drop(columns=["area_carrera", "carrera_especifica"])
y_area = df["area_carrera"]
y_carrera = df["carrera_especifica"]

# Codificar respuestas categóricas (features) a números
feature_encoders = {}
for col in X_raw.columns:
    if X_raw[col].dtype == 'object':
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col])
        feature_encoders[col] = le

# =================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# =================================================================
print("⚙️ Generando features de aptitud a partir de las respuestas...")

map_aptitud_a_preguntas = {
    'verbal': list(range(418, 465)) + list(range(568, 594)),
    'calculo': list(range(465, 505)),
    'logica_abstracta': list(range(505, 553)) + list(range(794, 824)),
    'mecanico': list(range(553, 568)) + list(range(794, 824)),
    'disciplina_organizacion': list(range(99, 183)) + list(range(232, 255)),
    'liderazgo_social': list(range(183, 232)) + list(range(255, 418))
}

X_aptitudes = pd.DataFrame(index=X_raw.index)
for aptitud, ids_preguntas in map_aptitud_a_preguntas.items():
    cols = [f'pregunta_{i}' for i in ids_preguntas if f'pregunta_{i}' in X_raw.columns]
    if len(cols) == 0:
        X_aptitudes[f'{aptitud}_mean'] = 0.0
        X_aptitudes[f'{aptitud}_std'] = 0.0
        X_aptitudes[f'{aptitud}_max'] = 0.0
    else:
        X_aptitudes[f'{aptitud}_mean'] = X_raw[cols].mean(axis=1)
        X_aptitudes[f'{aptitud}_std'] = X_raw[cols].std(axis=1)
        X_aptitudes[f'{aptitud}_max'] = X_raw[cols].max(axis=1)

print("✅ Features de aptitud generadas con éxito.")

# =================================================================
# 3. ENTRENAMIENTO DEL MODELO 1: PREDICTOR DE ÁREAS (RandomForest)
# =================================================================
print("\n🚀 Fase 1: Entrenando el modelo de ÁREAS con RandomForest...")

X_train, X_test, y_train_area, y_test_area = train_test_split(
    X_aptitudes, y_area, test_size=0.2, random_state=42, stratify=y_area
)

# Codificador del target de áreas
le_area = LabelEncoder()
y_train_area_enc = le_area.fit_transform(y_train_area)
y_test_area_enc = le_area.transform(y_test_area)

rf_area = RandomForestClassifier(
    n_estimators=200,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

rf_area.fit(X_train, y_train_area_enc)

y_pred_area = rf_area.predict(X_test)
acc_area = accuracy_score(y_test_area_enc, y_pred_area)
print(f"🎯 Accuracy ÁREAS: {acc_area:.4f}")
print("\n📊 Reporte de Clasificación (ÁREAS):\n",
      classification_report(y_test_area_enc, y_pred_area, target_names=le_area.classes_))

# Guardar modelo de áreas y encoders
joblib.dump(rf_area, "modelo_areas_elite.pkl", compress=3)
joblib.dump(feature_encoders, "codificadores_features.pkl", compress=3)
joblib.dump(le_area, "codificador_objetivo_area.pkl", compress=3)
print("✅ Modelo y codificadores de ÁREAS guardados.")

# =================================================================
# 4. ENTRENAMIENTO DEL MODELO 2: CARRERAS POR ÁREA (RandomForest REGULARIZADO)
# =================================================================
print("\n🚀 Fase 2: Entrenando modelos de CARRERAS ESPECÍFICAS por área...")

modelos_carrera_por_area = {}
accuracies_por_area = {}
cv_scores_por_area = {}

# Hiperparámetros REGULARIZADOS para carreras
rf_params_carreras = {
    "n_estimators": 200,
    "max_depth": 10,       # árboles relativamente poco profundos
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": 0.3,   # menos features por split -> más diversidad
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": 42
}

for area in df["area_carrera"].unique():
    print(f"\n📚 Entrenando modelo para el área: {area}")
    df_area = df[df["area_carrera"] == area]

    idx_area = df_area.index

    X_area_raw = X_raw.loc[idx_area]
    X_area_apt = X_aptitudes.loc[idx_area]

    # Features finales para carreras: preguntas + aptitudes agregadas
    X_area_full = pd.concat(
        [X_area_raw.reset_index(drop=True),
         X_area_apt.reset_index(drop=True)],
        axis=1
    )

    y_area_carreras = df_area["carrera_especifica"].reset_index(drop=True)

    # Codificador de carreras para ESTA área
    le_carrera_area = LabelEncoder()
    y_area_carreras_enc = le_carrera_area.fit_transform(y_area_carreras)

    # =============================
    # 4.1 Validación cruzada (CV)
    # =============================
    rf_base = RandomForestClassifier(**rf_params_carreras)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        rf_base, X_area_full, y_area_carreras_enc,
        cv=cv, scoring="accuracy", n_jobs=-1
    )
    cv_scores_por_area[area] = cv_scores
    print(f"   🔁 CV accuracy (3 folds) en {area}: "
          f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # =============================
    # 4.2 Train/Test split final
    # =============================
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_area_full, y_area_carreras_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_area_carreras_enc
    )

    rf_carreras_area = RandomForestClassifier(**rf_params_carreras)

    rf_carreras_area.fit(X_train_a, y_train_a)

    # Evaluación en test
    y_pred_a = rf_carreras_area.predict(X_test_a)
    acc_a = accuracy_score(y_test_a, y_pred_a)
    accuracies_por_area[area] = acc_a
    print(f"✅ {area} → Accuracy carreras (TEST): {acc_a:.3f}")

    # Evaluación en train (para ver overfitting)
    y_pred_train_a = rf_carreras_area.predict(X_train_a)
    acc_train_a = accuracy_score(y_train_a, y_pred_train_a)
    print(f"   🔎 Accuracy carreras (TRAIN) en {area}: {acc_train_a:.3f}")

    modelos_carrera_por_area[area] = {
        "modelo": rf_carreras_area,
        "encoder": le_carrera_area,
        "columnas": X_area_full.columns.tolist()
    }

# Mostrar resumen de accuracies por área
print("\n📈 Resumen de accuracy por área en carreras (TEST):")
for area, acc in accuracies_por_area.items():
    print(f"   - {area}: {acc:.3f}")

print("\n🔁 Resumen CV (3-fold) por área en carreras:")
for area, scores in cv_scores_por_area.items():
    print(f"   - {area}: {scores.mean():.3f} ± {scores.std():.3f}")

# Guardar todos los modelos por área en un solo .pkl
joblib.dump(modelos_carrera_por_area, "modelos_carrera_por_area.pkl", compress=3)
print("\n💾 Modelos de carreras específicas por área guardados exitosamente como 'modelos_carrera_por_area.pkl'")

print("\n🎉 ENTRENAMIENTO COMPLETADO CON ÉXITO 🎉")
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================
# 1. CARGAR Y PREPARAR EL DATASET ÉLITE
# =========================================
DATASET_FILENAME = "dataset_vocacional_elite_carreras_90.csv"

try:
    df = pd.read_csv(DATASET_FILENAME)
    print(f"✅ Dataset cargado: {DATASET_FILENAME}")
except FileNotFoundError:
    print("❌ Error: Ejecuta primero el script que genera 'dataset_vocacional_elite_carreras_90.csv'")
    exit()

# Features originales (preguntas) y targets
X_raw = df.drop(columns=["area_carrera", "carrera_especifica"])
y_area = df["area_carrera"]
y_carrera = df["carrera_especifica"]

# -----------------------------------------
# Codificador GLOBAL de carreras (como en tu versión original)
# -----------------------------------------
le_carrera_global = LabelEncoder()
le_carrera_global.fit(y_carrera)
# Lo guardaremos al final como codificador_objetivo_carrera.pkl

# Codificar respuestas categóricas (features) a números
feature_encoders = {}
for col in X_raw.columns:
    if X_raw[col].dtype == 'object':
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col])
        feature_encoders[col] = le

# =================================================================
# 2. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# =================================================================
print("⚙️ Generando features de aptitud a partir de las respuestas...")

map_aptitud_a_preguntas = {
    'verbal': list(range(418, 465)) + list(range(568, 594)),
    'calculo': list(range(465, 505)),
    'logica_abstracta': list(range(505, 553)) + list(range(794, 824)),
    'mecanico': list(range(553, 568)) + list(range(794, 824)),
    'disciplina_organizacion': list(range(99, 183)) + list(range(232, 255)),
    'liderazgo_social': list(range(183, 232)) + list(range(255, 418))
}

X_aptitudes = pd.DataFrame(index=X_raw.index)
for aptitud, ids_preguntas in map_aptitud_a_preguntas.items():
    cols = [f'pregunta_{i}' for i in ids_preguntas if f'pregunta_{i}' in X_raw.columns]
    if len(cols) == 0:
        X_aptitudes[f'{aptitud}_mean'] = 0.0
        X_aptitudes[f'{aptitud}_std'] = 0.0
        X_aptitudes[f'{aptitud}_max'] = 0.0
    else:
        X_aptitudes[f'{aptitud}_mean'] = X_raw[cols].mean(axis=1)
        X_aptitudes[f'{aptitud}_std'] = X_raw[cols].std(axis=1)
        X_aptitudes[f'{aptitud}_max'] = X_raw[cols].max(axis=1)

print("✅ Features de aptitud generadas con éxito.")

# =================================================================
# 3. ENTRENAMIENTO DEL MODELO 1: PREDICTOR DE ÁREAS (RandomForest)
# =================================================================
print("\n🚀 Fase 1: Entrenando el modelo de ÁREAS con RandomForest...")

X_train, X_test, y_train_area, y_test_area = train_test_split(
    X_aptitudes, y_area, test_size=0.2, random_state=42, stratify=y_area
)

# Codificador del target de áreas
le_area = LabelEncoder()
y_train_area_enc = le_area.fit_transform(y_train_area)
y_test_area_enc = le_area.transform(y_test_area)

rf_area = RandomForestClassifier(
    n_estimators=200,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

rf_area.fit(X_train, y_train_area_enc)

y_pred_area = rf_area.predict(X_test)
acc_area = accuracy_score(y_test_area_enc, y_pred_area)
print(f"🎯 Accuracy ÁREAS: {acc_area:.4f}")
print("\n📊 Reporte de Clasificación (ÁREAS):\n",
      classification_report(y_test_area_enc, y_pred_area, target_names=le_area.classes_))

# Guardar modelo de áreas y encoders
joblib.dump(rf_area, "modelo_areas_elite.pkl", compress=3)
joblib.dump(feature_encoders, "codificadores_features.pkl", compress=3)
joblib.dump(le_area, "codificador_objetivo_area.pkl", compress=3)
print("✅ Modelo y codificadores de ÁREAS guardados.")

# =================================================================
# 4. ENTRENAMIENTO DEL MODELO 2: CARRERAS POR ÁREA (RandomForest REGULARIZADO)
# =================================================================
print("\n🚀 Fase 2: Entrenando modelos de CARRERAS ESPECÍFICAS por área...")

modelos_carrera_por_area = {}
accuracies_por_area = {}
cv_scores_por_area = {}

# Hiperparámetros REGULARIZADOS para carreras
rf_params_carreras = {
    "n_estimators": 200,
    "max_depth": 10,       # árboles relativamente poco profundos
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": 0.3,   # menos features por split -> más diversidad
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": 42
}

for area in df["area_carrera"].unique():
    print(f"\n📚 Entrenando modelo para el área: {area}")
    df_area = df[df["area_carrera"] == area]

    idx_area = df_area.index

    X_area_raw = X_raw.loc[idx_area]
    X_area_apt = X_aptitudes.loc[idx_area]

    # Features finales para carreras: preguntas + aptitudes agregadas
    X_area_full = pd.concat(
        [X_area_raw.reset_index(drop=True),
         X_area_apt.reset_index(drop=True)],
        axis=1
    )

    y_area_carreras = df_area["carrera_especifica"].reset_index(drop=True)

    # Codificador de carreras para ESTA área
    le_carrera_area = LabelEncoder()
    y_area_carreras_enc = le_carrera_area.fit_transform(y_area_carreras)

    # =============================
    # 4.1 Validación cruzada (CV)
    # =============================
    rf_base = RandomForestClassifier(**rf_params_carreras)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        rf_base, X_area_full, y_area_carreras_enc,
        cv=cv, scoring="accuracy", n_jobs=-1
    )
    cv_scores_por_area[area] = cv_scores
    print(f"   🔁 CV accuracy (3 folds) en {area}: "
          f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # =============================
    # 4.2 Train/Test split final
    # =============================
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_area_full, y_area_carreras_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_area_carreras_enc
    )

    rf_carreras_area = RandomForestClassifier(**rf_params_carreras)

    rf_carreras_area.fit(X_train_a, y_train_a)

    # Evaluación en test
    y_pred_a = rf_carreras_area.predict(X_test_a)
    acc_a = accuracy_score(y_test_a, y_pred_a)
    accuracies_por_area[area] = acc_a
    print(f"✅ {area} → Accuracy carreras (TEST): {acc_a:.3f}")

    # Evaluación en train (para ver overfitting)
    y_pred_train_a = rf_carreras_area.predict(X_train_a)
    acc_train_a = accuracy_score(y_train_a, y_pred_train_a)
    print(f"   🔎 Accuracy carreras (TRAIN) en {area}: {acc_train_a:.3f}")

    modelos_carrera_por_area[area] = {
        "modelo": rf_carreras_area,
        "encoder": le_carrera_area,
        "columnas": X_area_full.columns.tolist()
    }

# Mostrar resumen de accuracies por área
print("\n📈 Resumen de accuracy por área en carreras (TEST):")
for area, acc in accuracies_por_area.items():
    print(f"   - {area}: {acc:.3f}")

print("\n🔁 Resumen CV (3-fold) por área en carreras:")
for area, scores in cv_scores_por_area.items():
    print(f"   - {area}: {scores.mean():.3f} ± {scores.std():.3f}")

# =================================================================
# 5. GUARDAR MODELOS Y CODIFICADORES COMO EN TU ESTRUCTURA ORIGINAL
# =================================================================

# 1) Modelo de CARRERAS:
#    Guardamos el diccionario por área con el nombre antiguo
joblib.dump(modelos_carrera_por_area, "modelo_carreras_elite.pkl", compress=3)

# 2) Codificador GLOBAL de CARRERAS
joblib.dump(le_carrera_global, "codificador_objetivo_carrera.pkl", compress=3)

print("\n💾 Modelos y codificadores guardados exitosamente:")
print("   - modelo_areas_elite.pkl")
print("   - modelo_carreras_elite.pkl  (diccionario por área)")
print("   - codificadores_features.pkl")
print("   - codificador_objetivo_area.pkl")
print("   - codificador_objetivo_carrera.pkl")

print("\n🎉 ENTRENAMIENTO COMPLETADO CON ÉXITO 🎉")
