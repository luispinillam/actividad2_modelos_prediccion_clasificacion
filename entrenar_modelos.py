from datetime import datetime

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Descargar el dataset
dataset = load_dataset("methodidacte/penguins")
nombre_split = list(dataset.keys())[0]
df = dataset[nombre_split].to_pandas()

# 2. Seleccionar y limpiar las variables
columnas = [
    "Species",
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
]

df_modelo = df[columnas].copy()
df_modelo = df_modelo.dropna()

# 3. Preparar las variables para la regresión
variables = [
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
]

X = df_modelo[variables]
y = df_modelo["Body Mass (g)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo_regresion = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)

modelo_regresion.fit(X_train, y_train)
y_pred = modelo_regresion.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 4. Preparar y entrenar el modelo de clasificación
X_clf = df_modelo[variables]
y_clf = df_modelo["Species"]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf,
)

modelo_clasificacion = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(random_state=42),
)

modelo_clasificacion.fit(X_train_clf, y_train_clf)
y_pred_clf = modelo_clasificacion.predict(X_test_clf)

exactitud = accuracy_score(y_test_clf, y_pred_clf)
matriz = confusion_matrix(y_test_clf, y_pred_clf)
reporte = classification_report(
    y_test_clf, y_pred_clf, zero_division=0
)

# 5. Guardar los resultados en un archivo TXT
with open("resumen_modelos.txt", "w", encoding="utf-8") as archivo:
    archivo.write("RESUMEN DEL PROYECTO DE MACHINE LEARNING\n")
    archivo.write("=" * 45 + "\n\n")

    archivo.write("DATASET Y PREPROCESAMIENTO\n")
    archivo.write("Fuente: https://huggingface.co/datasets/methodidacte/penguins\n")
    archivo.write(f"Filas originales: {df.shape[0]}\n")
    archivo.write(f"Columnas originales: {df.shape[1]}\n")
    archivo.write(f"Filas después de la limpieza: {df_modelo.shape[0]}\n")
    archivo.write(f"Columnas seleccionadas: {df_modelo.shape[1]}\n")
    archivo.write(f"Filas eliminadas: {df.shape[0] - df_modelo.shape[0]}\n")
    archivo.write(f"Duplicados originales: {df.duplicated().sum()}\n")
    archivo.write(
        f"Valores faltantes finales: {df_modelo.isnull().sum().sum()}\n\n"
    )

    archivo.write("MODELO DE REGRESIÓN LINEAL\n")
    archivo.write("Objetivo: predecir el peso en gramos.\n")
    archivo.write(f"Variables: {', '.join(variables)}\n")
    archivo.write("Algoritmo: LinearRegression.\n")
    archivo.write("División: 80% entrenamiento y 20% prueba.\n")
    archivo.write("Semilla aleatoria: 42.\n")
    archivo.write(f"Registros de entrenamiento: {len(X_train)}\n")
    archivo.write(f"Registros de prueba: {len(X_test)}\n")
    archivo.write(f"MAE: {mae:.2f} gramos\n")
    archivo.write(f"RMSE: {rmse:.2f} gramos\n")
    archivo.write(f"R²: {r2:.4f}\n")

    regresion = modelo_regresion.named_steps["linearregression"]
    archivo.write(f"Intercepto: {regresion.intercept_:.4f}\n")
    for variable, coeficiente in zip(variables, regresion.coef_):
        archivo.write(f"Coeficiente de {variable}: {coeficiente:.4f}\n")

    archivo.write(
        "\nInterpretación: R² indica la proporción de variación del peso "
        "explicada por el modelo en los datos de prueba.\n\n"
    )

    archivo.write("MODELO DE CLASIFICACIÓN\n")
    archivo.write("Objetivo: predecir la especie.\n")
    archivo.write(f"Variables: {', '.join(variables)}\n")
    archivo.write("Algoritmo: RandomForestClassifier.\n")
    archivo.write("Parámetro configurado: random_state=42.\n")
    archivo.write("División: 80% entrenamiento y 20% prueba, estratificada.\n")
    archivo.write(f"Registros de entrenamiento: {len(X_train_clf)}\n")
    archivo.write(f"Registros de prueba: {len(X_test_clf)}\n")
    archivo.write(f"Exactitud: {exactitud:.4f}\n")
    archivo.write("Matriz de confusión:\n")
    archivo.write(str(matriz) + "\n")
    archivo.write("Reporte de clasificación:\n")
    archivo.write(reporte + "\n")

    archivo.write("CONCLUSIONES Y LIMITACIONES\n")
    archivo.write(
        "Se entrenaron modelos para estimar el peso y clasificar la especie. "
        "Los resultados corresponden a una división específica de los datos. "
        "Se recomienda evaluar los modelos con validación cruzada y nuevos datos.\n"
    )
    archivo.write(
        f"\nFecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

print("Archivo resumen_modelos.txt generado correctamente.")
print(f"Registros utilizados: {len(df_modelo)}")
print(f"MAE: {mae:.2f} gramos | RMSE: {rmse:.2f} | R²: {r2:.4f}")
print(f"Exactitud de clasificación: {exactitud:.4f}")
