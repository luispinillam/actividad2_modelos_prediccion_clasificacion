Análisis y predicción con el dataset Palmer Penguins

1. Descripción del proyecto

Este proyecto tiene como objetivo realizar un análisis de datos y entrenar modelos de aprendizaje automático utilizando el conjunto de datos Palmer Penguins. Se desarrollan dos tareas: un modelo de regresión para estimar la masa corporal de los pingüinos y un modelo de clasificación para predecir su especie a partir de características físicas.

2. Fuente de los datos

El dataset utilizado se obtiene desde Hugging Face:

https://huggingface.co/datasets/methodidacte/penguins

Los datos obtenidos de Hugging Face, contienen información sobre las especies de pingüinos y sus características físicas, como la longitud y profundidad del pico, la longitud de las aletas y la masa corporal.

3. Preparación de los datos

Se seleccionaron las siguientes variables:

    Species
    Culmen Length (mm)
    Culmen Depth (mm)
    Flipper Length (mm)
    Body Mass (g)

El conjunto original contiene 344 registros. Después de seleccionar las variables y eliminar los registros incompletos, se utilizaron 342 registros para los modelos. Se eliminaron dos registros que tenían datos faltantes en las columnas seleccionadas.

Para el entrenamiento se utilizó una división del 80 % para entrenamiento y del 20 % para prueba. Los modelos incorporan un imputador de la mediana mediante un pipeline de Scikit-learn.

4. Modelo de regresión

El objetivo de la regresión es estimar la masa corporal de los pingüinos en gramos utilizando la longitud del pico, la profundidad del pico y la longitud de las aletas.

    Algoritmo: Regresión lineal.
    Resultados obtenidos:
    MAE: 282.89 gramos.
    RMSE: 349.98 gramos.
    R²: 0.7829.

El modelo explica aproximadamente el 78,29 % de la variabilidad observada en la masa corporal dentro de los datos de prueba. El MAE indica que las predicciones presentan un error absoluto medio de aproximadamente 283 gramos.

5. Modelo de clasificación

El objetivo de la clasificación es predecir la especie del pingüino utilizando sus características físicas.

Algoritmo: Random Forest Classifier.

Resultados obtenidos:

Exactitud (accuracy): 1.0000, equivalente al 100 %.

La matriz de confusión no presentó errores de clasificación en el conjunto de prueba.

Aunque el resultado fue perfecto en esta división de los datos, no garantiza que el modelo tenga la misma precisión con datos nuevos. Sería conveniente evaluar su comportamiento mediante validación cruzada.

6. Ejecución del proyecto

Para ejecutar el proyecto localmente, se recomienda utilizar Python y crear un entorno virtual. Se instalan las dependencias y se ejecuta el entrenamiento:

    pip install -r requirements.txt
    python entrenar_modelos.py

El script descarga el dataset, prepara los datos, entrena los modelos, calcula las métricas y genera el archivo resumen_modelos.txt.

También se incluye el notebook analisis_pinguinos.ipynb, donde se desarrolla el análisis exploratorio y la evaluación de los modelos.

7. Automatización con GitHub Actions

El proyecto incluye un flujo de trabajo ubicado en .github/workflows/modelos.yml. Este flujo se ejecuta automáticamente cuando se realizan cambios mediante un push a la rama main.

El proceso instala las dependencias, ejecuta el script de entrenamiento y publica resumen_modelos.txt como un artefacto descargable de la ejecución.

8. Herramientas utilizadas

    Python
    Pandas
    NumPy
    Matplotlib
    Seaborn
    Scikit-learn
    Hugging Face Datasets
    Jupyter Notebook
    Git y GitHub
    GitHub Actions