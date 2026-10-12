# Predicción de aprobación de créditos — Random Forest

Aplicación Streamlit para ingresar datos originales del cliente y obtener una predicción con Random Forest. Utiliza imputación, OneHotEncoder y selección RFE aprendidos en el entrenamiento.

## Archivo indispensable pendiente

**El modelo entrenado NO se incluye en este ZIP**, porque se genera en tu sesión de Google Colab y no se ha proporcionado aquí. Descarga `random_forest_credito.joblib` desde la sección 12 de tu notebook y colócalo junto a `app.py` antes de subir el repositorio a GitHub. **Sin ese archivo la aplicación no puede predecir.**

El archivo debe contener las claves: `modelo`, `preprocesador`, `selector_rfe`, `variables_originales`, `variables_seleccionadas`. Se espera un clasificador con clases 0 (no aprobado) y 1 (aprobado).

## Publicación en GitHub y Streamlit Community Cloud

1. Descomprime esta carpeta.
2. Agrega `random_forest_credito.joblib` a la carpeta raíz, junto a `app.py`.
3. Crea un repositorio en GitHub y sube el contenido de la carpeta (no es necesario subir el ZIP).
4. En https://share.streamlit.io/ inicia sesión con GitHub y selecciona **Create app**.
5. Elige el repositorio, rama `main` y archivo principal `app.py`; despliega la app.
6. Abre el enlace de la aplicación para probar una predicción.

Si el modelo contiene datos confidenciales o no debe publicarse, **no subas el `.joblib` a un repositorio público**. Usa almacenamiento privado y una estrategia de despliegue con acceso controlado.

## Ejecución alternativa

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Advertencias

- Solo carga archivos `.joblib` de fuentes confiables.
- Mantén una versión de scikit-learn compatible con la usada para entrenar el modelo. Si hay errores de deserialización, ajusta la versión de `requirements.txt` a la versión exacta del entorno de entrenamiento.
- La clasificación es una predicción estadística, no una decisión crediticia definitiva.
- La selección RFE fue ajustada antes de la validación cruzada del notebook; esto puede producir estimaciones optimistas de rendimiento.
