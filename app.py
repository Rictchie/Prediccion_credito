import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Predicción de créditos', page_icon='🏦', layout='wide')
st.title('🏦 Predicción de aprobación de créditos')
st.caption('Random Forest · Selección de variables RFE')

@st.cache_resource
def cargar_modelo():
    return joblib.load('random_forest_credito.joblib')

try:
    paquete = cargar_modelo()
except FileNotFoundError:
    st.error('Coloca random_forest_credito.joblib en la misma carpeta que app_credito.py.')
    st.stop()
except Exception as e:
    st.error(f'No se pudo cargar el modelo: {e}')
    st.stop()

modelo = paquete['modelo']
preprocesador = paquete['preprocesador']
selector = paquete['selector_rfe']
columnas = paquete['variables_originales']

# Se recuperan los nombres de columnas y categorías aprendidas al entrenar.
num_cols = list(preprocesador.transformers_[0][2])
cat_cols = list(preprocesador.transformers_[1][2])
num_imputer = preprocesador.named_transformers_['num'].named_steps['imputer']
cat_pipe = preprocesador.named_transformers_['cat']
cat_imputer = cat_pipe.named_steps['imputer']
onehot = cat_pipe.named_steps['onehot']

st.subheader('Datos del cliente')
with st.form('formulario'):
    entradas = {}
    c1, c2 = st.columns(2)
    for i, nombre in enumerate(columnas):
        with (c1 if i % 2 == 0 else c2):
            if nombre in num_cols:
                j = num_cols.index(nombre)
                valor = float(num_imputer.statistics_[j])
                if not np.isfinite(valor):
                    valor = 0.0
                entradas[nombre] = st.number_input(nombre, value=valor, format='%.2f')
            elif nombre in cat_cols:
                j = cat_cols.index(nombre)
                opciones = list(onehot.categories_[j])
                predeterminado = cat_imputer.statistics_[j]
                indice = next((k for k, x in enumerate(opciones) if x == predeterminado), 0)
                entradas[nombre] = st.selectbox(nombre, opciones, index=indice)
    enviar = st.form_submit_button('Predecir aprobación', type='primary', use_container_width=True)

if enviar:
    nuevo = pd.DataFrame([entradas], columns=columnas)
    try:
        X = selector.transform(preprocesador.transform(nuevo))
        pred = modelo.predict(X)[0]
        clases = list(modelo.classes_)
        proba = modelo.predict_proba(X)[0]
        prob_si = float(proba[clases.index(1)])
    except Exception as e:
        st.error(f'Error al predecir: {e}')
        st.stop()

    st.divider()
    st.subheader('Resultado de la predicción')
    if pred == 1:
        st.success('Predicción del modelo: CRÉDITO APROBADO')
    else:
        st.warning('Predicción del modelo: CRÉDITO NO APROBADO')
    a, b = st.columns(2)
    a.metric('Probabilidad de aprobación', f'{prob_si:.1%}')
    b.metric('Probabilidad de no aprobación', f'{1-prob_si:.1%}')
    st.progress(prob_si)
    st.caption('Resultado estadístico del modelo, no una decisión crediticia definitiva.')

    st.subheader('Importancia de variables (Random Forest)')
    nombres = preprocesador.get_feature_names_out()[selector.get_support()]
    importancia = pd.DataFrame({'Variable': nombres, 'Importancia': modelo.feature_importances_})
    importancia = importancia.sort_values('Importancia', ascending=False)
    st.bar_chart(importancia.set_index('Variable')['Importancia'])
    with st.expander('Ver datos ingresados'):
        st.dataframe(nuevo, hide_index=True)
