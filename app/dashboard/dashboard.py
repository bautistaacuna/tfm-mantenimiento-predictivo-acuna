# ============================================================
# DASHBOARD - Streamlit
# TFM: Deteccion de fallas en sistemas basados en sensores
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import anthropic
import os
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv('../../.env')

# ============================================================
# Configuracion de la pagina
# ============================================================

st.set_page_config(
    page_title="Deteccion de Fallas - Scania APS",
    page_icon="🔧",
    layout="wide"
)

# ============================================================
# Carga de modelos (se cachea para no recargar en cada interaccion)
# ============================================================

@st.cache_resource
def cargar_modelos():
    xgb_model     = joblib.load('../../models/xgboost.pkl')
    resultados     = joblib.load('../../models/resultados_modelos.pkl')
    shap_data      = joblib.load('../../models/shap_data.pkl')
    client         = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    explainer      = shap.TreeExplainer(xgb_model)
    return xgb_model, resultados, shap_data, client, explainer

xgb_model, resultados, shap_data, client, explainer = cargar_modelos()
umbral_opt_xgb  = resultados['umbral_opt_xgb']
X_test_muestra  = shap_data['X_test_muestra']
y_test_muestra  = shap_data['y_test_muestra']
shap_values_xgb = shap_data['shap_values_xgb']

# ============================================================
# Funciones auxiliares
# ============================================================

def obtener_top_shap(shap_vals, feature_names, top_n=5):
    indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    resultado = []
    for i in indices:
        resultado.append({
            'sensor'    : feature_names[i],
            'valor_shap': round(float(shap_vals[i]), 4),
            'direccion' : 'aumenta riesgo' if shap_vals[i] > 0 else 'reduce riesgo'
        })
    return resultado


def generar_diagnostico(prediccion, probabilidad, top_shap, perfil):
    shap_texto = '\n'.join([
        f"   - {s['sensor']}: SHAP={s['valor_shap']} -> {s['direccion']}"
        for s in top_shap
    ])
    prompt = f"""Eres un asistente experto en mantenimiento predictivo industrial.
Se te proporciona el resultado de un modelo de machine learning que analiza datos de sensores
de camiones Scania para detectar fallas en el sistema de presion de aire (APS).

RESULTADO DEL MODELO:
- Prediccion      : {prediccion}
- Probabilidad    : {probabilidad:.4f}
- Umbral utilizado: {umbral_opt_xgb:.4f}

VARIABLES MAS INFLUYENTES (SHAP):
{shap_texto}

PERFIL DEL USUARIO: {perfil}

INSTRUCCIONES:
1. Redacta un diagnostico claro y util para un {perfil}.
2. Explica que significa la prediccion y que sensores influyeron mas.
3. Sugiere una accion concreta de mantenimiento.
4. Usa unicamente la informacion proporcionada, no inventes datos adicionales.
5. El diagnostico debe tener entre 100 y 150 palabras.
6. No uses formato markdown, solo texto plano."""

    mensaje = client.messages.create(
        model      = "claude-haiku-4-5-20251001",
        max_tokens = 500,
        messages   = [{"role": "user", "content": prompt}]
    )
    return mensaje.content[0].text


def generar_respuesta_chat(pregunta, contexto):
    prompt = f"""Eres un asistente experto en mantenimiento predictivo industrial de camiones Scania.
Responde la pregunta del usuario basandote unicamente en el contexto proporcionado.
No inventes informacion adicional.

CONTEXTO DE LA PREDICCION ACTIVA:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}"""

    mensaje = client.messages.create(
        model      = "claude-haiku-4-5-20251001",
        max_tokens = 500,
        messages   = [{"role": "user", "content": prompt}]
    )
    return mensaje.content[0].text


# ============================================================
# Interfaz principal
# ============================================================

st.title("Sistema de Deteccion de Fallas - Scania APS")
st.markdown("Deteccion de fallas en el sistema de presion de aire mediante Machine Learning, XAI y LLMs.")
st.divider()

# Sidebar
st.sidebar.header("Configuracion")
perfil = st.sidebar.selectbox(
    "Perfil de usuario",
    ["operador", "ingeniero", "gerente"]
)

modo = st.sidebar.radio(
    "Modo de analisis",
    ["Seleccionar caso del dataset", "Ingresar caso manualmente"]
)

# ============================================================
# Modo 1: Seleccionar caso del dataset
# ============================================================

if modo == "Seleccionar caso del dataset":

    st.subheader("Seleccion de caso")

    col1, col2 = st.columns(2)
    with col1:
        solo_positivos = st.checkbox("Mostrar solo casos con falla APS real", value=True)

    if solo_positivos:
        indices_disponibles = y_test_muestra[y_test_muestra == 1].index.tolist()
    else:
        indices_disponibles = list(range(len(X_test_muestra)))

    with col2:
        caso_idx = st.selectbox("Selecciona el indice del caso", indices_disponibles)

    if st.button("Analizar caso", type="primary"):

        with st.spinner("Analizando caso..."):

            # Prediccion
            prob       = xgb_model.predict_proba(X_test_muestra.iloc[[caso_idx]])[0][1]
            prediccion = 'FALLA APS' if prob >= umbral_opt_xgb else 'NORMAL'
            real       = 'FALLA APS' if y_test_muestra.iloc[caso_idx] == 1 else 'NORMAL'
            top_shap   = obtener_top_shap(shap_values_xgb[caso_idx], X_test_muestra.columns.tolist())

            # Guardar en session state
            st.session_state['prediccion']  = prediccion
            st.session_state['probabilidad'] = prob
            st.session_state['real']         = real
            st.session_state['top_shap']     = top_shap
            st.session_state['caso_idx']     = caso_idx
            st.session_state['diagnostico']  = None
            st.session_state['historial_chat'] = []

        # Resultados
        st.subheader("Resultado del modelo")
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediccion",   prediccion)
        col2.metric("Probabilidad", f"{prob:.4f}")
        col3.metric("Valor real",   real)

        # SHAP
        st.subheader("Variables mas influyentes (SHAP)")
        df_shap = pd.DataFrame(top_shap)
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in df_shap['valor_shap']]
        ax.barh(df_shap['sensor'], df_shap['valor_shap'], color=colors, edgecolor='black')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_xlabel('Valor SHAP')
        ax.set_title('Contribucion de sensores a la prediccion')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Diagnostico
    if 'prediccion' in st.session_state:
        st.divider()
        st.subheader("Diagnostico en lenguaje natural")

        if st.button("Generar diagnostico con Claude"):
            with st.spinner("Generando diagnostico..."):
                diagnostico = generar_diagnostico(
                    st.session_state['prediccion'],
                    st.session_state['probabilidad'],
                    st.session_state['top_shap'],
                    perfil
                )
                st.session_state['diagnostico'] = diagnostico

        if st.session_state.get('diagnostico'):
            st.info(st.session_state['diagnostico'])

            # Chat
            st.divider()
            st.subheader("Chat - Consultas sobre el diagnostico")

            if 'historial_chat' not in st.session_state:
                st.session_state['historial_chat'] = []

            for msg in st.session_state['historial_chat']:
                with st.chat_message(msg['rol']):
                    st.write(msg['contenido'])

            pregunta = st.chat_input("Escribi tu consulta sobre este caso...")

            if pregunta:
                st.session_state['historial_chat'].append(
                    {'rol': 'user', 'contenido': pregunta}
                )
                contexto = f"""
Prediccion: {st.session_state['prediccion']}
Probabilidad: {st.session_state['probabilidad']:.4f}
Perfil: {perfil}
Diagnostico generado: {st.session_state['diagnostico']}
Top sensores SHAP: {st.session_state['top_shap']}
"""
                with st.spinner("Generando respuesta..."):
                    respuesta = generar_respuesta_chat(pregunta, contexto)

                st.session_state['historial_chat'].append(
                    {'rol': 'assistant', 'contenido': respuesta}
                )
                st.rerun()