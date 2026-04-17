# ============================================================
# API REST - FastAPI
# TFM: Deteccion de fallas en sistemas basados en sensores
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import anthropic
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv('../../.env')

# Inicializar app
app = FastAPI(
    title="API de Deteccion de Fallas - Scania APS",
    description="API REST para prediccion de fallas en el sistema de presion de aire de camiones Scania mediante Machine Learning, XAI y LLMs.",
    version="1.0.0"
)

# Cargar modelos y datos al iniciar
print("Cargando modelos...")
xgb_model      = joblib.load('../../models/xgboost.pkl')
resultados      = joblib.load('../../models/resultados_modelos.pkl')
shap_data       = joblib.load('../../models/shap_data.pkl')
umbral_opt_xgb  = resultados['umbral_opt_xgb']
client          = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
print("Modelos cargados correctamente.")


# ============================================================
# Modelos de datos (Pydantic)
# ============================================================

class RegistroSensor(BaseModel):
    sensores: dict
    perfil_usuario: Optional[str] = 'operador'

class SolicitudDiagnostico(BaseModel):
    prediccion: str
    probabilidad: float
    top_shap: list
    perfil_usuario: Optional[str] = 'operador'

class SolicitudChat(BaseModel):
    pregunta: str
    contexto_prediccion: Optional[str] = ''


# ============================================================
# Funciones auxiliares
# ============================================================

def obtener_top_shap_api(shap_vals, feature_names, top_n=5):
    import shap as shap_lib
    explainer = shap_lib.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(
        pd.DataFrame([shap_vals], columns=feature_names),
        check_additivity=False
    )
    indices = np.argsort(np.abs(shap_values[0]))[::-1][:top_n]
    resultado = []
    for i in indices:
        resultado.append({
            'sensor'    : feature_names[i],
            'valor_shap': round(float(shap_values[0][i]), 4),
            'direccion' : 'aumenta riesgo de falla' if shap_values[0][i] > 0 else 'reduce riesgo de falla'
        })
    return resultado


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {"mensaje": "API de Deteccion de Fallas Scania APS - v1.0.0"}


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": "XGBoost", "umbral": umbral_opt_xgb}


@app.post("/predecir")
def predecir(registro: RegistroSensor):
    try:
        feature_names = list(registro.sensores.keys())
        valores       = list(registro.sensores.values())
        df_input      = pd.DataFrame([valores], columns=feature_names)

        prob       = xgb_model.predict_proba(df_input)[0][1]
        prediccion = 'FALLA APS' if prob >= umbral_opt_xgb else 'NORMAL'

        top_shap = obtener_top_shap_api(valores, feature_names)

        return {
            "prediccion"  : prediccion,
            "probabilidad": round(float(prob), 4),
            "umbral"      : umbral_opt_xgb,
            "top_shap"    : top_shap
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diagnostico")
def diagnostico(solicitud: SolicitudDiagnostico):
    try:
        shap_texto = '\n'.join([
            f"   - {s['sensor']}: SHAP={s['valor_shap']} -> {s['direccion']}"
            for s in solicitud.top_shap
        ])

        prompt = f"""Eres un asistente experto en mantenimiento predictivo industrial.
Se te proporciona el resultado de un modelo de machine learning que analiza datos de sensores
de camiones Scania para detectar fallas en el sistema de presion de aire (APS).

RESULTADO DEL MODELO:
- Prediccion      : {solicitud.prediccion}
- Probabilidad    : {solicitud.probabilidad:.4f}
- Umbral utilizado: {umbral_opt_xgb:.4f}

VARIABLES MAS INFLUYENTES (SHAP):
{shap_texto}

PERFIL DEL USUARIO: {solicitud.perfil_usuario}

INSTRUCCIONES:
1. Redacta un diagnostico claro y util para un {solicitud.perfil_usuario}.
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

        return {"diagnostico": mensaje.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(solicitud: SolicitudChat):
    try:
        prompt = f"""Eres un asistente experto en mantenimiento predictivo industrial de camiones Scania.
Responde la pregunta del usuario basandote unicamente en el contexto proporcionado.
No inventes informacion adicional.

CONTEXTO DE LA PREDICCION ACTIVA:
{solicitud.contexto_prediccion}

PREGUNTA DEL USUARIO:
{solicitud.pregunta}"""

        mensaje = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 500,
            messages   = [{"role": "user", "content": prompt}]
        )

        return {"respuesta": mensaje.content[0].text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))