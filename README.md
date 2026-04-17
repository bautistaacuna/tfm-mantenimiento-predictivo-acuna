# tfm-mantenimiento-predictivo-acuna
Código del TFM: Detección de fallas en sistemas basados en sensores mediante ML, XAI y LLM

# Detección de fallas en sistemas de sensores mediante ML, XAI y LLM

Trabajo de Fin de Máster — Máster Universitario en Análisis de Datos Masivos (Big Data)  
Universidad Europea de Madrid  
Autor: Juan Bautista Acuña  
Director: Dr. Álvaro Calle Cordón  
Curso: 2024-25

## Descripción

Solución end-to-end para la detección temprana y diagnóstico de fallas en sistemas basados en sensores, integrando Machine Learning (ML), Inteligencia Artificial Explicable (XAI) y modelos de lenguaje de gran escala (LLM).

El sistema fue desarrollado y validado sobre el dataset **APS Failure at Scania Trucks** (UCI Machine Learning Repository).

## Estructura del repositorio
'''
tfm-mantenimiento-predictivo-acuna
|-app
| |-api
| | |-main.py				# API REST con FastAPI
| |-dashboard
| | |-dashboard.py			# Dashboard interactivo con Streamlit
|- data                  		# Dataset (no incluido, ver instrucciones)
|- models               		# Modelos entrenados serializados
| |-random_forest.pkl
| |-resultados_modelos.pkl
| |-shap_data.pkl
| |-xgboost.pkl
|- notebooks            		# Análisis exploratorio y experimentos
| |- 01_Preprocesamiento.ipynb   	# Preprocesamiento e imputación
| |- 02_modelado.ipynb           	# Entrenamiento de modelos
| |- 03_shap.ipynb   		      	# Explicabilidad con SHAP
| |- 04_diagnosticos_claude.ipynb	# Integración con Claude API
|- outputs
|- requirements.txt
|- README.md
'''

## Requisitos

- Python 3.10 o superior
- Cuenta en [Anthropic](https://www.anthropic.com/) con API key activa

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/bautistaacuna/tfm-mantenimiento-predictivo-acuna.git
cd tfm-mantenimiento-predictivo-acuna
```

2. Crear un entorno virtual e instalar dependencias:
```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configurar la API key de Anthropic:
```bash
# En Windows
set ANTHROPIC_API_KEY=tu_api_key

# En Mac/Linux
export ANTHROPIC_API_KEY=tu_api_key
```

## Dataset

El dataset no está incluido en el repositorio por su tamaño. Descargarlo desde:

https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks

Colocar los archivos descargados dentro de la carpeta `data/`.

## Ejecución

### 1. Exploración y entrenamiento (notebooks)

Ejecutar los notebooks en orden dentro de la carpeta `notebooks/`:
01_Preprocesamiento.ipynb   → Limpieza, imputación y normalización
02_modelado.ipynb           → Entrenamiento de Random Forest y XGBoost
03_shap.ipynb               → Análisis de explicabilidad con SHAP
04_diagnosticos_claude.ipynb→ Integración con Claude API

### 2. Iniciar la API REST

```bash
uvicorn app.api.main:app --reload
```

### 3. Iniciar el dashboard

```bash
streamlit run app/dashboard/dashboard.py
```

Abrir el navegador en `http://localhost:8501`

## Tecnologías utilizadas

| Componente | Tecnología |
|---|---|
| Modelos de ML | Scikit-learn, XGBoost |
| Explicabilidad | SHAP |
| API REST | FastAPI |
| Dashboard | Streamlit |
| Diagnósticos en lenguaje natural | Anthropic Claude API |
| Notebooks | Jupyter |
