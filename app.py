import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================================================================
# INDICACIONES OBLIGATORIAS (Nombre, Código, Enlace Colab)
# =========================================================================
st.title("Predicción de Riesgo de Anemia Infantil - 2025")
st.markdown("---")
st.sidebar.header("Información del Estudiante")
st.sidebar.write("**Nombre:** Joaquin Augusto Otoya Ravelo")
st.sidebar.write("**Código ISIL:** 70928761")

# Enlace obligatorio en modo lector a tu cuaderno de Colab
url_colab = "https://colab.research.google.com/drive/1guNkyWRVv13NZFqMGYbf_yF6vwA_F929?usp=sharing"
st.sidebar.markdown(f"[🔗 Ver Cuaderno Google Colab]({url_colab})")

st.markdown("""
Esta aplicación utiliza modelos de Machine Learning para predecir si un distrito 
presentará un **Nivel Crítico de Anemia (mayor al 20%)** en el año 2025, 
basándose en los indicadores de salud registrados durante el año 2024.
""")

# =========================================================================
# CARGA DE MODELOS Y ESTRUCTURA DE COLUMNAS
# =========================================================================
try:
    # Cargamos el modelo ganador (usaremos Random Forest como ejemplo base)
    modelo = joblib.load('modelos/modelo_random_forest.pkl')
    # Cargamos la lista de columnas que guardamos en Colab
    columnas_entrenamiento = joblib.load('modelos/columnas_modelo.pkl')
except Exception as e:
    st.error(f"Error al cargar los archivos del modelo: {e}")
    st.stop()

# =========================================================================
# INTERFAZ DE USUARIO - FORMULARIO DE ENTRADA
# =========================================================================
st.subheader("Ingrese los datos del Distrito para la predicción:")

# 1. Recuperamos la lista de provincias basándonos en las columnas One-Hot del modelo
provincias_disponibles = [col.replace('PROVINCIA_', '') for col in columnas_entrenamiento if col.startswith('PROVINCIA_')]
provincias_disponibles.sort()

# Campos interactivos en la web
provincia_seleccionada = st.selectbox("Seleccione la Provincia:", provincias_disponibles)
evaluados_2024 = st.number_input("Cantidad de niños evaluados en 2024:", min_value=1, value=100, step=1)
porc_2024 = st.slider("Porcentaje de anemia registrado en 2024 (%):", min_value=0.0, max_value=100.0, value=15.0, step=0.1)

# =========================================================================
# PROCESAMIENTO DE LA PREDICCIÓN
# =========================================================================
if st.button("Calcular Riesgo para 2025"):
    
    # 1. Creamos una fila vacía con ceros respetando la estructura exacta de columnas que pide el modelo
    datos_usuario = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)
    
    # 2. Asignamos los valores numéricos ingresados por el usuario
    datos_usuario['EVALUADOS_2024'] = evaluados_2024
    datos_usuario['PORC_2024'] = porc_2024
    
    # 3. Activamos con un 1 la columna One-Hot de la provincia seleccionada (si aplica)
    columna_provincia = f"PROVINCIA_{provincia_seleccionada}"
    if columna_provincia in datos_usuario.columns:
        datos_usuario[columna_provincia] = 1
        
    # 4. Realizamos la predicción con el modelo cargado
    prediccion = modelo.predict(datos_usuario)[0]
    
    st.markdown("---")
    st.subheader("Resultado de la Evaluación:")
    
    # 5. Mostramos el resultado visual al usuario de forma clara
    if prediccion == 1:
        st.error("🚨 **Predicción: RIESGO ALTO (CRÍTICO)**")
        st.write("El modelo estima que este distrito superará el 20% de incidencia de anemia infantil en 2025. Se recomiendan intervenciones sanitarias y nutricionales inmediatas.")
    else:
        st.success("✅ **Predicción: RIESGO BAJO (CONTENIDO)**")
        st.write("El modelo estima que el distrito mantendrá controlada su tasa de anemia infantil por debajo del umbral crítico del 20% para el 2025.")
