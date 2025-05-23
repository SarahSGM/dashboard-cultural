import os
print("Ruta actual de trabajo:", os.getcwd())
df = pd.read_excel("Ingenieria_caracteristicas.xlsx")

import pandas as pd
import streamlit as st
import os
@st.cache_data

def cargar_datos():
    base_path = os.path.dirname(__file__)
    ruta_excel = os.path.join(base_path, "Ingenieria_caracteristicas.xlsx")
    df = pd.read_excel(ruta_excel)
    return df

df = cargar_datos()

st.title("Primeros 10 registros del dataset de hábitos culturales")
st.dataframe(df.head(10))