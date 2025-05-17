import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Configuración general del dashboard
st.set_page_config(page_title="Hábitos Culturales en Colombia", layout="wide")
st.title("Análisis de Participación Cultural en Colombia")
st.markdown("""
Este tablero interactivo presenta un análisis detallado sobre los hábitos culturales de los colombianos,
utilizando datos del DANE. A través de gráficos, filtros y estadísticas, exploramos cómo factores sociodemográficos
se relacionan con la asistencia y participación en actividades culturales.
""")

# Cargar datos
@st.cache_data

def cargar_datos():
    df = pd.read_excel("Ingenieria_caracteristicas.xlsx")
    return df

df = cargar_datos()

# ---------------- LIMPIEZA DE DATOS ---------------- #
# Eliminar columnas con todos los valores nulos
df.dropna(axis=1, how='all', inplace=True)

# Imputación de valores faltantes (modo para categóricas)
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Rellenar numéricas con la mediana
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ---------------- FILTROS INTERACTIVOS ---------------- #
st.sidebar.header("Filtros")
sexo = st.sidebar.multiselect("Sexo", df['SEXO'].unique(), default=df['SEXO'].unique())
edades = st.sidebar.slider("Edad", int(df['EDAD'].min()), int(df['EDAD'].max()), (18, 80))
etnias = st.sidebar.multiselect("Etnia", df['ETNIA'].unique(), default=df['ETNIA'].unique())
educacion = st.sidebar.multiselect("Nivel Educativo", df['NIVEL EDUCATIVO'].unique(), default=df['NIVEL EDUCATIVO'].unique())

# Aplicar filtros
df_filt = df[
    (df['SEXO'].isin(sexo)) &
    (df['EDAD'] >= edades[0]) & (df['EDAD'] <= edades[1]) &
    (df['ETNIA'].isin(etnias)) &
    (df['NIVEL EDUCATIVO'].isin(educacion))
]

# ---------------- ANÁLISIS EXPLORATORIO ---------------- #
st.subheader("Distribución de Edad")
st.bar_chart(df_filt['EDAD'].value_counts().sort_index())

st.subheader("Distribución por Sexo")
st.pyplot(sns.countplot(data=df_filt, x='SEXO'))

st.subheader("Nivel Educativo")
st.pyplot(sns.countplot(data=df_filt, y='NIVEL EDUCATIVO', order=df_filt['NIVEL EDUCATIVO'].value_counts().index))

# ---------------- PARTICIPACIÓN CULTURAL ---------------- #
st.header("Participación en Actividades Culturales")
cultural_cols = [
    'P3', 'P4', 'P5', 'ASISTENCIA BIBLIOTECA', 'ASISTENCIA CASAS DE CULTURA',
    'ASISTENCIA CENTROS CUTURALES', 'ASISTENCIA MUSEOS', 'ASISTENCIA EXPOSICIONES',
    'ASISTENCIA MONUMENTOS', 'ASISTENCIA CURSOS', 'PRACTICA CULTURAL', 'lee libros'
]

participacion = {}
for col in cultural_cols:
    participacion[col] = df_filt[col].value_counts(normalize=True).get('Sí', 0)

st.bar_chart(pd.Series(participacion).sort_values())

# ---------------- INGENIERÍA DE CARACTERÍSTICAS ---------------- #
df_filt['PARTICIPA_CULTURAL'] = df_filt[cultural_cols].apply(lambda x: (x == 'Sí').sum(), axis=1)
st.subheader("Participación Cultural Acumulada")
st.hist_chart(df_filt['PARTICIPA_CULTURAL'])

st.subheader("Relación Edad - Participación")
st.scatter_chart(df_filt[['EDAD', 'PARTICIPA_CULTURAL']])

# ---------------- DOCUMENTACIÓN ---------------- #
st.sidebar.markdown("""
**Información del Proyecto**
- Fuente: DANE - Encuesta de Consumo Cultural
- Autor: Sarah & Mia
- Proyecto: PARCIAL III Ingeniería de Características
""")

st.sidebar.markdown("""
**Instrucciones**
1. Usa los filtros para explorar el comportamiento según edad, sexo, etnia y educación.
2. Observa cómo se distribuye la participación cultural.
3. Analiza la relación entre variables sociodemográficas y participación cultural.
""")