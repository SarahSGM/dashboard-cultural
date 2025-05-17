import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Cultural DANE",
    page_icon="📊",
    layout="wide"
)

# Título
st.title("📊 Análisis de Hábitos Culturales - DANE")
st.markdown("---")

# Carga de datos
@st.cache_data  # Cache para mejor rendimiento
def load_data():
    try:
        df = pd.read_excel("Ingeneria_caracteristicas.xlsx")
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

df = load_data()

# Mostrar datos si se cargaron
if not df.empty:
    st.success("¡Datos cargados correctamente!")
    
    # Sidebar con filtros
    st.sidebar.header("Filtros")
    
    # Filtro por sexo
    sexo_options = df['Sexo'].unique()
    selected_sexo = st.sidebar.multiselect(
        "Seleccione sexo:",
        options=sexo_options,
        default=sexo_options
    )
    
    # Filtro por edad
    min_age = int(df['¿cuántos años cumplidos tiene?'].min())
    max_age = int(df['¿cuántos años cumplidos tiene?'].max())
    age_range = st.sidebar.slider(
        "Rango de edad:",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    
    # Aplicar filtros
    filtered_df = df[
        (df['Sexo'].isin(selected_sexo)) & 
        (df['¿cuántos años cumplidos tiene?'] >= age_range[0]) & 
        (df['¿cuántos años cumplidos tiene?'] <= age_range[1])
    ]

    # Métricas clave
    st.subheader("Resumen Estadístico")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total registros", len(filtered_df))
    with col2:
        st.metric("Edad promedio", f"{filtered_df['¿cuántos años cumplidos tiene?'].mean():.1f} años")
    with col3:
        asistencia = filtered_df['En los últimos 12 meses, ¿usted asistió a teatro, ópera o danza?'].value_counts(normalize=True).get('Sí', 0)*100
        st.metric("Asistencia a teatro/danza", f"{asistencia:.1f}%")

    # Visualizaciones
    st.subheader("Distribución por Variables")
    
    # Gráfico 1: Participación cultural por edad
    fig1, ax1 = plt.subplots()
    sns.histplot(data=filtered_df, x='¿cuántos años cumplidos tiene?', hue='Sexo', kde=True, ax=ax1)
    ax1.set_title("Distribución por Edad y Sexo")
    st.pyplot(fig1)
    
    # Gráfico 2: Nivel educativo vs actividades culturales
    st.markdown("### Participación Cultural por Nivel Educativo")
    educacion_order = filtered_df['¿cuál es el nivel educativo más alto alcanzado?'].value_counts().index
    fig2 = plt.figure(figsize=(10, 6))
    sns.countplot(
        data=filtered_df,
        y='¿cuál es el nivel educativo más alto alcanzado?',
        order=educacion_order,
        hue='En los últimos 12 meses, ¿usted asistió a conciertos, recitales, presentaciones de música en espacios abiertos o cerrados en vivo?'
    )
    st.pyplot(fig2)
    
    # Mostrar datos filtrados
    if st.checkbox("Mostrar datos filtrados"):
        st.dataframe(filtered_df)
else:
    st.warning("No se encontraron datos para mostrar. Verifica tu archivo Excel.")

# Créditos
st.markdown("---")
st.caption("Proyecto académico - Datos del DANE | Desarrollado con Streamlit")