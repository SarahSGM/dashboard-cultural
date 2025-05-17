import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Hábitos Culturales en Colombia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar datos
@st.cache_data
def cargar_datos():
    """
    Carga los datos del DANE. Si no existe el archivo, muestra un uploader
    para que el usuario suba el archivo.
    """
    try:
        df = pd.read_csv('datos_dane.csv')
        return df
    except FileNotFoundError:
        st.warning("No se encontró el archivo de datos. Por favor, sube el archivo CSV del DANE.")
        uploaded_file = st.file_uploader("Subir archivo CSV del DANE", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.to_csv('datos_dane.csv', index=False)
            return df
        else:
            # Datos de ejemplo para demostración
            return pd.DataFrame()

# Función para limpiar los datos
def limpiar_datos(df):
    """
    Realiza limpieza de datos:
    - Renombra columnas para mayor claridad
    - Maneja valores nulos
    - Detecta y maneja outliers
    - Transforma variables según sea necesario
    """
    if df.empty:
        return df
    
    # Crear copia para no modificar el original
    df_limpio = df.copy()
    
    # Renombrar columnas para mayor claridad
    columnas_rename = {
        'Sexo': 'sexo',
        '¿cuántos años cumplidos tiene <...>?': 'edad',
        'De acuero con su cultura, pueblo o rasgos físicos usted se reconoce como:': 'etnia',
        '¿sabe leer y escribir?': 'alfabetismo',
        '¿actualmente <...>asiste al preescolar, escuela, colegio o universidad?': 'asiste_educacion',
        '¿cuál es el nivel educativo más alto alcanzado por <...>': 'nivel_educativo',
        'Factor de Expasión': 'factor_expansion',
        '¿en qué actividad ocupó <...> la mayor parte del tiempo la semana pasada?': 'actividad_principal',
        '¿ingreso mensual?': 'ingreso_mensual',
        'En los últimos 12 meses, ¿usted asistió a teatro, ópera o danza?': 'asistio_teatro',
        'Frecuencia': 'frecuencia_teatro',
        'En los últimos 12 meses, ¿usted asistió a conciertos, recitales, presentaciones de música en espacios abiertos o cerrados en vivo?': 'asistio_conciertos',
        'Frecuencia.1': 'frecuencia_conciertos',
        'Asistió a conciertos, recitales, eventos, presentaciones o espectáculos de música en vivo, en espacios abiertos o cerrados, en Bar, restaurante, café o similares': 'asistio_musica_bares',
        'En los últimos 12 meses, ¿usted asistió a exposiciones, ferias y muestras de fotografía, pintura, grabado, dibujo, escultura o artes gráficas?': 'asistio_exposiciones',
        'Frecuencia.2': 'frecuencia_exposiciones',
        '¿asistió a bibliotecas en los últimos 12 meses?': 'asistio_bibliotecas',
        '¿con qué frecuencia?': 'frecuencia_bibliotecas',
        '¿A qué ha ido <…> a la biblioteca en los últimos 12 meses?: a. Leer o consultar libros, periódicos o revistas': 'biblioteca_lectura',
        'En los últimos 12 meses, ¿fue a casas de la cultura?': 'asistio_casas_cultura',
        '¿con qué frecuencia?.1': 'frecuencia_casas_cultura',
        '¿Asistió a centros culturales en los últimos 12 meses?': 'asistio_centros_culturales',
        '¿Con qué frecuencia?': 'frecuencia_centros_culturales',
        'En los últimos 12 meses, ¿visitó museos?': 'visito_museos',
        '¿con qué frecuencia?.2': 'frecuencia_museos',
        '¿asistió a galerías de arte y salas de exposiciones en los últimos 12 meses?': 'asistio_galerias',
        '¿con qué frecuencia?.3': 'frecuencia_galerias',
        'En los últimos 12 meses, ¿fue a monumentos históricos, sitios arqueológicos, monumentos nacionales o centros históricos?': 'visito_monumentos',
        '¿con qué frecuencia?.4': 'frecuencia_monumentos',
        '¿Usted tomó cursos o talleres en áreas artísticas y culturales en los últimos 12 meses?': 'tomo_cursos_artisticos',
        'En los últimos 12 meses, ¿hizo alguna práctica cultural?': 'hizo_practica_cultural',
        'lee libros': 'lee_libros',
        'frecuencia con que lee libros': 'frecuencia_lectura'
    }
    
    # Aplicar renombrado si las columnas existen
    for old_col, new_col in columnas_rename.items():
        if old_col in df_limpio.columns:
            df_limpio = df_limpio.rename(columns={old_col: new_col})
    
    # Manejo de valores nulos
    # Para variables numéricas, usamos la mediana
    numeric_cols = df_limpio.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df_limpio[col].isnull().sum() > 0:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
    
    # Para variables categóricas, usamos el valor más frecuente
    cat_cols = df_limpio.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_limpio[col].isnull().sum() > 0:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode()[0])
    
    # Manejo de outliers en ingreso mensual (si existe)
    if 'ingreso_mensual' in df_limpio.columns:
        Q1 = df_limpio['ingreso_mensual'].quantile(0.25)
        Q3 = df_limpio['ingreso_mensual'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Winsorización (recortar valores extremos)
        df_limpio['ingreso_mensual'] = np.where(
            df_limpio['ingreso_mensual'] > upper_bound,
            upper_bound,
            np.where(
                df_limpio['ingreso_mensual'] < lower_bound,
                lower_bound,
                df_limpio['ingreso_mensual']
            )
        )
    
    # Convertir variables de frecuencia a numéricas si es posible
    frecuencia_cols = [col for col in df_limpio.columns if 'frecuencia' in col]
    
    # Mapeo para convertir frecuencias textuales a numéricas
    mapeo_frecuencia = {
        'Nunca': 0,
        'Una vez al año': 1,
        'Una vez cada seis meses': 2,
        'Una vez cada tres meses': 3,
        'Una vez al mes': 4,
        'Una vez cada quince días': 5,
        'Una vez a la semana': 6,
        'Más de una vez a la semana': 7
    }
    
    for col in frecuencia_cols:
        if col in df_limpio.columns:
            if df_limpio[col].dtype == 'object':
                df_limpio[col] = df_limpio[col].map(mapeo_frecuencia)
                df_limpio[col] = df_limpio[col].fillna(0)  # Si hay valores que no mapean, asumimos 0
    
    return df_limpio

# Función para crear nuevas características (ingeniería de características)
def ingenieria_caracteristicas(df):
    """
    Crea nuevas variables a partir de las existentes para enriquecer el análisis
    """
    if df.empty:
        return df
    
    df_nuevo = df.copy()
    
    # Categorías de edad
    if 'edad' in df_nuevo.columns:
        bins = [0, 12, 18, 30, 45, 60, 100]
        labels = ['Niño', 'Adolescente', 'Joven adulto', 'Adulto', 'Adulto mayor', 'Tercera edad']
        df_nuevo['grupo_edad'] = pd.cut(df_nuevo['edad'], bins=bins, labels=labels, right=False)
    
    # Índice de participación cultural
    columnas_asistencia = [col for col in df_nuevo.columns if col.startswith('asistio_') or col.startswith('visito_')]
    if columnas_asistencia:
        # Convertir a binario si no lo están (1 si asistió, 0 si no)
        for col in columnas_asistencia:
            if col in df_nuevo.columns:
                # Intentar convertir a valores binarios si son strings
                if df_nuevo[col].dtype == 'object':
                    df_nuevo[col] = df_nuevo[col].map({'Sí': 1, 'No': 0})
                    df_nuevo[col] = df_nuevo[col].fillna(0)
        
        # Crear índice sumando todas las asistencias
        df_nuevo['indice_participacion_cultural'] = df_nuevo[columnas_asistencia].sum(axis=1)
        
        # Categorizar el índice
        bins_participacion = [-1, 0, 2, 5, 100]
        labels_participacion = ['Nula', 'Baja', 'Media', 'Alta']
        df_nuevo['nivel_participacion_cultural'] = pd.cut(
            df_nuevo['indice_participacion_cultural'], 
            bins=bins_participacion, 
            labels=labels_participacion
        )
    
    # Nivel socioeconómico aproximado (basado en ingresos si existe la columna)
    if 'ingreso_mensual' in df_nuevo.columns:
        # Definir rangos para Colombia (aproximados en pesos colombianos)
        bins_ingresos = [-1, 1000000, 2500000, 5000000, float('inf')]
        labels_ingresos = ['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto']
        df_nuevo['nivel_socioeconomico'] = pd.cut(
            df_nuevo['ingreso_mensual'], 
            bins=bins_ingresos, 
            labels=labels_ingresos
        )
    
    # Índice de consumo lector
    if 'lee_libros' in df_nuevo.columns and 'frecuencia_lectura' in df_nuevo.columns:
        # Convertir a binario si no lo están
        if df_nuevo['lee_libros'].dtype == 'object':
            df_nuevo['lee_libros'] = df_nuevo['lee_libros'].map({'Sí': 1, 'No': 0})
            df_nuevo['lee_libros'] = df_nuevo['lee_libros'].fillna(0)
        
        # Normalizar frecuencia entre 0 y 1
        max_freq = df_nuevo['frecuencia_lectura'].max()
        if max_freq > 0:
            df_nuevo['frecuencia_lectura_norm'] = df_nuevo['frecuencia_lectura'] / max_freq
        else:
            df_nuevo['frecuencia_lectura_norm'] = 0
        
        # Índice combinado: si lee * frecuencia normalizada
        df_nuevo['indice_lector'] = df_nuevo['lee_libros'] * df_nuevo['frecuencia_lectura_norm']
        
        # Categorizar
        bins_lectura = [-0.001, 0, 0.33, 0.66, 1.01]
        labels_lectura = ['No lector', 'Lector ocasional', 'Lector regular', 'Lector frecuente']
        df_nuevo['perfil_lector'] = pd.cut(
            df_nuevo['indice_lector'], 
            bins=bins_lectura, 
            labels=labels_lectura
        )
    
    # Diversidad cultural (cantidad de diferentes tipos de actividades culturales)
    df_nuevo['diversidad_cultural'] = df_nuevo[columnas_asistencia].sum(axis=1)
    
    return df_nuevo

# Función para crear visualizaciones
def crear_visualizaciones(df):
    """
    Crea visualizaciones para el análisis exploratorio
    """
    if df.empty:
        st.warning("No hay datos para visualizar")
        return
    
    # Distribución de edad
    if 'edad' in df.columns:
        fig_edad = px.histogram(
            df, 
            x='edad', 
            color='sexo' if 'sexo' in df.columns else None,
            title='Distribución de Edades',
            labels={'edad': 'Edad', 'count': 'Cantidad de personas'},
            opacity=0.7,
            marginal='box'
        )
        return fig_edad
    else:
        return None

# Función para mostrar estadísticas descriptivas
def mostrar_estadisticas(df):
    """
    Muestra estadísticas descriptivas del dataset
    """
    if df.empty:
        st.warning("No hay datos para mostrar estadísticas")
        return
    
    st.write("### Estadísticas Descriptivas")
    
    # Seleccionar solo columnas numéricas para las estadísticas
    num_cols = df.select_dtypes(include=['number']).columns
    if not num_cols.empty:
        stats = df[num_cols].describe()
        st.dataframe(stats)
    else:
        st.info("No hay columnas numéricas para mostrar estadísticas")

# Función principal de la aplicación
def main():
    """
    Función principal que ejecuta la aplicación Streamlit
    """
    # Título principal
    st.title("📊 Análisis de Hábitos Culturales en Colombia")
    st.markdown("""
    Esta aplicación analiza los datos de la encuesta del DANE sobre consumo cultural en Colombia,
    permitiendo explorar patrones de participación en actividades culturales según variables sociodemográficas.
    """)
    
    # Sidebar con controles
    st.sidebar.title("Controles")
    
    # Cargar datos
    datos_originales = cargar_datos()
    
    if datos_originales.empty:
        st.info("Por favor, sube un archivo CSV para comenzar el análisis")
        return
    
    # Mostrar datos originales si se desea
    if st.sidebar.checkbox("Mostrar datos originales"):
        st.subheader("Datos originales")
        st.dataframe(datos_originales.head(10))
    
    # Limpieza de datos
    datos_limpios = limpiar_datos(datos_originales)
    
    # Ingeniería de características
    datos_procesados = ingenieria_caracteristicas(datos_limpios)
    
    # Menú de pestañas para organizar el contenido
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen General", 
        "👥 Perfil Sociodemográfico", 
        "🎭 Participación Cultural", 
        "📚 Hábitos de Lectura",
        "⚙️ Análisis Avanzado"
    ])
    
    # Tab 1: Resumen General
    with tab1:
        st.header("Resumen General de los Datos")
        
        # Métricas clave
        col1, col2, col3, col4 = st.columns(4)
        
        # Calcular métricas
        total_encuestados = len(datos_procesados)
        
        # Participación cultural (si existe la columna)
        participacion_cultural = 0
        if 'nivel_participacion_cultural' in datos_procesados.columns:
            participacion_cultural = (datos_procesados['nivel_participacion_cultural'] != 'Nula').mean() * 100
        
        # Porcentaje de lectores (si existe la columna)
        porcentaje_lectores = 0
        if 'lee_libros' in datos_procesados.columns:
            if datos_procesados['lee_libros'].dtype == 'object':
                porcentaje_lectores = (datos_procesados['lee_libros'] == 'Sí').mean() * 100
            else:
                porcentaje_lectores = (datos_procesados['lee_libros'] == 1).mean() * 100
        
        # Promedio de actividades culturales
        promedio_actividades = 0
        if 'indice_participacion_cultural' in datos_procesados.columns:
            promedio_actividades = datos_procesados['indice_participacion_cultural'].mean()
        
        with col1:
            st.metric("Total Encuestados", f"{total_encuestados:,}")
        
        with col2:
            st.metric("Participación Cultural", f"{participacion_cultural:.1f}%")
        
        with col3:
            st.metric("Lectores", f"{porcentaje_lectores:.1f}%")
        
        with col4:
            st.metric("Promedio Actividades", f"{promedio_actividades:.1f}")
        
        # Estadísticas descriptivas
        mostrar_estadisticas(datos_procesados)
        
        # Visualización general
        st.subheader("Distribución de la Población Encuestada")
        fig_edad = crear_visualizaciones(datos_procesados)
        if fig_edad:
            st.plotly_chart(fig_edad, use_container_width=True)
    
    # Tab 2: Perfil Sociodemográfico
    with tab2:
        st.header("Perfil Sociodemográfico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'sexo' in datos_procesados.columns:
                fig_sexo = px.pie(
                    datos_procesados, 
                    names='sexo',
                    title='Distribución por Sexo',
                    hole=0.4
                )
                st.plotly_chart(fig_sexo, use_container_width=True)
            
            if 'grupo_edad' in datos_procesados.columns:
                fig_grupo_edad = px.bar(
                    datos_procesados['grupo_edad'].value_counts().reset_index(),
                    x='grupo_edad',
                    y='count',
                    title='Distribución por Grupo de Edad',
                    labels={'count': 'Cantidad', 'grupo_edad': 'Grupo de Edad'}
                )
                st.plotly_chart(fig_grupo_edad, use_container_width=True)
        
        with col2:
            if 'nivel_educativo' in datos_procesados.columns:
                fig_educacion = px.bar(
                    datos_procesados['nivel_educativo'].value_counts().reset_index(),
                    x='count',
                    y='nivel_educativo',
                    title='Nivel Educativo',
                    labels={'count': 'Cantidad', 'nivel_educativo': 'Nivel Educativo'},
                    orientation='h'
                )
                st.plotly_chart(fig_educacion, use_container_width=True)
            
            if 'nivel_socioeconomico' in datos_procesados.columns:
                fig_nse = px.pie(
                    datos_procesados, 
                    names='nivel_socioeconomico',
                    title='Nivel Socioeconómico',
                    hole=0.4
                )
                st.plotly_chart(fig_nse, use_container_width=True)
        
        # Visualización adicional: Educación por sexo
        if 'nivel_educativo' in datos_procesados.columns and 'sexo' in datos_procesados.columns:
            st.subheader("Nivel Educativo por Sexo")
            fig_edu_sexo = px.histogram(
                datos_procesados,
                x='nivel_educativo',
                color='sexo',
                barmode='group',
                title='Distribución de Nivel Educativo por Sexo',
                labels={'nivel_educativo': 'Nivel Educativo', 'count': 'Cantidad'}
            )
            st.plotly_chart(fig_edu_sexo, use_container_width=True)
    
    # Tab 3: Participación Cultural
    with tab3:
        st.header("Participación en Actividades Culturales")
        
        # Filtros interactivos
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'sexo' in datos_procesados.columns:
                sexo_select = st.multiselect(
                    "Filtrar por Sexo",
                    options=datos_procesados['sexo'].unique(),
                    default=datos_procesados['sexo'].unique()
                )
            else:
                sexo_select = None
        
        with col2:
            if 'grupo_edad' in datos_procesados.columns:
                edad_select = st.multiselect(
                    "Filtrar por Grupo de Edad",
                    options=datos_procesados['grupo_edad'].unique(),
                    default=datos_procesados['grupo_edad'].unique()
                )
            else:
                edad_select = None
                
        with col3:
            if 'nivel_educativo' in datos_procesados.columns:
                edu_select = st.multiselect(
                    "Filtrar por Nivel Educativo",
                    options=datos_procesados['nivel_educativo'].unique(),
                    default=datos_procesados['nivel_educativo'].unique()
                )
            else:
                edu_select = None
        
        # Aplicar filtros
        df_filtrado = datos_procesados.copy()
        
        if sexo_select:
            df_filtrado = df_filtrado[df_filtrado['sexo'].isin(sexo_select)]
        
        if edad_select:
            df_filtrado = df_filtrado[df_filtrado['grupo_edad'].isin(edad_select)]
            
        if edu_select:
            df_filtrado = df_filtrado[df_filtrado['nivel_educativo'].isin(edu_select)]
        
        # Visualización de participación cultural
        columnas_asistencia = [col for col in df_filtrado.columns if col.startswith('asistio_') or col.startswith('visito_')]
        
        if columnas_asistencia:
            # Preparar datos para la visualización
            asistencia_data = []
            
            for col in columnas_asistencia:
                if col in df_filtrado.columns:
                    # Obtener nombre limpio para la actividad
                    nombre_actividad = col.replace('asistio_', '').replace('visito_', '').replace('_', ' ').title()
                    
                    # Calcular porcentaje de asistencia
                    if df_filtrado[col].dtype == 'object':
                        porcentaje = (df_filtrado[col] == 'Sí').mean() * 100
                    else:
                        porcentaje = (df_filtrado[col] == 1).mean() * 100
                    
                    asistencia_data.append({
                        'Actividad': nombre_actividad,
                        'Porcentaje': porcentaje
                    })
            
            # Crear DataFrame para visualización
            df_asistencia = pd.DataFrame(asistencia_data)
            
            if not df_asistencia.empty:
                # Ordenar por porcentaje
                df_asistencia = df_asistencia.sort_values('Porcentaje', ascending=False)
                
                # Crear gráfico de barras
                fig_asistencia = px.bar(
                    df_asistencia,
                    x='Actividad',
                    y='Porcentaje',
                    title='Porcentaje de Asistencia por Tipo de Actividad Cultural',
                    labels={'Porcentaje': 'Porcentaje (%)', 'Actividad': 'Tipo de Actividad'},
                    text_auto='.1f'
                )
                fig_asistencia.update_traces(texttemplate='%{text}%', textposition='outside')
                fig_asistencia.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                
                st.plotly_chart(fig_asistencia, use_container_width=True)
                
                # Heatmap de correlación entre actividades
                st.subheader("Correlación entre Actividades Culturales")
                
                # Crear matriz de correlación
                corr_matrix = df_filtrado[columnas_asistencia].corr()
                
                # Gráfico de correlación
                fig_corr = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlación"),
                    x=[col.replace('asistio_', '').replace('visito_', '').replace('_', ' ').title() for col in corr_matrix.columns],
                    y=[col.replace('asistio_', '').replace('visito_', '').replace('_', ' ').title() for col in corr_matrix.columns],
                    title="Correlación entre Actividades Culturales"
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Analisis por variables sociodemográficas
                st.subheader("Participación Cultural por Variables Sociodemográficas")
                
                # Selector de actividad específica
                actividad_seleccionada = st.selectbox(
                    "Selecciona una actividad cultural",
                    options=[col.replace('asistio_', '').replace('visito_', '').replace('_', ' ').title() for col in columnas_asistencia]
                )
                
                # Obtener columna original
                col_original = [col for col in columnas_asistencia if col.replace('asistio_', '').replace('visito_', '').replace('_', ' ').title() == actividad_seleccionada][0]
                
                # Gráficos por variables sociodemográficas
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'sexo' in df_filtrado.columns:
                        # Crear gráfico para sexo
                        temp_df = df_filtrado.groupby('sexo')[col_original].mean().reset_index()
                        temp_df[col_original] = temp_df[col_original] * 100  # Convertir a porcentaje
                        
                        fig_sexo = px.bar(
                            temp_df,
                            x='sexo',
                            y=col_original,
                            title=f'Asistencia a {actividad_seleccionada} por Sexo',
                            labels={col_original: 'Porcentaje (%)', 'sexo': 'Sexo'},
                            text_auto='.1f'
                        )
                        fig_sexo.update_traces(texttemplate='%{text}%', textposition='outside')
                        
                        st.plotly_chart(fig_sexo, use_container_width=True)
                
                with col2:
                    if 'grupo_edad' in df_filtrado.columns:
                        # Crear gráfico para grupo de edad
                        temp_df = df_filtrado.groupby('grupo_edad')[col_original].mean().reset_index()
                        temp_df[col_original] = temp_df[col_original] * 100  # Convertir a porcentaje
                        
                        fig_edad = px.bar(
                            temp_df,
                            x='grupo_edad',
                            y=col_original,
                            title=f'Asistencia a {actividad_seleccionada} por Grupo de Edad',
                            labels={col_original: 'Porcentaje (%)', 'grupo_edad': 'Grupo de Edad'},
                            text_auto='.1f'
                        )
                        fig_edad.update_traces(texttemplate='%{text}%', textposition='outside')
                        
                        st.plotly_chart(fig_edad, use_container_width=True)
                
                # Visualización adicional por nivel educativo
                if 'nivel_educativo' in df_filtrado.columns:
                    # Crear gráfico para nivel educativo
                    temp_df = df_filtrado.groupby('nivel_educativo')[col_original].mean().reset_index()
                    temp_df[col_original] = temp_df[col_original] * 100  # Convertir a porcentaje
                    
                    fig_edu = px