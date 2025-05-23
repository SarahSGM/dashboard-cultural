import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io

# Configuración de la página
st.set_page_config(
    page_title="🎭 Dashboard Cultural",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para el tema morado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #6a0dad, #9370db);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f3ff, #ede9fe);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #6a0dad;
        margin: 1rem 0;
    }
    .section-header {
        background: linear-gradient(90deg, #9370db, #dda0dd);
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f4ff, #f3e8ff);
    }
    .stSelectbox > div > div {
        background-color: #f8f4ff;
    }
    .plot-container {
        border: 2px solid #dda0dd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# Función para crear paleta de colores morada
def get_purple_palette(n_colors):
    """Genera una paleta de colores morados"""
    base_colors = ['#6a0dad', '#8a2be2', '#9370db', '#9932cc', '#ba55d3', 
                  '#da70d6', '#dda0dd', '#e6e6fa', '#f8f4ff']
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    else:
        # Si necesitamos más colores, interpolar
        return px.colors.sample_colorscale('Purples', n_colors)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🎭 Dashboard de Análisis Cultural</h1>
    <p>Análisis Estadístico de Participación Cultural en Colombia</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.markdown("## 🎨 Panel de Control")
page = st.sidebar.selectbox(
    "Selecciona una sección:",
    ["📊 Resumen Ejecutivo", "🔍 Análisis Demográfico", "🎪 Participación Cultural", 
     "📚 Actividades Específicas", "📈 Análisis Avanzado", "🔄 Comparativas"]
)

# Función para cargar y procesar datos
@st.cache_data
def load_and_process_data():
    """Carga y procesa los datos culturales"""
    # Aquí podrías cargar el archivo Excel directamente
    try:
        data = pd.read_excel('cultura.xlsx')
        df = pd.DataFrame(data)
        
        # Crear grupos de edad
        # Asegurarse que EDAD sea numérico
        df['EDAD'] = pd.to_numeric(df['EDAD'], errors='coerce')
        df['grupo_edad'] = pd.cut(df['EDAD'], bins=[0, 12, 18, 28, 40, 60, 100],
                          labels=["Niñez", "Adolescencia", "Juventud", "Adultez temprana", "Adultez media", "Adulto mayor"])
        
        # Crear grupos de ingreso - Corregido para manejar valores no numéricos
        # Primero, asegurarse que P2 sea numérico
        df['P2'] = pd.to_numeric(df['P2'], errors='coerce')
        
        # Manejar NaNs para evitar errores en qcut
        if not df['P2'].isna().all():  # Si hay al menos algunos valores válidos
            # Usar solo valores no-NaN para calcular los cuartiles
            valid_incomes = df['P2'].dropna()
            # Crear labels para los cuartiles
            labels = ['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto']
            
            if len(valid_incomes) > 0 and len(valid_incomes.unique()) >= 4:
                # Si hay suficientes valores únicos para crear 4 cuartiles
                df['grupo_ingreso'] = pd.qcut(valid_incomes, q=4, labels=labels)
            else:
                # Si no hay suficientes valores únicos, usar cut con valores fijos
                max_val = valid_incomes.max() if len(valid_incomes) > 0 else 0
                bins = [0, max_val/4, max_val/2, 3*max_val/4, max_val]
                df['grupo_ingreso'] = pd.cut(df['P2'], bins=bins, labels=labels, include_lowest=True)
        else:
            # Si todos son NaN, crear una columna de NaN
            df['grupo_ingreso'] = np.nan
        
        # Convertir variables SI/NO a numéricas para facilitar análisis
        for col in df.columns:
            if df[col].dtype == 'object':  # Solo para columnas de tipo objeto
                # Crear una columna numérica para cada variable SI/NO
                if set(df[col].unique()).issubset({'SI', 'NO', np.nan}):
                    df[f'{col.lower()}_num'] = df[col].map({'SI': 1, 'NO': 0})
        
        # Crear índice de participación cultural
        cultural_vars = ['P3', 'P4', 'P5', 'ASISTENCIA BIBLIOTECA',
                        'ASISTENCIA CASAS DE CULTURA', 'ASISTENCIA CENTROS CUTURALES',
                        'ASISTENCIA MUSEOS', 'ASISTENCIA EXPOSICIONES',
                        'ASISTENCIA MONUMENTOS', 'ASISTENCIA CURSOS',
                        'PRACTICA CULTURAL', 'LECTURA LIBROS']
        
        # Convertir las variables culturales a numéricas para el índice
        cultural_vars_numeric = []
        for var in cultural_vars:
            col_name = f'{var.lower()}_num' if var in df.columns else var
            if col_name in df.columns:
                cultural_vars_numeric.append(col_name)
            elif var in df.columns:
                # Si no existe la versión numérica, crear una temporal para el cálculo
                df[f'{var}_temp'] = df[var].map({'SI': 1, 'NO': 0})
                cultural_vars_numeric.append(f'{var}_temp')
        
        # Calcular el índice usando las variables numéricas
        df['indice_cultural'] = df[cultural_vars_numeric].sum(axis=1)
        df['nivel_participacion'] = pd.cut(df['indice_cultural'], 
                                          bins=[-1, 2, 5, 8, 12], 
                                          labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])
        
        # Eliminar columnas temporales si se crearon
        temp_cols = [col for col in df.columns if col.endswith('_temp')]
        df = df.drop(columns=temp_cols, errors='ignore')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        # Proporcionar un DataFrame vacío o con datos de muestra para evitar errores
        return pd.DataFrame()

# Cargar datos
df = load_and_process_data()

# Verificar si hay datos cargados
if df.empty:
    st.error("No se pudieron cargar los datos. Por favor, verifica el archivo 'cultura.xlsx'.")
    st.stop()

# PÁGINA: RESUMEN EJECUTIVO
if page == "📊 Resumen Ejecutivo":
    st.markdown('<div class="section-header"><h2>📊 Resumen Ejecutivo</h2></div>', unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Participantes", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        participation_rate = (df['indice_cultural'] > 3).mean() * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Participación Cultural", f"{participation_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_age = df['EDAD'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Edad Promedio", f"{avg_age:.1f} años")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        reading_rate = (df['LECTURA LIBROS'] == 'SI').mean() * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Hábito de Lectura", f"{reading_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("🎭 Distribución por Nivel de Participación Cultural")
        participation_counts = df['nivel_participacion'].value_counts()
        colors = get_purple_palette(len(participation_counts))
        
        fig = px.pie(values=participation_counts.values, 
                    names=participation_counts.index,
                    color_discrete_sequence=colors,
                    hole=0.4)
        fig.update_layout(font=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("👥 Distribución por Género y Edad")
        # Asegurarse de que hay datos válidos para el gráfico
        if not df['grupo_edad'].isna().all() and not df['SEXO'].isna().all():
            fig = px.histogram(df, x='grupo_edad', color='SEXO',
                            barmode='group',
                            color_discrete_sequence=['#6a0dad', '#ba55d3'],
                            title="")
            fig.update_layout(
                xaxis_title="Grupo de Edad",
                yaxis_title="Número de Personas",
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para mostrar la distribución por género y edad.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Top actividades culturales
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🏆 Top 10 Actividades Culturales Más Populares")
    
    activities = {
        'Lectura de Libros': (df['LECTURA LIBROS'] == 'SI').mean(),
        'Práctica Cultural': (df['PRACTICA CULTURAL'] == 'SI').mean(),
        'Asistencia a Bibliotecas': (df['ASISTENCIA BIBLIOTECA'] == 'SI').mean(),
        'Conciertos/Música en Vivo': (df['P4'] == 'SI').mean(),
        'Monumentos Históricos': (df['ASISTENCIA MONUMENTOS'] == 'SI').mean(),
        'Teatro/Ópera/Danza': (df['P3'] == 'SI').mean(),
        'Centros Culturales': (df['ASISTENCIA CENTROS CUTURALES'] == 'SI').mean(),
        'Casas de Cultura': (df['ASISTENCIA CASAS DE CULTURA'] == 'SI').mean(),
        'Cursos/Talleres': (df['ASISTENCIA CURSOS'] == 'SI').mean(),
        'Museos': (df['ASISTENCIA MUSEOS'] == 'SI').mean()
    }
    
    activities_df = pd.DataFrame(list(activities.items()), columns=['Actividad', 'Porcentaje'])
    activities_df = activities_df.sort_values('Porcentaje', ascending=True)
    activities_df['Porcentaje'] *= 100
    
    fig = px.bar(activities_df, x='Porcentaje', y='Actividad',
                orientation='h',
                color='Porcentaje',
                color_continuous_scale='Purples',
                title="")
    fig.update_layout(
        xaxis_title="Porcentaje de Participación (%)",
        yaxis_title="",
        font=dict(size=12),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# PÁGINA: ANÁLISIS DEMOGRÁFICO
elif page == "🔍 Análisis Demográfico":
    st.markdown('<div class="section-header"><h2>🔍 Análisis Demográfico</h2></div>', unsafe_allow_html=True)
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_options = ['Todos'] + sorted(df['SEXO'].dropna().unique().tolist())
        selected_gender = st.selectbox("Filtrar por Género:", options=gender_options)
    
    with col2:
        education_options = ['Todos'] + sorted(df['NIVEL EDUCATIVO'].dropna().unique().tolist())
        selected_education = st.selectbox("Filtrar por Educación:", options=education_options)
    
    with col3:
        ethnicity_options = ['Todos'] + sorted(df['ETNIA'].dropna().unique().tolist())
        selected_ethnicity = st.selectbox("Filtrar por Etnia:", options=ethnicity_options)
    
    # Aplicar filtros
    filtered_df = df.copy()
    if selected_gender != 'Todos':
        filtered_df = filtered_df[filtered_df['SEXO'] == selected_gender]
    if selected_education != 'Todos':
        filtered_df = filtered_df[filtered_df['NIVEL EDUCATIVO'] == selected_education]
    if selected_ethnicity != 'Todos':
        filtered_df = filtered_df[filtered_df['ETNIA'] == selected_ethnicity]
    
    # Pirámide poblacional
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("👥 Pirámide Poblacional")
        
        # Verificar datos para la pirámide
        has_men = 'HOMBRE' in filtered_df['SEXO'].values and not filtered_df[filtered_df['SEXO'] == 'HOMBRE']['grupo_edad'].isna().all()
        has_women = 'MUJER' in filtered_df['SEXO'].values and not filtered_df[filtered_df['SEXO'] == 'MUJER']['grupo_edad'].isna().all()
        
        if has_men or has_women:
            # Crear datos para pirámide
            men_data = filtered_df[filtered_df['SEXO'] == 'HOMBRE']['grupo_edad'].value_counts().sort_index() if has_men else pd.Series()
            women_data = filtered_df[filtered_df['SEXO'] == 'MUJER']['grupo_edad'].value_counts().sort_index() if has_women else pd.Series()
            
            fig = go.Figure()
            
            if has_men:
                fig.add_trace(go.Bar(
                    y=men_data.index,
                    x=-men_data.values,
                    name='Hombres',
                    orientation='h',
                    marker_color='#6a0dad'
                ))
            
            if has_women:
                fig.add_trace(go.Bar(
                    y=women_data.index,
                    x=women_data.values,
                    name='Mujeres',
                    orientation='h',
                    marker_color='#ba55d3'
                ))
            
            fig.update_layout(
                barmode='relative',
                bargap=0.1,
                xaxis_title="Número de Personas",
                yaxis_title="Grupo de Edad",
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para mostrar la pirámide poblacional.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("🎓 Distribución por Nivel Educativo")
        
        if not filtered_df['NIVEL EDUCATIVO'].isna().all():
            education_counts = filtered_df['NIVEL EDUCATIVO'].value_counts()
            colors = get_purple_palette(len(education_counts))
            
            fig = px.donut(values=education_counts.values,
                        names=education_counts.index,
                        color_discrete_sequence=colors)
            fig.update_layout(font=dict(size=12))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos de nivel educativo para mostrar.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis de ingresos solo si hay datos de P2 válidos
    if not filtered_df['P2'].isna().all():
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("💰 Análisis de Ingresos por Características Demográficas")
        
        # Verificar si hay suficientes datos para cada grupo
        valid_genders = [gender for gender in filtered_df['SEXO'].unique() if not pd.isna(gender) and not filtered_df[filtered_df['SEXO'] == gender]['P2'].isna().all()]
        valid_education = [edu for edu in filtered_df['NIVEL EDUCATIVO'].unique() if not pd.isna(edu) and not filtered_df[filtered_df['NIVEL EDUCATIVO'] == edu]['P2'].isna().all()]
        valid_age_groups = [age for age in filtered_df['grupo_edad'].unique() if not pd.isna(age) and not filtered_df[filtered_df['grupo_edad'] == age]['P2'].isna().all()]
        
        if valid_genders or valid_education or valid_age_groups:
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Por Género', 'Por Nivel Educativo', 'Por Grupo de Edad'),
                specs=[[{"type": "box"}, {"type": "box"}, {"type": "box"}]]
            )
            
            # Box plot por género
            if valid_genders:
                for i, gender in enumerate(valid_genders):
                    data = filtered_df[filtered_df['SEXO'] == gender]['P2'].dropna()
                    if not data.empty:
                        fig.add_trace(
                            go.Box(y=data, name=gender, marker_color=get_purple_palette(len(valid_genders))[i]),
                            row=1, col=1
                        )
            
            # Box plot por educación
            if valid_education:
                for i, edu in enumerate(valid_education):
                    data = filtered_df[filtered_df['NIVEL EDUCATIVO'] == edu]['P2'].dropna()
                    if not data.empty:
                        fig.add_trace(
                            go.Box(y=data, name=edu, marker_color=get_purple_palette(len(valid_education))[i]),
                            row=1, col=2
                        )
            
            # Box plot por edad
            if valid_age_groups:
                for i, age in enumerate(valid_age_groups):
                    data = filtered_df[filtered_df['grupo_edad'] == age]['P2'].dropna()
                    if not data.empty:
                        fig.add_trace(
                            go.Box(y=data, name=age, marker_color=get_purple_palette(len(valid_age_groups))[i]),
                            row=1, col=3
                        )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                font=dict(size=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos de ingresos para generar las gráficas.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No hay datos de ingresos disponibles para el análisis.")

# PÁGINA: PARTICIPACIÓN CULTURAL
elif page == "🎪 Participación Cultural":
    st.markdown('<div class="section-header"><h2>🎪 Análisis de Participación Cultural</h2></div>', unsafe_allow_html=True)
    
    # Mapa de calor de participación
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🔥 Mapa de Calor: Participación Cultural por Demografía")
    
    # Verificar si hay datos suficientes para el mapa de calor
    if not df['grupo_edad'].isna().all() and not df['NIVEL EDUCATIVO'].isna().all() and not df['indice_cultural'].isna().all():
        # Crear matriz de participación
        participation_matrix = df.groupby(['grupo_edad', 'NIVEL EDUCATIVO'])['indice_cultural'].mean().reset_index()
        
        # Verificar si hay suficientes datos después del groupby
        if not participation_matrix.empty:
            participation_pivot = participation_matrix.pivot(
                index='grupo_edad', 
                columns='NIVEL EDUCATIVO', 
                values='indice_cultural'
            )
            
            fig = px.imshow(participation_pivot,
                        color_continuous_scale='Purples',
                        aspect="auto",
                        title="")
            fig.update_layout(
                xaxis_title="Nivel Educativo",
                yaxis_title="Grupo de Edad",
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para generar el mapa de calor.")
    else:
        st.warning("Faltan datos necesarios para el mapa de calor de participación cultural.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis por género
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("♂️♀️ Participación Cultural por Género")
        
        cultural_activities = {
            'Teatro/Danza': 'P3',
            'Conciertos': 'P4',
            'Música en Bares': 'P5',
            'Bibliotecas': 'ASISTENCIA BIBLIOTECA',
            'Casas de Cultura': 'ASISTENCIA CASAS DE CULTURA',  # Corregido el nombre de la columna
            'Centros Culturales': 'ASISTENCIA CENTROS CUTURALES',
            'Museos': 'ASISTENCIA MUSEOS',
            'Exposiciones': 'ASISTENCIA EXPOSICIONES',
            'Monumentos': 'ASISTENCIA MONUMENTOS',
            'Cursos': 'ASISTENCIA CURSOS'
        }
        
        # Verificar que hay datos de género
        if not df['SEXO'].isna().all():
            # Calcular participación por género para actividades existentes
            gender_participation = {}
            for activity, col_name in cultural_activities.items():
                if col_name in df.columns:
                    gender_participation[activity] = df.groupby('SEXO')[col_name].apply(
                        lambda x: (x == 'SI').mean() * 100 if 'SI' in x.values else 0
                    )
            
            if gender_participation:
                gender_df = pd.DataFrame(gender_participation).T
                if not gender_df.empty:
                    fig = px.bar(gender_df, 
                                barmode='group',
                                color_discrete_sequence=['#6a0dad', '#ba55d3'],
                                title="")
                    fig.update_layout(
                        xaxis_title="Actividades Culturales",
                        yaxis_title="Porcentaje de Participación (%)",
                        font=dict(size=11),
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay suficientes datos para mostrar la participación por género.")
            else:
                st.warning("No se encontraron actividades culturales en los datos.")
        else:
            st.warning("No hay datos de género disponibles.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("🌈 Participación por Etnia")
        
        # Verificar si hay datos de etnia y participación cultural
        if not df['ETNIA'].isna().all() and not df['indice_cultural'].isna().all():
            ethnicity_participation = df.groupby('ETNIA')['indice_cultural'].mean().sort_values(ascending=True)
            
            if not ethnicity_participation.empty:
                colors = get_purple_palette(len(ethnicity_participation))
                
                fig = px.bar(x=ethnicity_participation.values,
                            y=ethnicity_participation.index,
                            orientation='h',
                            color=ethnicity_participation.values,
                            color_continuous_scale='Purples',
                            title="")
                fig.update_layout(
                    xaxis_title="Índice de Participación Cultural Promedio",
                    yaxis_title="Etnia",
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay suficientes datos para mostrar la participación por etnia.")
        else:
            st.warning("Faltan datos de etnia o participación cultural.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis de correlaciones
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🔗 Matriz de Correlaciones: Actividades Culturales")
    
    # Seleccionar solo variables numéricas de actividades culturales
    cultural_numeric_vars = [col for col in df.columns if col.lower().endswith('_num') and 
                          ('asistencia' in col.lower() or 'p3' in col.lower() or 
                           'p4' in col.lower() or 'p5' in col.lower())]
    
    # Verificar que hay suficientes variables para la correlación
    if len(cultural_numeric_vars) >= 2:
        # Asegurarse de que hay datos suficientes
        valid_data = df[cultural_numeric_vars].dropna()
        
        if not valid_data.empty and valid_data.shape[0] > 1:
            correlation_matrix = valid_data.corr()
            
            # Crear nombres más legibles
            new_names = {
                'p3_num': 'Teatro/Danza',
                'p4_num': 'Conciertos',
                'p5_num': 'Música en Bares',
                'asistencia biblioteca_num': 'Bibliotecas',
                'asistencia casas de cultura_num': 'Casas de Cultura',
                'asistencia centros cuturales_num': 'Centros Culturales',
                'asistencia museos_num': 'Museos',
                'asistencia exposiciones_num': 'Exposiciones',
                'asistencia monumentos_num': 'Monumentos',
                'asistencia cursos_num': 'Cursos',
                'practica cultural_num': 'Práctica Cultural',
                'lectura libros_num': 'Lectura'
            }
            
            correlation_matrix.index = [new_names.get(idx.lower(), idx) for idx in correlation_matrix.index]
            correlation_matrix.columns = [new_names.get(col.lower(), col) for col in correlation_matrix.columns]
            
            fig = px.imshow(correlation_matrix,
                   color_continuous_scale='Purples',
                   aspect="auto",
                   title="")
    fig.update_layout(
        width=800,
        height=600,
        font=dict(size=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# PÁGINA: ACTIVIDADES ESPECÍFICAS
elif page == "📚 Actividades Específicas":
    st.markdown('<div class="section-header"><h2>📚 Análisis de Actividades Específicas</h2></div>', unsafe_allow_html=True)
    
    # Selector de actividad
    activities_dict = {
        'Teatro, Ópera y Danza': 'P3',
        'Conciertos y Recitales': 'P4',
        'Música en Bares/Restaurantes': 'P5',
        'Bibliotecas': 'ASISTENCIA BIBLIOTECA',
        'Casas de Cultura': 'ASISTENCIA CASAS DE CULTURA',
        'Centros Culturales': 'ASISTENCIA CENTROS CUTURALES',
        'Museos': 'ASISTENCIA MUSEOS',
        'Exposiciones y Galerías': 'ASISTENCIA EXPOSICIONES',
        'Monumentos Históricos': 'ASISTENCIA MONUMENTOS',
        'Cursos y Talleres': 'ASISTENCIA CURSOS',
        'Lectura de Libros': 'LECTURA LIBROS'
    }
    
    selected_activity = st.selectbox("Selecciona una actividad cultural:", list(activities_dict.keys()))
    activity_col = activities_dict[selected_activity]
    
    # Análisis de la actividad seleccionada
    col1, col2, col3 = st.columns(3)
    
    with col1:
        participation_rate = (df[activity_col] == 'SI').mean() * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tasa de Participación", f"{participation_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        total_participants = (df[activity_col] == 'SI').sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Participantes", f"{total_participants:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Calcular el grupo demográfico con mayor participación
        max_group = df.groupby('grupo_edad')[activity_col].apply(lambda x: (x == 'SI').mean()).idxmax()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Grupo Más Activo", max_group)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos de análisis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader(f"📊 {selected_activity} por Grupo de Edad")
        
        age_participation = df.groupby('grupo_edad')[activity_col].apply(lambda x: (x == 'SI').mean() * 100)
        
        fig = px.bar(x=age_participation.index,
                    y=age_participation.values,
                    color=age_participation.values,
                    color_continuous_scale='Purples',
                    title="")
        fig.update_layout(
            xaxis_title="Grupo de Edad",
            yaxis_title="Porcentaje de Participación (%)",
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)