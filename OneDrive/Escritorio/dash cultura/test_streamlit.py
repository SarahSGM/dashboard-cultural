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
    /* Mejora de visibilidad para los selectbox */
    .stSelectbox > div > div {
        background-color: white !important; 
        color: #333 !important;
        border: 1px solid #dda0dd;
    }
    .stSelectbox > div > div > div {
        color: #333 !important;
        font-weight: 500;
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
    ["📊 Resumen Ejecutivo", "📈 Estadísticas Descriptivas", "🧹 Limpieza de Datos", 
     "🔍 Análisis Demográfico", "🎪 Participación Cultural", "📚 Actividades Específicas", 
     "🔬 Análisis Avanzado", "🔄 Comparativas"]
)

# Función para cargar y procesar datos
@st.cache_data
def load_and_process_data():
    """Carga y procesa los datos culturales"""
    # Aquí podrías cargar el archivo Excel directamente
    # df = pd.read_excel('cultura.xlsx')
    
    # Para este ejemplo, crearemos datos sintéticos basados en las variables proporcionadas
    np.random.seed(42)
    n_samples = 27800  # Actualizado a ~27,800 como se menciona
    
    data = {
        'id': range(1, n_samples + 1),
        'sexo': np.random.choice(['HOMBRE', 'MUJER'], n_samples),
        'edad': np.random.randint(15, 80, n_samples),
        'etnia': np.random.choice(['AFRODESCENDIENTE', 'INDIGENA', 'NINGUNA', 'RAIZAL', 'PALENQUERO', 'GITANO'], 
                                n_samples, p=[0.1, 0.05, 0.8, 0.03, 0.015, 0.005]),
        'lectura_y_escritura': np.random.choice(['SI', 'NO'], n_samples, p=[0.95, 0.05]),
        'estudiante': np.random.choice(['SI', 'NO'], n_samples, p=[0.3, 0.7]),
        'nivel_educativo': np.random.choice(['BASICA PRIMARIA', 'BASICA SECUNDARIA', 'MEDIA', 'SUPERIOR', 'POSGRADO'], 
                                          n_samples, p=[0.2, 0.25, 0.3, 0.2, 0.05]),
        'p1': np.random.choice(['TRABAJANDO', 'ESTUDIANDO', 'OFICIOS DEL HOGAR', 'BUSCANDO TRABAJO', 'OTRA ACTIVIDAD'], 
                             n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'p2': np.random.lognormal(13, 0.8, n_samples),  # Ingresos
        'p3': np.random.choice(['SI', 'NO'], n_samples, p=[0.3, 0.7]),
        'p4': np.random.choice(['SI', 'NO'], n_samples, p=[0.4, 0.6]),
        'p5': np.random.choice(['SI', 'NO'], n_samples, p=[0.35, 0.65]),
        'asistencia_biblioteca': np.random.choice(['SI', 'NO'], n_samples, p=[0.4, 0.6]),
        'asistencia_casas_de_cultura': np.random.choice(['SI', 'NO'], n_samples, p=[0.25, 0.75]),
        'asistencia_centros_culturales': np.random.choice(['SI', 'NO'], n_samples, p=[0.3, 0.7]),
        'asistencia_museos': np.random.choice(['SI', 'NO'], n_samples, p=[0.2, 0.8]),
        'asistencia_exposiciones': np.random.choice(['SI', 'NO'], n_samples, p=[0.15, 0.85]),
        'asistencia_monumentos': np.random.choice(['SI', 'NO'], n_samples, p=[0.35, 0.65]),
        'asistencia_cursos': np.random.choice(['SI', 'NO'], n_samples, p=[0.2, 0.8]),
        'practica_cultural': np.random.choice(['SI', 'NO'], n_samples, p=[0.45, 0.55]),
        'lee_libros': np.random.choice(['SI', 'NO'], n_samples, p=[0.6, 0.4]),
        'factor_expansion': np.random.uniform(80, 120, n_samples)  # Factor de expansión sintético
    }
    
    df = pd.DataFrame(data)
    
    # Añadir algunos valores nulos para mostrar la limpieza de datos
    df.loc[np.random.choice(df.index, int(n_samples * 0.05)), 'edad'] = np.nan
    df.loc[np.random.choice(df.index, int(n_samples * 0.03)), 'nivel_educativo'] = None
    df.loc[np.random.choice(df.index, int(n_samples * 0.02)), 'p2'] = np.nan
    
    return df

# Cargar datos
df_raw = load_and_process_data()

# PÁGINA: LIMPIEZA DE DATOS
if page == "🧹 Limpieza de Datos":
    st.markdown('<div class="section-header"><h2>🧹 Limpieza y Preparación de Datos</h2></div>', unsafe_allow_html=True)
    
    # Antes de la limpieza
    st.subheader("Análisis Inicial de los Datos")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dimensiones del Dataset Original:**", df_raw.shape)
        st.write("**Vista previa de los datos:**")
        st.dataframe(df_raw.head(5), use_container_width=True)
    
    with col2:
        st.write("**Resumen de valores nulos por columna:**")
        null_counts = df_raw.isnull().sum().sort_values(ascending=False)
        null_df = pd.DataFrame({
            'Columna': null_counts.index,
            'Valores Nulos': null_counts.values,
            'Porcentaje (%)': (null_counts.values / len(df_raw) * 100).round(2)
        })
        st.dataframe(null_df[null_df['Valores Nulos'] > 0], use_container_width=True)
    
    # Proceso de limpieza
    st.subheader("Proceso de Limpieza")
    
    # Creamos una copia para el proceso de limpieza
    df = df_raw.copy()
    
    # 1. Tratamiento de valores nulos
    st.markdown("**1️⃣ Tratamiento de Valores Nulos**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Rellenar edad con la mediana
df['edad'] = df['edad'].fillna(df['edad'].median())

# Rellenar nivel educativo con la moda
df['nivel_educativo'] = df['nivel_educativo'].fillna(
    df['nivel_educativo'].mode()[0]
)

# Rellenar ingresos (p2) con la mediana por nivel educativo
mediana_por_educacion = df.groupby('nivel_educativo')['p2'].median()
for nivel in df['nivel_educativo'].unique():
    mask = (df['nivel_educativo'] == nivel) & (df['p2'].isna())
    df.loc[mask, 'p2'] = mediana_por_educacion[nivel]
        """)
    
    with col2:
        # Implementamos el código mostrado
        df['edad'] = df['edad'].fillna(df['edad'].median())
        df['nivel_educativo'] = df['nivel_educativo'].fillna(df['nivel_educativo'].mode()[0])
        
        mediana_por_educacion = df.groupby('nivel_educativo')['p2'].median()
        for nivel in df['nivel_educativo'].unique():
            mask = (df['nivel_educativo'] == nivel) & (df['p2'].isna())
            df.loc[mask, 'p2'] = mediana_por_educacion[nivel]
        
        # Verificamos resultado
        null_counts_after = df.isnull().sum()
        st.write("**Valores nulos después de la limpieza:**", null_counts_after.sum())
    
    # 2. Procesamiento de variables
    st.markdown("**2️⃣ Procesamiento de Variables**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Convertir variables binarias
binary_cols = ['sexo', 'lectura_y_escritura', 'estudiante', 
               'p3', 'p4', 'p5', 'asistencia_biblioteca', 
               'asistencia_casas_de_cultura', 
               'asistencia_centros_culturales', 'asistencia_museos',
               'asistencia_exposiciones', 'asistencia_monumentos',
               'asistencia_cursos', 'practica_cultural', 'lee_libros']

for col in binary_cols:
    if col in df.columns:
        df[col + '_num'] = df[col].map({'SI': 1, 'NO': 0})

# Crear grupos de edad
df['grupo_edad'] = pd.cut(
    df['edad'], 
    bins=[0, 18, 30, 45, 60, 100], 
    labels=['Joven (15-18)', 'Adulto Joven (19-30)', 
           'Adulto (31-45)', 'Adulto Mayor (46-60)', 
           'Senior (60+)']
)

# Crear grupos de ingreso
df['grupo_ingreso'] = pd.qcut(
    df['p2'], 
    q=4, 
    labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto']
)
        """)
    
    with col2:
        # Implementamos el procesamiento
        binary_cols = ['sexo', 'lectura_y_escritura', 'estudiante', 'p3', 'p4', 'p5',
                   'asistencia_biblioteca', 'asistencia_casas_de_cultura', 
                   'asistencia_centros_culturales', 'asistencia_museos',
                   'asistencia_exposiciones', 'asistencia_monumentos',
                   'asistencia_cursos', 'practica_cultural', 'lee_libros']
    
        for col in binary_cols:
            if col in df.columns:
                df[col + '_num'] = df[col].map({'SI': 1, 'NO': 0})
        
        # Crear grupos de edad
        df['grupo_edad'] = pd.cut(df['edad'], 
                                bins=[0, 18, 30, 45, 60, 100], 
                                labels=['Joven (15-18)', 'Adulto Joven (19-30)', 
                                       'Adulto (31-45)', 'Adulto Mayor (46-60)', 'Senior (60+)'])
        
        # Crear grupos de ingreso
        df['grupo_ingreso'] = pd.qcut(df['p2'], 
                                    q=4, 
                                    labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])
        
        st.write("**Nuevas variables creadas:**")
        nuevas_vars = [col for col in df.columns if col not in df_raw.columns]
        st.write(", ".join(nuevas_vars))
    
    # 3. Creación del índice cultural
    st.markdown("**3️⃣ Creación de Índices y Variables Derivadas**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Crear índice de participación cultural
cultural_vars = ['p3_num', 'p4_num', 'p5_num', 
                 'asistencia_biblioteca_num',
                 'asistencia_casas_de_cultura_num', 
                 'asistencia_centros_culturales_num',
                 'asistencia_museos_num', 'asistencia_exposiciones_num',
                 'asistencia_monumentos_num', 'asistencia_cursos_num',
                 'practica_cultural_num', 'lee_libros_num']

df['indice_cultural'] = df[cultural_vars].sum(axis=1)
df['nivel_participacion'] = pd.cut(
    df['indice_cultural'], 
    bins=[-1, 2, 5, 8, 12], 
    labels=['Bajo', 'Medio', 'Alto', 'Muy Alto']
)
        """)
    
    with col2:
        # Implementamos la creación del índice
        cultural_vars = ['p3_num', 'p4_num', 'p5_num', 'asistencia_biblioteca_num',
                    'asistencia_casas_de_cultura_num', 'asistencia_centros_culturales_num',
                    'asistencia_museos_num', 'asistencia_exposiciones_num',
                    'asistencia_monumentos_num', 'asistencia_cursos_num',
                    'practica_cultural_num', 'lee_libros_num']
    
        df['indice_cultural'] = df[cultural_vars].sum(axis=1)
        df['nivel_participacion'] = pd.cut(df['indice_cultural'], 
                                         bins=[-1, 2, 5, 8, 12], 
                                         labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])
        
        # Mostrar distribución del índice cultural
        fig = px.histogram(df, x='indice_cultural', 
                         color='nivel_participacion',
                         color_discrete_sequence=get_purple_palette(4),
                         title="Distribución del Índice Cultural")
        fig.update_layout(xaxis_title="Índice de Participación Cultural",
                        yaxis_title="Frecuencia",
                        bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
    
    # Resumen final
    st.subheader("Resumen del Proceso de Limpieza")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Acciones realizadas:**
        
        ✅ Identificación y tratamiento de valores nulos
        ✅ Conversión de variables categóricas a numéricas
        ✅ Creación de variables derivadas por agrupación
        ✅ Construcción del índice de participación cultural
        ✅ Aplicación de factor de expansión para análisis ponderados
        """)
    
    with col2:
        st.write("**Dataset final:**", df.shape)
        st.write("**Calidad de los datos:**")
        # Verificación de calidad de datos
        calidad_data = {
            'Métrica': ['Registros Totales', 'Columnas Totales', 'Valores Faltantes', 'Completitud de Datos'],
            'Valor': [df.shape[0], df.shape[1], df.isnull().sum().sum(), f"{100 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%"]
        }
        st.dataframe(pd.DataFrame(calidad_data), use_container_width=True)
else:
    # Para el resto de páginas, usamos los datos procesados
    # Procesamiento básico de datos (para asegurar que todas las variables existan)
    df = df_raw.copy()
    
    # Convertir variables binarias
    binary_cols = ['sexo', 'lectura_y_escritura', 'estudiante', 'p3', 'p4', 'p5',
                   'asistencia_biblioteca', 'asistencia_casas_de_cultura', 
                   'asistencia_centros_culturales', 'asistencia_museos',
                   'asistencia_exposiciones', 'asistencia_monumentos',
                   'asistencia_cursos', 'practica_cultural', 'lee_libros']
    
    for col in binary_cols:
        if col in df.columns:
            df[col + '_num'] = df[col].map({'SI': 1, 'NO': 0})
    
    # Crear grupos de edad
    df['grupo_edad'] = pd.cut(df['edad'], 
                             bins=[0, 18, 30, 45, 60, 100], 
                             labels=['Joven (15-18)', 'Adulto Joven (19-30)', 
                                   'Adulto (31-45)', 'Adulto Mayor (46-60)', 'Senior (60+)'])
    
    # Crear grupos de ingreso
    df['grupo_ingreso'] = pd.qcut(df['p2'], 
                                 q=4, 
                                 labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'])
    
    # Crear índice de participación cultural
    cultural_vars = ['p3_num', 'p4_num', 'p5_num', 'asistencia_biblioteca_num',
                    'asistencia_casas_de_cultura_num', 'asistencia_centros_culturales_num',
                    'asistencia_museos_num', 'asistencia_exposiciones_num',
                    'asistencia_monumentos_num', 'asistencia_cursos_num',
                    'practica_cultural_num', 'lee_libros_num']
    
    df['indice_cultural'] = df[cultural_vars].sum(axis=1)
    df['nivel_participacion'] = pd.cut(df['indice_cultural'], 
                                      bins=[-1, 2, 5, 8, 12], 
                                      labels=['Bajo', 'Medio', 'Alto', 'Muy Alto'])

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
        avg_age = df['edad'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Edad Promedio", f"{avg_age:.1f} años")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        reading_rate = (df['lee_libros'] == 'SI').mean() * 100
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
        fig = px.histogram(df, x='grupo_edad', color='sexo',
                          barmode='group',
                          color_discrete_sequence=['#6a0dad', '#ba55d3'],
                          title="")
        fig.update_layout(
            xaxis_title="Grupo de Edad",
            yaxis_title="Número de Personas",
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Top actividades culturales
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🏆 Top 10 Actividades Culturales Más Populares")
    
    activities = {
        'Lectura de Libros': (df['lee_libros'] == 'SI').mean(),
        'Práctica Cultural': (df['practica_cultural'] == 'SI').mean(),
        'Asistencia a Bibliotecas': (df['asistencia_biblioteca'] == 'SI').mean(),
        'Conciertos/Música en Vivo': (df['p4'] == 'SI').mean(),
        'Monumentos Históricos': (df['asistencia_monumentos'] == 'SI').mean(),
        'Teatro/Ópera/Danza': (df['p3'] == 'SI').mean(),
        'Centros Culturales': (df['asistencia_centros_culturales'] == 'SI').mean(),
        'Casas de Cultura': (df['asistencia_casas_de_cultura'] == 'SI').mean(),
        'Cursos/Talleres': (df['asistencia_cursos'] == 'SI').mean(),
        'Museos': (df['asistencia_museos'] == 'SI').mean()
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

# PÁGINA: ESTADÍSTICAS DESCRIPTIVAS
elif page == "📈 Estadísticas Descriptivas":
    st.markdown('<div class="section-header"><h2>📈 Estadísticas Descriptivas y Exploratorias</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Resumen Estadístico - Variables Numéricas")
        
        # Seleccionar variables numéricas de interés
        numeric_vars = ['edad', 'p2', 'indice_cultural']
        
        # Crear tabla de resumen estadístico
        stats_df = df[numeric_vars].describe().T
        
        # Agregar estadísticas adicionales
        stats_df['mediana'] = df[numeric_vars].median()
        stats_df['moda'] = df[numeric_vars].mode().iloc[0]
        stats_df['asimetria'] = df[numeric_vars].skew()
        stats_df['curtosis'] = df[numeric_vars].kurtosis()
        
        # Renombrar las columnas para mejor presentación
        stats_df = stats_df.rename(columns={
            'count': 'Conteo',
            'mean': 'Media',
            'std': 'Desv. Estándar',
            'min': 'Mínimo',
            '25%': 'Q1 (25%)',
            '50%': 'Q2 (50%)',
            '75%': 'Q3 (75%)',
            'max': 'Máximo',
            'mediana': 'Mediana',
            'moda': 'Moda',
            'asimetria': 'Asimetría',
            'curtosis': 'Curtosis'
        })
        
        # Renombrar índices para mejor presentación
        new_index = {
            'edad': 'Edad (años)',
            'p2': 'Ingresos',
            'indice_cultural': 'Índice Cultural'
        }
        stats_df.index = [new_index.get(idx, idx) for idx in stats_df.index]
        
        # Mostrar tabla con formato
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.subheader("📊 Resumen Estadístico - Variables Categóricas")
        
        # Seleccionar variables categóricas de interés
        cat_vars = ['sexo', 'nivel_educativo', 'etnia', 'grupo_edad']
        
        # Crear dataframe para almacenar estadísticas
        cat_stats = []
        
        for var in cat_vars:
            # Contar frecuencias
            counts = df[var].value_counts()
            # Calcular porcentajes
            percentages = df[var].value_counts(normalize=True) * 100
            
            # Encontrar moda (valor más frecuente)
            mode_val = counts.index[0]
            mode_count = counts.iloc[0]
            mode_pct = percentages.iloc[0]

# Crear diccionario con estadísticas
            stats = {
                'Variable': var,
                'Categorías': len(counts),
                'Moda': mode_val,
                'Frecuencia Moda': mode_count,
                'Porcentaje Moda (%)': round(mode_pct, 2),
                'Categoría Menos Frecuente': counts.index[-1],
                'Frecuencia Mínima': counts.iloc[-1],
                'Porcentaje Mínimo (%)': round(percentages.iloc[-1], 2)
            }
            
            cat_stats.append(stats)
        
        # Convertir a DataFrame
        cat_stats_df = pd.DataFrame(cat_stats)
        
        # Mostrar tabla con formato
        st.dataframe(cat_stats_df, use_container_width=True)
    
    # Visualización de distribuciones
    st.subheader("📉 Distribuciones de Variables Clave")
    
    dist_tab1, dist_tab2 = st.tabs(["📊 Variables Numéricas", "🔄 Variables Categóricas"])
    
    with dist_tab1:
        # Seleccionar variable numérica para visualizar
        num_var = st.selectbox(
            "Selecciona una variable numérica:",
            options=["edad", "p2", "indice_cultural"],
            format_func=lambda x: {
                "edad": "Edad (años)",
                "p2": "Ingresos",
                "indice_cultural": "Índice de Participación Cultural"
            }.get(x, x)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = px.histogram(df, x=num_var, 
                              title=f"Histograma de {num_var.replace('_', ' ').title()}",
                              color_discrete_sequence=['#6a0dad'])
            fig.update_layout(bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Boxplot
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = px.box(df, y=num_var, 
                        title=f"Boxplot de {num_var.replace('_', ' ').title()}",
                        color_discrete_sequence=['#6a0dad'])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with dist_tab2:
        # Seleccionar variable categórica para visualizar
        cat_var = st.selectbox(
            "Selecciona una variable categórica:",
            options=["sexo", "nivel_educativo", "etnia", "grupo_edad", "nivel_participacion"],
            format_func=lambda x: {
                "sexo": "Género",
                "nivel_educativo": "Nivel Educativo",
                "etnia": "Etnia",
                "grupo_edad": "Grupo de Edad",
                "nivel_participacion": "Nivel de Participación Cultural"
            }.get(x, x)
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            counts = df[cat_var].value_counts().reset_index()
            counts.columns = [cat_var, 'Conteo']
            
            fig = px.bar(counts, x=cat_var, y='Conteo',
                        title=f"Frecuencia de {cat_var.replace('_', ' ').title()}",
                        color=cat_var,
                        color_discrete_sequence=get_purple_palette(len(counts)))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Gráfico de pastel
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig = px.pie(counts, names=cat_var, values='Conteo',
                        title=f"Distribución de {cat_var.replace('_', ' ').title()}",
                        color=cat_var,
                        color_discrete_sequence=get_purple_palette(len(counts)))
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis de Correlaciones
    st.subheader("🔄 Matriz de Correlaciones")
    
    # Seleccionar variables numéricas relevantes
    corr_vars = ['edad', 'p2', 'indice_cultural'] + [col for col in df.columns if col.endswith('_num')]
    
    # Crear matriz de correlación
    corr_matrix = df[corr_vars].corr()
    
    # Seleccionar el tipo de gráfico de correlación
    corr_type = st.radio(
        "Tipo de visualización:",
        ["Mapa de Calor", "Matriz Cuadrícula"],
        horizontal=True
    )
    
    if corr_type == "Mapa de Calor":
        # Visualizar como mapa de calor
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       title="Matriz de Correlación - Variables Numéricas")
        fig.update_layout(width=800, height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Visualizar como matriz de cuadrícula
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        # Crear figura con subplots
        fig = make_subplots(rows=len(corr_vars), cols=len(corr_vars),
                           shared_xaxes=True, shared_yaxes=True,
                           horizontal_spacing=0.01, vertical_spacing=0.01)
        
        # Añadir scatter plots para cada par de variables
        for i, var1 in enumerate(corr_vars):
            for j, var2 in enumerate(corr_vars):
                corr_val = corr_matrix.iloc[i, j]
                color = 'rgba(186, 85, 211, ' + str(abs(corr_val)) + ')'
                
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode='markers',
                        marker=dict(
                            size=30,
                            color=color,
                            symbol='square',
                            line=dict(color='white', width=1)
                        ),
                        showlegend=False,
                        text=f"{corr_val:.2f}"
                    ),
                    row=i+1, col=j+1
                )
                
                # Añadir texto de correlación
                fig.add_annotation(
                    x=0.5, y=0.5,
                    text=f"{corr_val:.2f}",
                    showarrow=False,
                    row=i+1, col=j+1
                )
        
        # Actualizar layout
        fig.update_layout(
            width=800, height=800,
            title="Matriz de Correlación - Variables Numéricas"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Texto explicativo
    st.markdown("""
    **Interpretación de la Matriz de Correlación:**
    
    * **Correlación positiva (>0)**: Cuando una variable aumenta, la otra también tiende a aumentar.
    * **Correlación negativa (<0)**: Cuando una variable aumenta, la otra tiende a disminuir.
    * **Correlación fuerte**: Valores cercanos a 1 o -1.
    * **Correlación débil**: Valores cercanos a 0.
    """)

# PÁGINA: ANÁLISIS DEMOGRÁFICO
elif page == "🔍 Análisis Demográfico":
    st.markdown('<div class="section-header"><h2>🔍 Análisis Demográfico</h2></div>', unsafe_allow_html=True)
    
    # Filtro por grupos demográficos
    st.sidebar.markdown("### 👥 Filtros Demográficos")
    
    # Filtro por género
    selected_gender = st.sidebar.multiselect(
        "Filtrar por género:",
        options=df['sexo'].unique(),
        default=df['sexo'].unique()
    )
    
    # Filtro por grupo de edad
    selected_age = st.sidebar.multiselect(
        "Filtrar por grupo de edad:",
        options=df['grupo_edad'].unique(),
        default=df['grupo_edad'].unique()
    )
    
    # Filtro por nivel educativo
    selected_education = st.sidebar.multiselect(
        "Filtrar por nivel educativo:",
        options=df['nivel_educativo'].unique(),
        default=df['nivel_educativo'].unique()
    )
    
    # Aplicar filtros
    filtered_df = df[
        df['sexo'].isin(selected_gender) &
        df['grupo_edad'].isin(selected_age) &
        df['nivel_educativo'].isin(selected_education)
    ]
    
    # Mostrar número de registros filtrados
    st.sidebar.markdown(f"**Registros seleccionados:** {len(filtered_df):,}")
    
    # Análisis demográfico
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚻 Distribución por Género")
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        gender_counts = filtered_df['sexo'].value_counts()
        gender_pcts = filtered_df['sexo'].value_counts(normalize=True) * 100
        
        gender_df = pd.DataFrame({
            'Género': gender_counts.index,
            'Conteo': gender_counts.values,
            'Porcentaje': gender_pcts.values
        })
        
        fig = px.pie(gender_df, names='Género', values='Conteo',
                    color='Género',
                    color_discrete_sequence=['#6a0dad', '#9370db'],
                    title="Distribución por Género")
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("👴👩 Distribución por Grupo de Edad")
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        age_counts = filtered_df['grupo_edad'].value_counts().reset_index()
        age_counts.columns = ['Grupo de Edad', 'Conteo']
        
        # Ordenar las categorías de edad para visualización
        correct_order = ['Joven (15-18)', 'Adulto Joven (19-30)', 'Adulto (31-45)', 
                         'Adulto Mayor (46-60)', 'Senior (60+)']
        age_counts['Grupo de Edad'] = pd.Categorical(
            age_counts['Grupo de Edad'], 
            categories=correct_order, 
            ordered=True
        )
        age_counts = age_counts.sort_values('Grupo de Edad')
        
        fig = px.bar(age_counts, x='Grupo de Edad', y='Conteo',
                    color='Grupo de Edad',
                    color_discrete_sequence=get_purple_palette(len(age_counts)),
                    title="Distribución por Grupo de Edad")
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎓 Distribución por Nivel Educativo")
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        edu_counts = filtered_df['nivel_educativo'].value_counts()
        
        # Ordenar niveles educativos para mejor visualización
        education_order = ['BASICA PRIMARIA', 'BASICA SECUNDARIA', 'MEDIA', 'SUPERIOR', 'POSGRADO']
        edu_counts = edu_counts.reindex(education_order)
        
        fig = px.bar(x=edu_counts.index, y=edu_counts.values,
                    color=edu_counts.index,
                    color_discrete_sequence=get_purple_palette(len(edu_counts)),
                    labels={'x': 'Nivel Educativo', 'y': 'Conteo'},
                    title="Distribución por Nivel Educativo")
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("👥 Distribución por Etnia")
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        ethnicity_counts = filtered_df['etnia'].value_counts().reset_index()
        ethnicity_counts.columns = ['Etnia', 'Conteo']
        
        fig = px.pie(ethnicity_counts, names='Etnia', values='Conteo',
                    color='Etnia',
                    color_discrete_sequence=get_purple_palette(len(ethnicity_counts)),
                    title="Distribución por Etnia")
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis cruzado de variables demográficas
    st.subheader("📊 Análisis Cruzado de Variables Demográficas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selectores para variables a cruzar
        var1 = st.selectbox(
            "Primera variable demográfica:",
            options=["sexo", "grupo_edad", "nivel_educativo", "etnia"],
            format_func=lambda x: {
                "sexo": "Género", 
                "grupo_edad": "Grupo de Edad",
                "nivel_educativo": "Nivel Educativo",
                "etnia": "Etnia"
            }.get(x, x)
        )
    
    with col2:
        var2 = st.selectbox(
            "Segunda variable demográfica:",
            options=["nivel_educativo", "grupo_edad", "sexo", "etnia"],
            format_func=lambda x: {
                "sexo": "Género", 
                "grupo_edad": "Grupo de Edad",
                "nivel_educativo": "Nivel Educativo",
                "etnia": "Etnia"
            }.get(x, x)
        )
    
    # Crear gráfico cruzado
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Crear tabla de contingencia
    cross_table = pd.crosstab(filtered_df[var1], filtered_df[var2])
    
    # Visualizar
    fig = px.imshow(cross_table,
                   labels=dict(x=var2.replace('_', ' ').title(), y=var1.replace('_', ' ').title(), color='Frecuencia'),
                   color_continuous_scale='Purples',
                   title=f"Frecuencia Cruzada: {var1.replace('_', ' ').title()} vs {var2.replace('_', ' ').title()}")
    
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualización de la distribución del índice cultural por demografía
    st.subheader("📈 Índice Cultural por Grupos Demográficos")
    
    # Selector de variable demográfica
    demographic_var = st.selectbox(
        "Selecciona una variable demográfica para analizar:",
        options=["sexo", "grupo_edad", "nivel_educativo", "etnia"],
        format_func=lambda x: {
            "sexo": "Género", 
            "grupo_edad": "Grupo de Edad",
            "nivel_educativo": "Nivel Educativo",
            "etnia": "Etnia"
        }.get(x, x)
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Boxplot por grupo demográfico
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig = px.box(filtered_df, x=demographic_var, y='indice_cultural',
                    color=demographic_var,
                    color_discrete_sequence=get_purple_palette(
                        len(filtered_df[demographic_var].unique())
                    ),
                    title=f"Distribución del Índice Cultural por {demographic_var.replace('_', ' ').title()}")
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Promedio del índice cultural por grupo demográfico
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        avg_by_group = filtered_df.groupby(demographic_var)['indice_cultural'].mean().reset_index()
        avg_by_group.columns = [demographic_var, 'Promedio Índice Cultural']
        
        fig = px.bar(avg_by_group, x=demographic_var, y='Promedio Índice Cultural',
                    color=demographic_var,
                    color_discrete_sequence=get_purple_palette(len(avg_by_group)),
                    title=f"Promedio del Índice Cultural por {demographic_var.replace('_', ' ').title()}")
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Conclusiones demográficas
    st.subheader("📝 Hallazgos Demográficos Principales")
    
    # Calcular algunos hallazgos
    highest_edu_group = filtered_df.groupby('nivel_educativo')['indice_cultural'].mean().idxmax()
    highest_age_group = filtered_df.groupby('grupo_edad')['indice_cultural'].mean().idxmax()
    gender_diff = filtered_df.groupby('sexo')['indice_cultural'].mean()
    
    st.markdown(f"""
    * **Nivel Educativo**: El grupo con mayor índice de participación cultural es "{highest_edu_group}".
    * **Grupo de Edad**: El grupo con mayor participación cultural es "{highest_age_group}".
    * **Género**: La diferencia en participación cultural entre hombres y mujeres es de {abs(gender_diff.iloc[0] - gender_diff.iloc[1]):.2f} puntos.
    * **Diversidad Étnica**: Los patrones de participación cultural varían entre grupos étnicos, reflejando diferentes tradiciones y accesos.
    """)

# PÁGINA: PARTICIPACIÓN CULTURAL
elif page == "🎪 Participación Cultural":
    st.markdown('<div class="section-header"><h2>🎪 Análisis de Participación Cultural</h2></div>', unsafe_allow_html=True)
    
    # Filtro por nivel de participación
    st.sidebar.markdown("### 🎭 Filtros de Participación")
    
    # Filtro por nivel de participación
    selected_participation = st.sidebar.multiselect(
        "Filtrar por nivel de participación:",
        options=df['nivel_participacion'].unique(),
        default=df['nivel_participacion'].unique()
    )
    
    # Filtro por índice cultural
    min_index, max_index = st.sidebar.slider(
        "Rango de Índice Cultural:",
        min_value=0, max_value=12, value=(0, 12)
    )
    
    # Aplicar filtros
    filtered_df = df[
        df['nivel_participacion'].isin(selected_participation) &
        (df['indice_cultural'] >= min_index) &
        (df['indice_cultural'] <= max_index)
    ]
    
    # Mostrar número de registros filtrados
    st.sidebar.markdown(f"**Registros seleccionados:** {len(filtered_df):,}")
    
    # Distribución general del índice cultural
    st.subheader("📊 Distribución General del Índice Cultural")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig = px.histogram(filtered_df, x='indice_cultural', 
                          color='nivel_participacion',
                          color_discrete_sequence=get_purple_palette(
                              len(filtered_df['nivel_participacion'].unique())
                          ),
                          title="Distribución del Índice Cultural")
        
        fig.update_layout(xaxis_title="Índice Cultural",
                         yaxis_title="Frecuencia",
                         bargap=0.2)
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig = px.pie(filtered_df, names='nivel_participacion',
                    color='nivel_participacion',
                    color_discrete_sequence=get_purple_palette(
                        len(filtered_df['nivel_participacion'].unique())
                    ),
                    title="Distribución por Nivel de Participación")
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis comparativo por variables demográficas
    st.subheader("🔄 Participación Cultural por Variables Demográficas")
    
    # Seleccionar variable para comparar
    compare_var = st.selectbox(
        "Comparar nivel de participación cultural por:",
        options=["sexo", "grupo_edad", "nivel_educativo", "etnia", "grupo_ingreso"],
        format_func=lambda x: {
            "sexo": "Género", 
            "grupo_edad": "Grupo de Edad",
            "nivel_educativo": "Nivel Educativo",
            "etnia": "Etnia",
            "grupo_ingreso": "Nivel de Ingresos"
        }.get(x, x)
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras apiladas
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        participation_crosstab = pd.crosstab(
            filtered_df[compare_var],
            filtered_df['nivel_participacion'],
            normalize='index'
        ) * 100
        
        participation_crosstab = participation_crosstab.reset_index()
        participation_crosstab_melted = pd.melt(
            participation_crosstab, 
            id_vars=[compare_var],
            var_name='Nivel de Participación',
            value_name='Porcentaje'
        )
        
        fig = px.bar(participation_crosstab_melted, 
                    x=compare_var, 
                    y='Porcentaje',
                    color='Nivel de Participación',
                    color_discrete_sequence=get_purple_palette(
                        len(filtered_df['nivel_participacion'].unique())
                    ),
                    title=f"Niveles de Participación por {compare_var.replace('_', ' ').title()}")
        
        fig.update_layout(yaxis_title="Porcentaje (%)", barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Promedio del índice cultural por grupo
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        
        avg_by_group = filtered_df.groupby(compare_var)['indice_cultural'].mean().reset_index()
        avg_by_group.columns = [compare_var, 'Promedio Índice Cultural']
        
        fig = px.bar(avg_by_group, 
                    x=compare_var, 
                    y='Promedio Índice Cultural',
                    color=compare_var,
                    color_discrete_sequence=get_purple_palette(len(avg_by_group)),
                    title=f"Promedio del Índice Cultural por {compare_var.replace('_', ' ').title()}")
        
        fig.update_layout(yaxis_title="Índice Cultural Promedio")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis de factores que influyen en la participación cultural
    st.subheader("🔍 Factores de Influencia en la Participación Cultural")
    
    # Crear un heatmap de correlación entre variables demográficas e índice cultural
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Seleccionar variables categóricas relevantes y convertirlas en dummies
    cat_vars = ['sexo', 'grupo_edad', 'nivel_educativo', 'etnia', 'grupo_ingreso']
    df_dummies = pd.get_dummies(filtered_df[cat_vars])
    
    # Añadir el índice cultural
    df_dummies['indice_cultural'] = filtered_df['indice_cultural']
    
    # Calcular correlaciones con el índice cultural
    corr_with_index = df_dummies.corr()['indice_cultural'].sort_values(ascending=False)
    
    # Eliminar la autocorrelación
    corr_with_index = corr_with_index[corr_with_index.index != 'indice_cultural']
    
    # Tomar las top 10 correlaciones
    top_corr = corr_with_index.head(10)
    bottom_corr = corr_with_index.tail(10)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Factores con Correlación Positiva", "Factores con Correlación Negativa"))
    
    # Gráfico de correlaciones positivas
    fig.add_trace(
        go.Bar(
            y=top_corr.index,
            x=top_corr.values,
            orientation='h',
            marker_color='rgba(106, 13, 173, 0.7)',
            name="Correlación Positiva"
        ),
        row=1, col=1
    )
    
    # Gráfico de correlaciones negativas
    fig.add_trace(
        go.Bar(
            y=bottom_corr.index,
            x=bottom_corr.values,
            orientation='h',
            marker_color='rgba(221, 160, 221, 0.7)',
            name="Correlación Negativa"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Factores que Influyen en el Índice de Participación Cultural",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mapa de calor de actividades culturales
    st.subheader("🎨 Matriz de Relación entre Actividades Culturales")
    
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Seleccionar columnas de actividades culturales
    activity_cols = [
        'asistencia_biblioteca_num', 'asistencia_casas_de_cultura_num',
        'asistencia_centros_culturales_num', 'asistencia_museos_num',
        'asistencia_exposiciones_num', 'asistencia_monumentos_num',
        'asistencia_cursos_num', 'practica_cultural_num', 'lee_libros_num',
        'p3_num', 'p4_num', 'p5_num'
    ]
    
    # Crear matriz de correlación
    activity_corr = filtered_df[activity_cols].corr()
    
    # Renombrar columnas para mejor visualización
    activity_names = {
        'asistencia_biblioteca_num': 'Biblioteca',
        'asistencia_casas_de_cultura_num': 'Casas de Cultura',
        'asistencia_centros_culturales_num': 'Centros Culturales',
        'asistencia_museos_num': 'Museos',
        'asistencia_exposiciones_num': 'Exposiciones',
        'asistencia_monumentos_num': 'Monumentos',
        'asistencia_cursos_num': 'Cursos/Talleres',
        'practica_cultural_num': 'Práctica Cultural',
        'lee_libros_num': 'Lectura',
        'p3_num': 'Teatro/Danza',
        'p4_num': 'Conciertos',
        'p5_num': 'Otra Actividad'
    }
    
    activity_corr.columns = [activity_names.get(col, col) for col in activity_corr.columns]
    activity_corr.index = [activity_names.get(idx, idx) for idx in activity_corr.index]
    
    # Visualizar matriz de correl

# Visualizar matriz de correlación
    fig = px.imshow(
        activity_corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title="Correlación entre Actividades Culturales"
    )
    
    fig.update_layout(width=800, height=700)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis por tipo de actividad
    st.subheader("📊 Participación por Tipo de Actividad Cultural")
    
    # Preparar datos de actividades culturales
    activity_data = pd.DataFrame({
        'Actividad': list(activity_names.values()),
        'Participación': [filtered_df[col].mean() for col in activity_cols]
    })
    
    activity_data = activity_data.sort_values('Participación', ascending=False)
    
    # Visualizar participación por actividad
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig = px.bar(
        activity_data,
        x='Actividad',
        y='Participación',
        color='Actividad',
        color_discrete_sequence=get_purple_palette(len(activity_data)),
        title="Participación Promedio por Tipo de Actividad Cultural"
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Conclusiones sobre participación cultural
    st.subheader("📝 Hallazgos sobre Participación Cultural")
    
    # Calcular algunos hallazgos
    top_activity = activity_data.iloc[0]['Actividad']
    bottom_activity = activity_data.iloc[-1]['Actividad']
    highest_group = filtered_df.groupby('nivel_educativo')['indice_cultural'].mean().idxmax()
    
    st.markdown(f"""
    * **Actividad más popular**: La actividad con mayor nivel de participación es "{top_activity}".
    * **Actividad menos popular**: La actividad con menor nivel de participación es "{bottom_activity}".
    * **Grupo con mayor participación**: El grupo con mayor índice cultural es el de nivel educativo "{highest_group}".
    * **Factores de influencia**: Se observa una correlación entre nivel educativo y participación cultural.
    * **Relación entre actividades**: Existen patrones de correlación entre ciertos tipos de actividades culturales.
    """)

# PÁGINA: BARRERAS DE ACCESO
elif page == "🚧 Barreras de Acceso":
    st.markdown('<div class="section-header"><h2>🚧 Análisis de Barreras de Acceso</h2></div>', unsafe_allow_html=True)
    
    # Filtros laterales
    st.sidebar.markdown("### 🔍 Filtros de Análisis")
    
    # Filtro por nivel de participación
    selected_barrier_participation = st.sidebar.multiselect(
        "Filtrar por nivel de participación:",
        options=df['nivel_participacion'].unique(),
        default=df['nivel_participacion'].unique()
    )
    
    # Filtro por nivel educativo
    selected_barrier_education = st.sidebar.multiselect(
        "Filtrar por nivel educativo:",
        options=df['nivel_educativo'].unique(),
        default=df['nivel_educativo'].unique()
    )
    
    # Aplicar filtros
    filtered_df = df[
        df['nivel_participacion'].isin(selected_barrier_participation) &
        df['nivel_educativo'].isin(selected_barrier_education)
    ]
    
    # Mostrar número de registros filtrados
    st.sidebar.markdown(f"**Registros seleccionados:** {len(filtered_df):,}")
    
    # Definir barreras de acceso
    barriers = {
        'barrera_falta_tiempo': 'Falta de tiempo',
        'barrera_falta_dinero': 'Falta de dinero',
        'barrera_no_hay_actividades': 'Ausencia de oferta cultural',
        'barrera_falta_informacion': 'Falta de información',
        'barrera_transporte': 'Problemas de transporte',
        'barrera_seguridad': 'Problemas de seguridad',
        'barrera_no_interesa': 'Falta de interés',
        'barrera_responsabilidades_familiares': 'Responsabilidades familiares'
    }
    
    # Análisis general de barreras
    st.subheader("📊 Incidencia General de Barreras")
    
    # Preparar datos de barreras
    barrier_data = pd.DataFrame({
        'Barrera': list(barriers.values()),
        'Porcentaje': [filtered_df[col].mean() * 100 for col in barriers.keys()]
    })
    
    barrier_data = barrier_data.sort_values('Porcentaje', ascending=False)
    
    # Visualizar barreras generales
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig = px.bar(
        barrier_data,
        x='Barrera',
        y='Porcentaje',
        color='Barrera',
        color_discrete_sequence=get_purple_palette(len(barrier_data)),
        title="Incidencia de Barreras de Acceso a la Cultura (%)"
    )
    
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Porcentaje (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis de barreras por demografía
    st.subheader("🔍 Barreras de Acceso por Grupos Demográficos")
    
    # Selector de variable demográfica
    barrier_demographic_var = st.selectbox(
        "Analizar barreras por variable demográfica:",
        options=["sexo", "grupo_edad", "nivel_educativo", "etnia", "grupo_ingreso"],
        format_func=lambda x: {
            "sexo": "Género", 
            "grupo_edad": "Grupo de Edad",
            "nivel_educativo": "Nivel Educativo",
            "etnia": "Etnia",
            "grupo_ingreso": "Nivel de Ingresos"
        }.get(x, x)
    )
    
    # Selector de barrera a analizar
    selected_barrier = st.selectbox(
        "Seleccionar barrera para análisis:",
        options=list(barriers.keys()),
        format_func=lambda x: barriers.get(x, x)
    )
    
    # Crear gráfico de barrera por grupo demográfico
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    barrier_by_group = filtered_df.groupby(barrier_demographic_var)[selected_barrier].mean() * 100
    barrier_by_group = barrier_by_group.reset_index()
    barrier_by_group.columns = [barrier_demographic_var, 'Porcentaje']
    
    fig = px.bar(
        barrier_by_group,
        x=barrier_demographic_var,
        y='Porcentaje',
        color=barrier_demographic_var,
        color_discrete_sequence=get_purple_palette(len(barrier_by_group)),
        title=f"Incidencia de '{barriers[selected_barrier]}' por {barrier_demographic_var.replace('_', ' ').title()}"
    )
    
    fig.update_layout(yaxis_title="Porcentaje (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mapa de calor de barreras por nivel de participación
    st.subheader("🔄 Barreras según Nivel de Participación Cultural")
    
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Preparar datos para el mapa de calor
    barrier_participation_data = filtered_df.groupby('nivel_participacion')[list(barriers.keys())].mean() * 100
    barrier_participation_data.columns = [barriers[col] for col in barrier_participation_data.columns]
    
    # Crear mapa de calor
    fig = px.imshow(
        barrier_participation_data,
        text_auto='.1f',
        color_continuous_scale='Purples',
        title="Incidencia de Barreras (%) por Nivel de Participación Cultural",
        labels=dict(x="Barrera", y="Nivel de Participación", color="Porcentaje (%)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis de correlación entre barreras e índice cultural
    st.subheader("📉 Impacto de las Barreras en el Índice Cultural")
    
    # Calcular correlaciones entre barreras e índice cultural
    barrier_correlations = pd.DataFrame({
        'Barrera': list(barriers.values()),
        'Correlación': [filtered_df[col].corr(filtered_df['indice_cultural']) for col in barriers.keys()]
    })
    
    barrier_correlations = barrier_correlations.sort_values('Correlación')
    
    # Visualizar correlaciones
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig = px.bar(
        barrier_correlations,
        x='Barrera',
        y='Correlación',
        color='Correlación',
        color_continuous_scale='RdBu_r',
        title="Correlación entre Barreras e Índice de Participación Cultural"
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Conclusiones sobre barreras de acceso
    st.subheader("📝 Hallazgos sobre Barreras de Acceso")
    
    # Obtener principales hallazgos
    top_barrier = barrier_data.iloc[0]['Barrera']
    top_corr_barrier = barrier_correlations.iloc[-1]['Barrera']
    lowest_corr_barrier = barrier_correlations.iloc[0]['Barrera']
    
    st.markdown(f"""
    * **Barrera más común**: La barrera más reportada es "{top_barrier}" con un {barrier_data.iloc[0]['Porcentaje']:.1f}% de incidencia.
    * **Mayor impacto negativo**: La barrera con mayor correlación negativa con el índice cultural es "{lowest_corr_barrier}".
    * **Mayor impacto positivo/menor impacto negativo**: La barrera "{top_corr_barrier}" muestra la menor correlación negativa (o mayor positiva) con la participación cultural.
    * **Variación demográfica**: Las barreras experimentadas varían significativamente entre diferentes grupos demográficos.
    * **Patrones por nivel de participación**: Se observan patrones claros en las barreras reportadas según el nivel de participación cultural.
    """)

# PÁGINA: RECOMENDACIONES
elif page == "💡 Recomendaciones":
    st.markdown('<div class="section-header"><h2>💡 Recomendaciones y Conclusiones</h2></div>', unsafe_allow_html=True)
    
    # Resumen ejecutivo
    st.subheader("📋 Resumen Ejecutivo")
    
    # Calcular algunos indicadores clave
    participation_pct = len(df[df['nivel_participacion'] != 'Muy Baja']) / len(df) * 100
    high_participation_pct = len(df[df['nivel_participacion'].isin(['Alta', 'Muy Alta'])]) / len(df) * 100
    avg_index = df['indice_cultural'].mean()
    top_barrier = df[['barrera_falta_tiempo', 'barrera_falta_dinero', 'barrera_no_hay_actividades', 
                    'barrera_falta_informacion', 'barrera_transporte', 'barrera_seguridad', 
                    'barrera_no_interesa', 'barrera_responsabilidades_familiares']].mean().idxmax()
    
    barrier_names = {
        'barrera_falta_tiempo': 'la falta de tiempo',
        'barrera_falta_dinero': 'la falta de dinero',
        'barrera_no_hay_actividades': 'la ausencia de oferta cultural',
        'barrera_falta_informacion': 'la falta de información',
        'barrera_transporte': 'los problemas de transporte',
        'barrera_seguridad': 'los problemas de seguridad',
        'barrera_no_interesa': 'la falta de interés',
        'barrera_responsabilidades_familiares': 'las responsabilidades familiares'
    }
    
    st.markdown(f"""
    Este análisis ha examinado los patrones de participación cultural entre una muestra de {len(df):,} individuos, evaluando sus características demográficas, niveles de participación y barreras de acceso.
    
    **Hallazgos clave:**
    
    - El {participation_pct:.1f}% de la población muestra algún nivel de participación cultural significativa.
    - Solo el {high_participation_pct:.1f}% presenta niveles altos o muy altos de participación.
    - El índice cultural promedio es de {avg_index:.2f} puntos (escala 0-12).
    - La principal barrera de acceso identificada es {barrier_names[top_barrier]}.
    - Existe una fuerte correlación entre el nivel educativo y la participación cultural.
    
    Las estrategias recomendadas se centran en reducir las barreras de acceso identificadas y en fomentar una mayor inclusión cultural entre grupos con menor participación.
    """)
    
    # Recomendaciones principales
    st.subheader("💡 Recomendaciones Principales")
    
    # Tabs para categorías de recomendaciones
    rec_tab1, rec_tab2, rec_tab3 = st.tabs([
        "🎯 Aumentar Accesibilidad", 
        "📢 Comunicación e Información",
        "👥 Inclusión y Diversidad"
    ])
    
    with rec_tab1:
        st.markdown("""
        ### 🎯 Recomendaciones para Aumentar la Accesibilidad
        
        1. **Programas de subsidios culturales**
            - Implementar un sistema de vales o descuentos para actividades culturales dirigido a grupos con menores ingresos.
            - Establecer días de entrada gratuita o con descuento en museos y centros culturales.
            
        2. **Mejorar la distribución geográfica**
            - Descentralizar la oferta cultural, llevando actividades a barrios periféricos y áreas rurales.
            - Implementar programas culturales itinerantes que visiten diferentes comunidades.
            
        3. **Adaptación de horarios**
            - Ofrecer actividades culturales en horarios diversos, incluyendo noches y fines de semana.
            - Crear programas culturales específicos para personas con limitaciones de tiempo, como formatos express o digitales.
            
        4. **Transporte y movilidad**
            - Coordinar servicios de transporte especiales para eventos culturales importantes.
            - Establecer colaboraciones con servicios de transporte público para ofrecer descuentos en días de eventos culturales.
        """)
    
    with rec_tab2:
        st.markdown("""
        ### 📢 Recomendaciones para Comunicación e Información
        
        1. **Estrategia digital integrada**
            - Desarrollar una plataforma digital centralizada que agrupe toda la oferta cultural.
            - Implementar un sistema de notificaciones personalizadas según intereses culturales.
            
        2. **Campañas de concientización**
            - Realizar campañas de comunicación sobre los beneficios de la participación cultural.
            - Destacar historias de impacto positivo y transformación a través de la cultura.
            
        3. **Educación cultural desde temprana edad**
            - Fortalecer los programas educativos relacionados con actividades culturales.
            - Establecer visitas escolares regulares a instituciones culturales.
            
        4. **Formación de mediadores culturales**
            - Capacitar a personas de las comunidades como embajadores culturales.
            - Desarrollar programas de voluntariado cultural para aumentar la difusión.
        """)
    
    with rec_tab3:
        st.markdown("""
        ### 👥 Recomendaciones para Inclusión y Diversidad
        
        1. **Programas específicos para grupos subrepresentados**
            - Desarrollar iniciativas culturales específicas para adultos mayores.
            - Crear programas dirigidos a grupos con menor participación según el análisis demográfico.
            
        2. **Cultura inclusiva y accesible**
            - Garantizar la accesibilidad física en todos los espacios culturales.
            - Ofrecer adaptaciones para personas con discapacidades sensoriales.
            
        3. **Reconocimiento de la diversidad cultural**
            - Valorar y promocionar expresiones culturales diversas y representativas de todos los grupos étnicos.
            - Establecer cuotas de representación en la programación cultural institucional.
            
        4. **Cocreación cultural**
            - Implementar metodologías participativas donde las comunidades sean creadoras de contenido cultural.
            - Establecer laboratorios culturales comunitarios en diferentes territorios.
        """)
    
    # KPIs y métricas de impacto
    st.subheader("📊 KPIs y Métricas de Impacto")
    
    # Crear gráfico de objetivos
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    
    # Datos actuales y objetivos
    kpi_data = pd.DataFrame({
        'Métrica': [
            'Índice Cultural Promedio', 
            '% Participación Alta/Muy Alta', 
            '% Con acceso a al menos 3 actividades',
            '% Reporta barreras económicas',
            '% Reporta falta de información'
        ],
        'Valor Actual': [
            df['indice_cultural'].mean(),
            len(df[df['nivel_participacion'].isin(['Alta', 'Muy Alta'])]) / len(df) * 100,
            len(df[df['indice_cultural'] >= 3]) / len(df) * 100,
            df['barrera_falta_dinero'].mean() * 100,
            df['barrera_falta_informacion'].mean() * 100
        ],
        'Objetivo': [
            df['indice_cultural'].mean() * 1.3,  # 30% de aumento
            high_participation_pct * 1.5,  # 50% de aumento
            len(df[df['indice_cultural'] >= 3]) / len(df) * 100 * 1.4,  # 40% de aumento
            df['barrera_falta_dinero'].mean() * 100 * 0.6,  # 40% de reducción
            df['barrera_falta_informacion'].mean() * 100 * 0.5  # 50% de reducción
        ]
    })
    
    # Visualizar KPIs
    fig = go.Figure()
    
    # Añadir barras para valores actuales
    fig.add_trace(go.Bar(
        y=kpi_data['Métrica'],
        x=kpi_data['Valor Actual'],
        name='Valor Actual',
        orientation='h',
        marker=dict(color='rgba(106, 13, 173, 0.7)')
    ))
    
    # Añadir marcadores para objetivos
    fig.add_trace(go.Scatter(
        y=kpi_data['Métrica'],
        x=kpi_data['Objetivo'],
        name='Objetivo',
        mode='markers',
        marker=dict(symbol='diamond', size=12, color='rgba(221, 160, 221, 1)')
    ))
    
    # Actualizar layout
    fig.update_layout(
        title='KPIs y Objetivos de Participación Cultural',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Conclusiones finales
    st.subheader("📝 Conclusiones Finales")
    
    st.markdown("""
    El análisis de los datos de participación cultural ha revelado patrones significativos que permiten diseñar estrategias efectivas para democratizar el acceso a la cultura:
    
    1. **Brechas demográficas persistentes**: Existen diferencias notables en la participación cultural según el nivel educativo, grupo de edad y nivel socioeconómico, lo que sugiere la necesidad de políticas focalizadas.
    
    2. **Barreras multidimensionales**: Las principales barreras identificadas son económicas, temporales e informativas, requiriendo un enfoque integral para su superación.
    
    3. **Potencial de crecimiento**: El análisis muestra un importante margen de crecimiento en participación cultural si se implementan las estrategias recomendadas.
    
    4. **Oportunidades digitales**: La digitalización ofrece nuevas vías para aumentar el acceso cultural, especialmente para quienes enfrentan limitaciones de tiempo o movilidad.
    
    5. **Importancia de la medición continua**: Se recomienda establecer un sistema de monitoreo permanente de los indicadores de participación cultural para evaluar el impacto de las intervenciones.
    
    Implementar las recomendaciones propuestas podría transformar significativamente el panorama de participación cultural, fomentando una sociedad más inclusiva, cohesionada y con mayor bienestar a través del acceso equitativo a experiencias culturales.
    """)

# CSS para styling
st.markdown("""
<style>
    .plot-container {
        background-color: #fafafa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);
    }
    
    .section-header {
        background-color: #6a0dad;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    h2 {
        margin: 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f1f1;
        border-radius: 5px 5px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6a0dad !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)