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
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
# SOLUCIÓN DE EMERGENCIA - ELIMINA ESTO LUEGO DE FUNCIONE
import subprocess
import sys

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA Y ESTILOS CSS
# =============================================================================

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
    .info-box {
        background-color: #f8f4ff;
        border-left: 4px solid #6a0dad;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff8dc;
        border-left: 4px solid #ffa500;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #f0fff0;
        border-left: 4px solid #32cd32;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_purple_palette(n_colors):
    """
    Genera una paleta de colores morados para visualizaciones.
    
    Parámetros:
    -----------
    n_colors : int
        Número de colores necesarios para la paleta
        
    Retorna:
    --------
    list: Lista de códigos de colores hexadecimales en tonos morados
    """
    base_colors = ['#6a0dad', '#8a2be2', '#9370db', '#9932cc', '#ba55d3', 
                  '#da70d6', '#dda0dd', '#e6e6fa', '#f8f4ff']
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    else:
        # Si necesitamos más colores, usar interpolación de Plotly
        return px.colors.sample_colorscale('Purples', n_colors)

def limpiar_e_imputar_datos(df, umbral_na=0.30, metodo_numerico='knn', k_neighbors=5):
    """
    Limpia un DataFrame eliminando variables con exceso de valores faltantes
    e imputa los valores restantes usando diferentes estrategias.

    EXPLICACIÓN DEL PROCESO:
    1. Identifica columnas con más del umbral especificado de valores NA
    2. Elimina estas columnas porque son poco útiles para análisis
    3. Imputa valores categóricos faltantes con "NO INFORMACION"
    4. Imputa valores numéricos usando el método seleccionado

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame original a limpiar
    umbral_na : float, default=0.30
        Proporción máxima de NAs permitida (0.30 = 30%)
        Columnas con más NAs se eliminan
    metodo_numerico : str, default='knn'
        Método para imputar variables numéricas:
        - 'knn': K-Nearest Neighbors (más sofisticado)
        - 'mediana': Usar la mediana de cada columna
        - 'interpolacion': Interpolación lineal
        - 'forward_fill': Propagar último valor válido
    k_neighbors : int, default=5
        Número de vecinos para KNN (solo aplica si metodo_numerico='knn')

    Retorna:
    --------
    tuple: (DataFrame limpio, diccionario con información del proceso)
    """
    
    # Crear copia para no modificar el original
    df_limpio = df.copy()
    
    # Diccionario para registrar todo el proceso de limpieza
    info_limpieza = {
        'forma_original': df.shape,
        'columnas_eliminadas': [],
        'columnas_imputadas_categoricas': [],
        'columnas_imputadas_numericas': [],
        'metodo_numerico_usado': metodo_numerico,
        'observaciones': []
    }

    # PASO 1: ANÁLISIS DE VALORES FALTANTES POR COLUMNA
    na_por_columna = df_limpio.isnull().sum()
    porcentaje_na = (na_por_columna / len(df_limpio)) * 100
    
    # PASO 2: IDENTIFICAR Y ELIMINAR COLUMNAS CON EXCESO DE NAs
    columnas_a_eliminar = porcentaje_na[porcentaje_na > (umbral_na * 100)].index.tolist()
    
    if columnas_a_eliminar:
        df_limpio = df_limpio.drop(columns=columnas_a_eliminar)
        info_limpieza['columnas_eliminadas'] = columnas_a_eliminar
        info_limpieza['observaciones'].append(f"Eliminadas {len(columnas_a_eliminar)} columnas con >{umbral_na*100}% de NAs")

    # PASO 3: SEPARAR VARIABLES POR TIPO DE DATO
    columnas_categoricas = df_limpio.select_dtypes(include=['object', 'category']).columns.tolist()
    columnas_numericas = df_limpio.select_dtypes(include=[np.number]).columns.tolist()
    
    info_limpieza['observaciones'].append(f"Identificadas {len(columnas_categoricas)} variables categóricas y {len(columnas_numericas)} numéricas")

    # PASO 4: IMPUTACIÓN DE VARIABLES CATEGÓRICAS
    # Estrategia: Reemplazar NAs con "NO INFORMACION" para mantener la información
    # de que el dato no estaba disponible
    for col in columnas_categoricas:
        if df_limpio[col].isnull().sum() > 0:
            # Manejar columnas categóricas de pandas (tienen categorías predefinidas)
            if df_limpio[col].dtype.name == 'category':
                # Agregar la nueva categoría si no existe
                if "NO INFORMACION" not in df_limpio[col].cat.categories:
                    df_limpio[col] = df_limpio[col].cat.add_categories(["NO INFORMACION"])
                df_limpio[col] = df_limpio[col].fillna("NO INFORMACION")
            else:
                # Para columnas de texto normales
                df_limpio[col] = df_limpio[col].fillna("NO INFORMACION")
            
            info_limpieza['columnas_imputadas_categoricas'].append(col)

    # PASO 5: IMPUTACIÓN DE VARIABLES NUMÉRICAS
    columnas_numericas_con_na = [col for col in columnas_numericas 
                                if df_limpio[col].isnull().sum() > 0]

    if columnas_numericas_con_na:
        if metodo_numerico == 'knn':
            # KNN requiere al menos 2 columnas numéricas para funcionar correctamente
            if len(columnas_numericas) >= 2:
                try:
                    # KNN Imputer usa los k vecinos más cercanos para estimar valores faltantes
                    imputer = KNNImputer(n_neighbors=min(k_neighbors, len(df_limpio)-1))
                    df_limpio[columnas_numericas] = imputer.fit_transform(df_limpio[columnas_numericas])
                    info_limpieza['observaciones'].append(f"KNN aplicado exitosamente con {k_neighbors} vecinos")
                except Exception as e:
                    # Si KNN falla, usar mediana como respaldo
                    info_limpieza['observaciones'].append(f"KNN falló: {str(e)}. Usando mediana como respaldo")
                    for col in columnas_numericas_con_na:
                        mediana = df_limpio[col].median()
                        df_limpio[col] = df_limpio[col].fillna(mediana)
            else:
                # Con una sola columna numérica, KNN no es útil
                info_limpieza['observaciones'].append("Solo 1 columna numérica disponible. Usando mediana")
                for col in columnas_numericas_con_na:
                    mediana = df_limpio[col].median()
                    df_limpio[col] = df_limpio[col].fillna(mediana)
                    
        elif metodo_numerico == 'mediana':
            # Mediana es robusta a valores atípicos
            for col in columnas_numericas_con_na:
                mediana = df_limpio[col].median()
                df_limpio[col] = df_limpio[col].fillna(mediana)
                
        elif metodo_numerico == 'interpolacion':
            # Interpolación lineal - útil para series temporales
            for col in columnas_numericas_con_na:
                df_limpio[col] = df_limpio[col].interpolate(method='linear')
                # Rellenar valores al inicio/final que no se pueden interpolar
                df_limpio[col] = df_limpio[col].fillna(method='ffill').fillna(method='bfill')
                
        elif metodo_numerico == 'forward_fill':
            # Propagar último valor válido hacia adelante
            for col in columnas_numericas_con_na:
                df_limpio[col] = df_limpio[col].fillna(method='ffill').fillna(method='bfill')
        
        info_limpieza['columnas_imputadas_numericas'] = columnas_numericas_con_na

    info_limpieza['forma_final'] = df_limpio.shape
    
    return df_limpio, info_limpieza

def mostrar_resultados_limpieza(info_limpieza):
    """
    Muestra los resultados del proceso de limpieza en la interfaz de Streamlit.
    Presenta información clara sobre qué se hizo durante la limpieza.
    """
    st.markdown("### 📊 Resultados de la Limpieza de Datos")

    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📋 Filas Originales", f"{info_limpieza['forma_original'][0]:,}")
        st.metric("📋 Filas Finales", f"{info_limpieza['forma_final'][0]:,}")
        diferencia_filas = info_limpieza['forma_final'][0] - info_limpieza['forma_original'][0]
        if diferencia_filas != 0:
            st.metric("📋 Diferencia", f"{diferencia_filas:+,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Columnas Originales", f"{info_limpieza['forma_original'][1]:,}")
        st.metric("📊 Columnas Finales", f"{info_limpieza['forma_final'][1]:,}")
        diferencia_cols = info_limpieza['forma_final'][1] - info_limpieza['forma_original'][1]
        if diferencia_cols != 0:
            st.metric("📊 Diferencia", f"{diferencia_cols:+,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔧 Método Numérico", info_limpieza['metodo_numerico_usado'])
        st.metric("🗑️ Columnas Eliminadas", len(info_limpieza['columnas_eliminadas']))
        total_imputadas = len(info_limpieza['columnas_imputadas_categoricas']) + len(info_limpieza['columnas_imputadas_numericas'])
        st.metric("🔧 Columnas Imputadas", total_imputadas)
        st.markdown('</div>', unsafe_allow_html=True)

    # Detalles del proceso
    if info_limpieza['columnas_eliminadas']:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**🗑️ Columnas Eliminadas (exceso de valores faltantes):**")
        for i, col in enumerate(info_limpieza['columnas_eliminadas'], 1):
            st.write(f"{i}. {col}")
        st.markdown('</div>', unsafe_allow_html=True)

    if info_limpieza['columnas_imputadas_categoricas']:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**🔤 Columnas Categóricas Imputadas (reemplazadas con 'NO INFORMACION'):**")
        for i, col in enumerate(info_limpieza['columnas_imputadas_categoricas'], 1):
            st.write(f"{i}. {col}")
        st.markdown('</div>', unsafe_allow_html=True)

    if info_limpieza['columnas_imputadas_numericas']:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"**🔢 Columnas Numéricas Imputadas (método: {info_limpieza['metodo_numerico_usado']}):**")
        for i, col in enumerate(info_limpieza['columnas_imputadas_numericas'], 1):
            st.write(f"{i}. {col}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Observaciones del proceso
    if info_limpieza['observaciones']:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("**📝 Observaciones del Proceso:**")
        for i, obs in enumerate(info_limpieza['observaciones'], 1):
            st.write(f"{i}. {obs}")
        st.markdown('</div>', unsafe_allow_html=True)

def analizar_duplicados_detallado(df):
    """
    Realiza un análisis detallado de duplicados en el DataFrame.
    
    EXPLICACIÓN DE TIPOS DE DUPLICADOS:
    1. Duplicados completos: Filas idénticas en todas las columnas
    2. Duplicados parciales: Registros con mismos valores en variables clave
       (edad, sexo, educación, etc.) pero diferentes en otras variables
    
    ¿POR QUÉ ES IMPORTANTE?
    - Duplicados completos pueden indicar errores de captura de datos
    - Duplicados parciales pueden ser legítimos (ej: personas con mismas características)
    - En encuestas, algunos duplicados parciales son esperables
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame a analizar
        
    Retorna:
    --------
    dict: Diccionario con información detallada sobre duplicados
    """
    
    info_duplicados = {
        'duplicados_completos': 0,
        'duplicados_parciales': 0,
        'variables_clave_encontradas': [],
        'recomendaciones': []
    }
    
    # ANÁLISIS 1: DUPLICADOS COMPLETOS
    # Son filas exactamente iguales en todas las columnas
    duplicados_completos = df.duplicated().sum()
    info_duplicados['duplicados_completos'] = duplicados_completos
    
    if duplicados_completos > 0:
        porcentaje_dup_comp = (duplicados_completos / len(df)) * 100
        info_duplicados['recomendaciones'].append(
            f"🚨 CRÍTICO: {duplicados_completos} duplicados completos ({porcentaje_dup_comp:.2f}%) - Revisar proceso de captura de datos"
        )
    else:
        info_duplicados['recomendaciones'].append("✅ No hay duplicados completos")
    
    # ANÁLISIS 2: DUPLICADOS PARCIALES EN VARIABLES CLAVE
    # Variables que típicamente identifican a una persona única
    variables_clave_posibles = [
        'EDAD', 'SEXO', 'NIVEL EDUCATIVO', 'ETNIA', 'ESTRATO', 
        'DEPARTAMENTO', 'MUNICIPIO', 'ZONA', 'P2'  # P2 podría ser ingreso
    ]
    
    # Encontrar qué variables clave existen en el dataset
    variables_clave_existentes = [var for var in variables_clave_posibles if var in df.columns]
    info_duplicados['variables_clave_encontradas'] = variables_clave_existentes
    
    if len(variables_clave_existentes) >= 2:  # Necesitamos al menos 2 variables para el análisis
        duplicados_parciales = df.duplicated(subset=variables_clave_existentes).sum()
        info_duplicados['duplicados_parciales'] = duplicados_parciales
        
        if duplicados_parciales > 0:
            porcentaje_dup_parc = (duplicados_parciales / len(df)) * 100
            if porcentaje_dup_parc > 20:  # Más del 20% es sospechoso
                info_duplicados['recomendaciones'].append(
                    f"⚠️ ATENCIÓN: {duplicados_parciales} duplicados parciales ({porcentaje_dup_parc:.2f}%) - Revisar si es normal para este tipo de encuesta"
                )
            else:
                info_duplicados['recomendaciones'].append(
                    f"ℹ️ INFO: {duplicados_parciales} duplicados parciales ({porcentaje_dup_parc:.2f}%) - Nivel normal para encuestas"
                )
        else:
            info_duplicados['recomendaciones'].append("✅ No hay duplicados en variables clave")
    else:
        info_duplicados['recomendaciones'].append(
            f"⚠️ Solo {len(variables_clave_existentes)} variables clave encontradas - Análisis de duplicados parciales limitado"
        )
    
    return info_duplicados

# =============================================================================
# HEADER PRINCIPAL Y NAVEGACIÓN
# =============================================================================

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
    ["🧹 Limpieza y Descriptivas", "📊 Resumen Ejecutivo", "🔍 Análisis Demográfico", 
     "🎪 Participación Cultural", "📚 Actividades Específicas"]
)

# Información sobre factor de expansión
st.sidebar.markdown("""
<div class="info-box">
    <h4>ℹ️ Factor de Expansión</h4>
    <p>Este dashboard aplica el <b>FACTOR DE EXPANSIÓN</b> en todos los análisis para garantizar representatividad estadística.</p>
    <p><small>Cada encuestado representa a un grupo de la población total según su peso muestral.</small></p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# FUNCIÓN PRINCIPAL DE CARGA Y PROCESAMIENTO DE DATOS
# =============================================================================

@st.cache_data
def load_and_process_data():
    """
    Carga y procesa los datos culturales desde el archivo Excel.
    
    PROCESO PASO A PASO:
    1. Carga el archivo cultura.xlsx
    2. Valida y limpia el factor de expansión
    3. Crea variables derivadas (grupos de edad, ingresos)
    4. Convierte variables SI/NO a numéricas
    5. Calcula índices de participación cultural
    6. Maneja errores y casos especiales
    
    Retorna:
    --------
    pandas.DataFrame: DataFrame procesado y listo para análisis
    """
    try:
        # PASO 1: CARGAR DATOS
        st.info("📂 Cargando datos desde cultura.xlsx...")
        data = pd.read_excel('cultura.xlsx')
        df = pd.DataFrame(data)
        
        # PASO 2: VALIDAR Y LIMPIAR FACTOR DE EXPANSIÓN
        if 'FACTOR DE EXPANSION' in df.columns:
            # Convertir a numérico, errores se convierten en NaN
            df['FACTOR DE EXPANSION'] = pd.to_numeric(df['FACTOR DE EXPANSION'], errors='coerce')
            
            # Contar valores problemáticos
            valores_nulos = df['FACTOR DE EXPANSION'].isnull().sum()
            if valores_nulos > 0:
                st.warning(f"⚠️ {valores_nulos} valores no válidos en FACTOR DE EXPANSION reemplazados con 1")
                df['FACTOR DE EXPANSION'].fillna(1, inplace=True)
                
            # Validar que no haya valores negativos o cero
            valores_invalidos = (df['FACTOR DE EXPANSION'] <= 0).sum()
            if valores_invalidos > 0:
                st.warning(f"⚠️ {valores_invalidos} valores ≤0 en FACTOR DE EXPANSION reemplazados con 1")
                df.loc[df['FACTOR DE EXPANSION'] <= 0, 'FACTOR DE EXPANSION'] = 1
        else:
            # Si no existe la columna, crear con valor 1
            st.warning("⚠️ No se encontró FACTOR DE EXPANSION. Se asume valor 1 para todos los registros.")
            df['FACTOR DE EXPANSION'] = 1
        
        # PASO 3: CREAR GRUPOS DE EDAD
        if 'EDAD' in df.columns:
            df['EDAD'] = pd.to_numeric(df['EDAD'], errors='coerce')
            
            # Definir grupos etarios según estándares demográficos
            df['grupo_edad'] = pd.cut(
                df['EDAD'], 
                bins=[0, 12, 18, 28, 40, 60, 100],
                labels=["Niñez (0-12)", "Adolescencia (13-18)", "Juventud (19-28)", 
                       "Adultez temprana (29-40)", "Adultez media (41-60)", "Adulto mayor (60+)"],
                include_lowest=True
            )
        
        # PASO 4: CREAR GRUPOS DE INGRESO
        if 'P2' in df.columns:
            df['P2'] = pd.to_numeric(df['P2'], errors='coerce')
            
            # Solo crear grupos si hay datos válidos suficientes
            if not df['P2'].isna().all():
                valid_incomes = df['P2'].dropna()
                
                if len(valid_incomes) > 10 and len(valid_incomes.unique()) >= 4:
                    try:
                        # Intentar quartiles primero
                        df['grupo_ingreso'] = pd.qcut(
                            df['P2'], 
                            q=4, 
                            labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'], 
                            duplicates='drop'
                        )
                    except ValueError:
                        # Si hay muchos valores duplicados, usar percentiles manuales
                        percentiles = np.percentile(valid_incomes, [25, 50, 75])
                        df['grupo_ingreso'] = pd.cut(
                            df['P2'],
                            bins=[-np.inf] + list(percentiles) + [np.inf],
                            labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto'],
                            include_lowest=True
                        )
                else:
                    df['grupo_ingreso'] = 'Sin clasificar'
            else:
                df['grupo_ingreso'] = np.nan
        
        # PASO 5: CONVERTIR VARIABLES SI/NO A NUMÉRICAS
        variables_convertidas = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Obtener valores únicos limpiando espacios y convirtiendo a mayúsculas
                unique_vals = set(df[col].dropna().astype(str).str.strip().str.upper().unique())
                
                # Verificar si es una variable SI/NO
                if unique_vals.issubset({'SI', 'NO', 'SÍ'}):  # Incluir versión con acento
                    # Crear columna numérica equivalente
                    col_num = f'{col.lower().replace(" ", "_")}_num'
                    df[col_num] = df[col].astype(str).str.strip().str.upper().map({
                        'SI': 1, 'SÍ': 1, 'NO': 0
                    })
                    # Manejar valores NaN como 0 (no participación)
                    df[col_num] = df[col_num].fillna(0).astype(int)
                    variables_convertidas.append((col, col_num))
        
        if variables_convertidas:
            st.success(f"✅ Convertidas {len(variables_convertidas)} variables SI/NO a numéricas")
        
        # PASO 6: CREAR ÍNDICE DE PARTICIPACIÓN CULTURAL
        # Variables que típicamente indican participación cultural
        cultural_vars_base = [
            'P3', 'P4', 'P5', 'ASISTENCIA BIBLIOTECA', 'ASISTENCIA CASAS DE CULTURA',
            'ASISTENCIA CENTROS CUTURALES', 'ASISTENCIA MUSEOS', 'ASISTENCIA EXPOSICIONES',
            'ASISTENCIA MONUMENTOS', 'ASISTENCIA CURSOS', 'PRACTICA CULTURAL', 'LECTURA LIBROS'
        ]
        
        # Identificar variables numéricas disponibles para el índice
        cultural_vars_numeric = []
        for var in cultural_vars_base:
            if var in df.columns:
                if pd.api.types.is_numeric_dtype(df[var]):
                    cultural_vars_numeric.append(var)
                else:
                    # Buscar versión numérica creada anteriormente
                    var_num = f'{var.lower().replace(" ", "_")}_num'
                    if var_num in df.columns:
                        cultural_vars_numeric.append(var_num)
        
        # Calcular índice cultural si hay variables disponibles
        if cultural_vars_numeric:
            # Asegurar que todas las variables sean numéricas
            for col in cultural_vars_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Sumar participación en diferentes actividades
            df['indice_cultural'] = df[cultural_vars_numeric].sum(axis=1, skipna=True)
            
            # Crear niveles de participación
            max_possible = len(cultural_vars_numeric)
            if max_possible >= 3:
                # Dividir en terciles
                tercio_1 = max_possible / 3
                tercio_2 = 2 * max_possible / 3
                
                df['nivel_participacion'] = pd.cut(
                    df['indice_cultural'],
                    bins=[-0.1, tercio_1, tercio_2, max_possible],
                    labels=['Bajo', 'Medio', 'Alto'],
                    include_lowest=True
                )
            else:
                # Si hay muy pocas variables, clasificación simple
                df['nivel_participacion'] = df['indice_cultural'].apply(
                    lambda x: 'Alto' if x >= max_possible * 0.7 else ('Medio' if x >= max_possible * 0.3 else 'Bajo')
                )
            
            st.success(f"✅ Índice cultural creado con {len(cultural_vars_numeric)} variables")
        else:
            # Si no hay variables, crear valores por defecto
            df['indice_cultural'] = 0
            df['nivel_participacion'] = 'Sin datos'
            st.warning("⚠️ No se encontraron variables para crear el índice cultural")
        
        # PASO 7: CALCULAR POBLACIÓN REPRESENTADA
        try:
            poblacion_total = df['FACTOR DE EXPANSION'].sum()
            df['poblacion_representada'] = df['FACTOR DE EXPANSION']
            st.success(f"✅ Población total representada: {poblacion_total:,.0f} personas")
        except Exception as e:
            df['poblacion_representada'] = 1
            st.error(f"Error calculando población representada: {str(e)}")
        
        return df
        
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'cultura.xlsx'. Asegúrate de que esté en la misma carpeta.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {str(e)}")
        st.error("Verifica que el archivo 'cultura.xlsx' tenga el formato correcto.")
        return pd.DataFrame()

# =============================================================================
# CARGAR DATOS PRINCIPALES
# =============================================================================

# Cargar datos con cache para mejor rendimiento
df = load_and_process_data()

# Verificar si hay datos cargados
if df.empty:
    st.error("❌ No se pudieron cargar los datos. Verifica el archivo 'cultura.xlsx'.")
    st.stop()

# =============================================================================
# PÁGINA: LIMPIEZA Y DESCRIPTIVAS
# =============================================================================

if page == "🧹 Limpieza y Descriptivas":
    st.markdown('<div class="section-header"><h2>🧹 Limpieza y Estadísticas Descriptivas</h2></div>', unsafe_allow_html=True)
    
    # SECCIÓN 1: INFORMACIÓN BÁSICA DEL DATASET
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("📋 Información General del Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 Total de Registros", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📋 Total de Variables", f"{len(df.columns)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_missing = df.isnull().sum().sum()
        st.metric("❌ Total de NAs", f"{total_missing:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        try:
            poblacion_total = df['FACTOR DE EXPANSION'].sum()
            st.metric("👥 Población Representada", f"{poblacion_total:,.0f}")
        except:
            st.metric("👥 Población Representada", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 2: APLICAR LIMPIEZA AUTOMÁTICA
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🔧 Limpieza Automática de Datos")
    
    # Explicación del proceso de limpieza
    st.markdown("""
    <div class="info-box">
    <h4>🔍 ¿Qué hace la limpieza automática?</h4>
    <ol>
    <li><strong>Elimina columnas con exceso de datos faltantes</strong> (>30% por defecto)</li>
    <li><strong>Imputa valores categóricos</strong> faltantes con "NO INFORMACION"</li>
    <li><strong>Imputa valores numéricos</strong> usando el método seleccionado (KNN por defecto)</li>
    <li><strong>Genera un reporte detallado</strong> de todos los cambios realizados</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuración de parámetros de limpieza
    col1, col2, col3 = st.columns(3)
    
    with col1:
        umbral_na = st.slider(
            "🎯 Umbral para eliminar columnas (% de NAs)",
            min_value=10, max_value=80, value=30, step=5,
            help="Columnas con más de este porcentaje de valores faltantes serán eliminadas"
        )
    
    with col2:
        metodo_numerico = st.selectbox(
            "🔢 Método para imputar valores numéricos",
            options=['knn', 'mediana', 'interpolacion', 'forward_fill'],
            help="KNN usa vecinos cercanos, mediana es robusto, interpolación para series temporales"
        )
    
    with col3:
        if metodo_numerico == 'knn':
            k_neighbors = st.number_input(
                "🎯 Número de vecinos (K)",
                min_value=1, max_value=20, value=5,
                help="Número de registros similares a considerar para la imputación"
            )
        else:
            k_neighbors = 5
    
    # Mostrar estado actual de limpieza
    total_nas_actual = df.isnull().sum().sum()
    porcentaje_nas_actual = (total_nas_actual / (len(df) * len(df.columns))) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        if total_nas_actual == 0:
            st.markdown("""
            <div class="success-box">
            <h4>✅ Datos ya están limpios</h4>
            <p>No se detectaron valores faltantes en el dataset actual.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
            <h4>⚠️ Datos requieren limpieza</h4>
            <p>Se detectaron <strong>{total_nas_actual:,}</strong> valores faltantes ({porcentaje_nas_actual:.2f}%)</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Botón para aplicar limpieza
        if st.button("🧹 Aplicar Limpieza Automática", type="primary", key="clean_data"):
            with st.spinner("🔄 Procesando limpieza de datos..."):
                # Aplicar la función de limpieza
                df_cleaned, info_limpieza = limpiar_e_imputar_datos(
                    df, 
                    umbral_na=umbral_na/100, 
                    metodo_numerico=metodo_numerico,
                    k_neighbors=k_neighbors
                )
                
                # Actualizar el dataframe (en una implementación real, esto debería manejarse con session state)
                # df = df_cleaned  # Comentado para evitar problemas con st.cache_data
                
                # Mostrar resultados
                mostrar_resultados_limpieza(info_limpieza)
                
                st.success("✅ ¡Limpieza completada exitosamente!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 3: ANÁLISIS DE VALORES FALTANTES
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🔥 Análisis de Valores Faltantes")
    
    # Calcular estadísticas de valores faltantes
    missing_data = pd.DataFrame({
        'Variable': df.columns,
        'Valores_Faltantes': df.isnull().sum(),
        'Porcentaje_Faltante': (df.isnull().sum() / len(df)) * 100,
        'Tipo_Dato': df.dtypes,
        'Valores_Únicos': [df[col].nunique() for col in df.columns]
    })
    missing_data = missing_data.sort_values('Porcentaje_Faltante', ascending=False)
    
    # Filtrar solo variables con valores faltantes para visualización
    missing_data_filtered = missing_data[missing_data['Porcentaje_Faltante'] > 0]
    
    if not missing_data_filtered.empty:
        # Gráfico de barras para variables con más valores faltantes
        fig = px.bar(
            missing_data_filtered.head(20),
            x='Variable',
            y='Porcentaje_Faltante',
            color='Porcentaje_Faltante',
            color_continuous_scale='Reds',
            title="Top 20 Variables con Mayor Porcentaje de Datos Faltantes",
            labels={'Porcentaje_Faltante': 'Porcentaje de Datos Faltantes (%)'}
        )
        
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Porcentaje de Datos Faltantes (%)",
            font=dict(size=11),
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
        # Añadir líneas de referencia para interpretación
        fig.add_hline(y=50, line_dash="dash", line_color="red", 
                     annotation_text="Crítico (>50%)", annotation_position="top right")
        fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                     annotation_text="Moderado (>20%)", annotation_position="top right")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretación automática
        criticas = missing_data_filtered[missing_data_filtered['Porcentaje_Faltante'] > 50]
        moderadas = missing_data_filtered[(missing_data_filtered['Porcentaje_Faltante'] > 20) & 
                                        (missing_data_filtered['Porcentaje_Faltante'] <= 50)]
        leves = missing_data_filtered[missing_data_filtered['Porcentaje_Faltante'] <= 20]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #dc3545;">
            <h4>🔴 Variables Críticas</h4>
            <p><strong>{len(criticas)}</strong> variables con >50% de NAs</p>
            <small>Recomendación: Eliminar del análisis</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #ffc107;">
            <h4>🟡 Variables Moderadas</h4>
            <p><strong>{len(moderadas)}</strong> variables con 20-50% de NAs</p>
            <small>Recomendación: Investigar patrón</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #28a745;">
            <h4>🟢 Variables Leves</h4>
            <p><strong>{len(leves)}</strong> variables con <20% de NAs</p>
            <small>Recomendación: Imputar valores</small>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.markdown("""
        <div class="success-box">
        <h3>🎉 ¡Excelente!</h3>
        <p>No se detectaron valores faltantes en el dataset. Los datos están completos y listos para análisis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 4: TABLA DETALLADA DE INFORMACIÓN POR VARIABLE
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("📊 Tabla Detallada por Variable")
    
    # Crear tabla completa con información detallada
    detailed_info = pd.DataFrame({
        'Variable': df.columns,
        'Tipo_Dato': df.dtypes.astype(str),
        'Registros_Totales': len(df),
        'Valores_Únicos': [df[col].nunique() for col in df.columns],
        'Valores_Faltantes': df.isnull().sum(),
        'Porcentaje_Faltante': round((df.isnull().sum() / len(df)) * 100, 2),
        'Valores_Completos': len(df) - df.isnull().sum(),
        'Porcentaje_Completo': round(((len(df) - df.isnull().sum()) / len(df)) * 100, 2)
    })
    
    # Añadir clasificación de calidad
    def clasificar_calidad(porcentaje_faltante):
        if porcentaje_faltante == 0:
            return "🟢 Excelente"
        elif porcentaje_faltante <= 5:
            return "🟢 Muy Buena"
        elif porcentaje_faltante <= 20:
            return "🟡 Buena"
        elif porcentaje_faltante <= 50:
            return "🟠 Regular"
        else:
            return "🔴 Crítica"
    
    detailed_info['Calidad'] = detailed_info['Porcentaje_Faltante'].apply(clasificar_calidad)
    
    # Mostrar tabla interactiva
    st.dataframe(
        detailed_info, 
        use_container_width=True, 
        height=400,
        column_config={
            "Porcentaje_Faltante": st.column_config.ProgressColumn(
                "% Faltante",
                help="Porcentaje de valores faltantes",
                min_value=0,
                max_value=100,
            ),
            "Porcentaje_Completo": st.column_config.ProgressColumn(
                "% Completo",
                help="Porcentaje de valores completos",
                min_value=0,
                max_value=100,
            )
        }
    )
    
    # Resumen estadístico
    st.markdown("### 📈 Resumen Estadístico de Calidad de Datos")
    
    calidad_counts = detailed_info['Calidad'].value_counts()
    
    cols = st.columns(len(calidad_counts))
    for i, (calidad, count) in enumerate(calidad_counts.items()):
        with cols[i]:
            st.metric(calidad, f"{count} variables")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 5: ANÁLISIS DETALLADO DE DUPLICADOS
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🔍 Análisis Detallado de Duplicados")
    
    # Explicación pedagógica sobre duplicados
    st.markdown("""
    <div class="info-box">
    <h4>📚 ¿Qué son los duplicados y por qué importan?</h4>
    <ul>
    <li><strong>Duplicados completos:</strong> Filas idénticas en todas las columnas. Usualmente indican errores de captura.</li>
    <li><strong>Duplicados parciales:</strong> Registros con mismas características demográficas (edad, sexo, educación, etc.).</li>
    <li><strong>En encuestas:</strong> Algunos duplicados parciales son normales (personas con características similares).</li>
    <li><strong>Problema:</strong> Duplicados completos sesgan los resultados y deben ser eliminados.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Realizar análisis detallado de duplicados
    info_duplicados = analizar_duplicados_detallado(df)
    
    # Mostrar métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        duplicados_completos = info_duplicados['duplicados_completos']
        st.metric("🔴 Duplicados Completos", duplicados_completos)
        if duplicados_completos > 0:
            porcentaje = (duplicados_completos / len(df)) * 100
            st.markdown(f"<small>{porcentaje:.2f}% del total</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        duplicados_parciales = info_duplicados['duplicados_parciales']
        st.metric("🟡 Duplicados Parciales", duplicados_parciales)
        if duplicados_parciales > 0:
            porcentaje = (duplicados_parciales / len(df)) * 100
            st.markdown(f"<small>{porcentaje:.2f}% del total</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        variables_clave = len(info_duplicados['variables_clave_encontradas'])
        st.metric("🔑 Variables Clave", variables_clave)
        st.markdown(f"<small>Para análisis parcial</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Mostrar variables clave encontradas
    if info_duplicados['variables_clave_encontradas']:
        st.markdown("**🔑 Variables clave encontradas para análisis de duplicados parciales:**")
        variables_str = ", ".join(info_duplicados['variables_clave_encontradas'])
        st.markdown(f"*{variables_str}*")
    
    # Mostrar recomendaciones
    st.markdown("### 💡 Recomendaciones")
    for recomendacion in info_duplicados['recomendaciones']:
        if "CRÍTICO" in recomendacion:
            st.markdown(f"""
            <div class="warning-box">
            {recomendacion}
            </div>
            """, unsafe_allow_html=True)
        elif "ATENCIÓN" in recomendacion:
            st.markdown(f"""
            <div class="warning-box">
            {recomendacion}
            </div>
            """, unsafe_allow_html=True)
        elif "✅" in recomendacion:
            st.markdown(f"""
            <div class="success-box">
            {recomendacion}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
            {recomendacion}
            </div>
            """, unsafe_allow_html=True)
    
    # Botón para eliminar duplicados completos si existen
    if info_duplicados['duplicados_completos'] > 0:
        if st.button("🗑️ Eliminar Duplicados Completos", type="secondary"):
            registros_antes = len(df)
            # df_sin_duplicados = df.drop_duplicates()  # En implementación real
            registros_despues = registros_antes - info_duplicados['duplicados_completos']
            
            st.success(f"✅ Se eliminarían {info_duplicados['duplicados_completos']} duplicados completos")
            st.info(f"📊 Registros: {registros_antes:,} → {registros_despues:,}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # SECCIÓN 6: INFORMACIÓN SOBRE EL FACTOR DE EXPANSIÓN
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("⚡ Información del Factor de Expansión")
    
    st.markdown("""
    <div class="info-box">
    <h4>📊 ¿Qué es el Factor de Expansión?</h4>
    <p>El factor de expansión es un peso estadístico que permite que cada encuestado represente 
    a un grupo específico de personas en la población total. Es fundamental para obtener 
    estimaciones representativas y precisas.</p>
    <ul>
    <li><strong>Propósito:</strong> Corregir sesgos de muestreo y representar la población real</li>
    <li><strong>Uso:</strong> Todos los análisis multiplican los datos por este factor</li>
    <li><strong>Interpretación:</strong> Un factor de 1000 significa que ese encuestado representa a 1000 personas</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if 'FACTOR DE EXPANSION' in df.columns:
        factor_stats = df['FACTOR DE EXPANSION'].describe()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Media", f"{factor_stats['mean']:.2f}")
        with col2:
            st.metric("📈 Mediana", f"{factor_stats['50%']:.2f}")
        with col3:
            st.metric("📉 Mínimo", f"{factor_stats['min']:.2f}")
        with col4:
            st.metric("📈 Máximo", f"{factor_stats['max']:.2f}")
        
        # Distribución del factor de expansión
        fig = px.histogram(
            df, 
            x='FACTOR DE EXPANSION',
            nbins=50,
            title="Distribución del Factor de Expansión",
            labels={'FACTOR DE EXPANSION': 'Factor de Expansión', 'count': 'Frecuencia'},
            color_discrete_sequence=['#6a0dad']
        )
        
        fig.update_layout(
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Verificar problemas en el factor de expansión
        factor_nulos = df['FACTOR DE EXPANSION'].isnull().sum()
        factor_ceros = (df['FACTOR DE EXPANSION'] == 0).sum()
        factor_negativos = (df['FACTOR DE EXPANSION'] < 0).sum()
        
        if factor_nulos > 0 or factor_ceros > 0 or factor_negativos > 0:
            st.markdown("### ⚠️ Problemas Detectados en Factor de Expansión")
            
            if factor_nulos > 0:
                st.warning(f"🔴 {factor_nulos} valores nulos detectados")
            if factor_ceros > 0:
                st.warning(f"🔴 {factor_ceros} valores igual a cero detectados")
            if factor_negativos > 0:
                st.warning(f"🔴 {factor_negativos} valores negativos detectados")
                
            st.info("💡 Estos valores se reemplazan automáticamente con 1 para mantener la integridad del análisis")
        else:
            st.success("✅ Factor de expansión sin problemas detectados")
    else:
        st.warning("⚠️ No se encontró la columna 'FACTOR DE EXPANSION' en los datos")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Análisis de variables categóricas
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🏷️ Análisis de Variables Categóricas")
    
    # Identificar variables categóricas (tipo object)
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_vars:
        selected_cat_var = st.selectbox("Selecciona una variable categórica para analizar:", categorical_vars)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Frecuencias de la variable seleccionada
            value_counts = df[selected_cat_var].value_counts(dropna=False)
            
            # Incluir NaNs en el conteo si existen
            if df[selected_cat_var].isnull().sum() > 0:
                value_counts.loc['NaN/Faltante'] = df[selected_cat_var].isnull().sum()
            
            fig = px.bar(x=value_counts.index, 
                        y=value_counts.values,
                        color=value_counts.values,
                        color_continuous_scale='Purples',
                        title=f"Distribución de {selected_cat_var}")
            
            fig.update_layout(
                xaxis_title="Categorías",
                yaxis_title="Frecuencia",
                font=dict(size=11),
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Estadísticas de la variable
            st.markdown("**Estadísticas de la Variable:**")
            
            stats_data = {
                'Métrica': ['Valores únicos', 'Valor más frecuente', 'Frecuencia máxima', 'Valores faltantes', '% Faltantes'],
                'Valor': [
                    df[selected_cat_var].nunique(),
                    df[selected_cat_var].mode().iloc[0] if not df[selected_cat_var].mode().empty else 'N/A',
                    df[selected_cat_var].value_counts().iloc[0] if not df[selected_cat_var].empty else 0,
                    df[selected_cat_var].isnull().sum(),
                    f"{(df[selected_cat_var].isnull().sum() / len(df)) * 100:.2f}%"
                ]
            }
                    

##############################################################################################################################################################################

# PÁGINA: RESUMEN EJECUTIVO
elif page == "📊 Resumen Ejecutivo":
    st.markdown('<div class="section-header"><h2>📊 Resumen Ejecutivo</h2></div>', unsafe_allow_html=True)
    
    # Métricas principales con factor de expansión
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        poblacion_total = int(df['FACTOR DE EXPANSION'].sum())
        st.metric("Población Representada", f"{poblacion_total:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Porcentaje de participación usando factor de expansión
        participation_high = ((df['indice_cultural'] > 3) * df['FACTOR DE EXPANSION']).sum() / df['FACTOR DE EXPANSION'].sum() * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Participación Cultural", f"{participation_high:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Edad promedio ponderada
        avg_age = (df['EDAD'] * df['FACTOR DE EXPANSION']).sum() / df['FACTOR DE EXPANSION'].sum()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Edad Promedio", f"{avg_age:.1f} años")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        # Porcentaje de lectura usando factor de expansión
        df_reading = df[df['LECTURA LIBROS'].isin(['SI', 'NO'])].copy()
        reading_rate = ((df_reading['LECTURA LIBROS'] == 'SI') * df_reading['FACTOR DE EXPANSION']).sum() / df_reading['FACTOR DE EXPANSION'].sum() * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Hábito de Lectura", f"{reading_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("🎭 Distribución por Nivel de Participación Cultural")
        
        # Calcular distribución con factor de expansión
        participation_data = df.groupby('nivel_participacion')['FACTOR DE EXPANSION'].sum().reset_index()
        participation_data.columns = ['Nivel de Participación', 'Población']
        participation_data['Porcentaje'] = participation_data['Población'] / participation_data['Población'].sum() * 100
        
        colors = get_purple_palette(len(participation_data))
        
        fig = px.pie(participation_data, 
                     values='Población', 
                     names='Nivel de Participación',
                     color_discrete_sequence=colors,
                     hole=0.4)
        
        fig.update_layout(
            annotations=[dict(text=f'Total: {int(participation_data["Población"].sum()):,}', 
                             x=0.5, y=0.5, font_size=12, showarrow=False)],
            font=dict(size=12)
        )
        
        # Agregar etiquetas con porcentajes
        fig.update_traces(textposition='outside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("👥 Distribución por Género y Edad")
        # Asegurarse de que hay datos válidos para el gráfico
        if not df['grupo_edad'].isna().all() and not df['SEXO'].isna().all():
            # Agrupar por género y grupo de edad, aplicando el factor de expansión
            gender_age_data = df.groupby(['grupo_edad', 'SEXO'])['FACTOR DE EXPANSION'].sum().reset_index()
            gender_age_data.columns = ['Grupo de Edad', 'Género', 'Población']
            
            fig = px.bar(gender_age_data, 
                         x='Grupo de Edad', 
                         y='Población', 
                         color='Género',
                         barmode='group',
                         text_auto='.2s',
                         color_discrete_sequence=['#6a0dad', '#ba55d3'])
            
            fig.update_layout(
                xaxis_title="Grupo de Edad",
                yaxis_title="Población Representada",
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para mostrar la distribución por género y edad.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Top actividades culturales con factor de expansión
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🏆 Top 10 Actividades Culturales Más Populares")
    
    # Función para calcular el porcentaje ponderado con factor de expansión
    def weighted_percentage(df, column, value='SI'):
        df_valid = df[df[column].isin(['SI', 'NO'])].copy()
        if df_valid.empty:
            return 0
        return ((df_valid[column] == value) * df_valid['FACTOR DE EXPANSION']).sum() / df_valid['FACTOR DE EXPANSION'].sum()
    
    activities = {
        'Lectura de Libros': weighted_percentage(df, 'LECTURA LIBROS'),
        'Práctica Cultural': weighted_percentage(df, 'PRACTICA CULTURAL'),
        'Asistencia a Bibliotecas': weighted_percentage(df, 'ASISTENCIA BIBLIOTECA'),
        'Conciertos/Música en Vivo': weighted_percentage(df, 'P4'),
        'Monumentos Históricos': weighted_percentage(df, 'ASISTENCIA MONUMENTOS'),
        'Teatro/Ópera/Danza': weighted_percentage(df, 'P3'),
        'Centros Culturales': weighted_percentage(df, 'ASISTENCIA CENTROS CUTURALES'),
        'Casas de Cultura': weighted_percentage(df, 'ASISTENCIA CASAS DE CULTURA'),
        'Cursos/Talleres': weighted_percentage(df, 'ASISTENCIA CURSOS'),
        'Museos': weighted_percentage(df, 'ASISTENCIA MUSEOS')
    }
    
    activities_df = pd.DataFrame(list(activities.items()), columns=['Actividad', 'Porcentaje'])
    activities_df = activities_df.sort_values('Porcentaje', ascending=True)
    activities_df['Porcentaje'] *= 100
    
    fig = px.bar(activities_df, x='Porcentaje', y='Actividad',
                orientation='h',
                color='Porcentaje',
                color_continuous_scale='Purples',
                text='Porcentaje')
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
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
    
    # Pirámide poblacional con FACTOR DE EXPANSION
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("👥 Pirámide Poblacional")
        
        # Verificar datos para la pirámide
        has_men = 'HOMBRE' in filtered_df['SEXO'].values and not filtered_df[filtered_df['SEXO'] == 'HOMBRE']['grupo_edad'].isna().all()
        has_women = 'MUJER' in filtered_df['SEXO'].values and not filtered_df[filtered_df['SEXO'] == 'MUJER']['grupo_edad'].isna().all()
        
        if has_men or has_women:
            # Crear datos para pirámide utilizando FACTOR DE EXPANSION
            if has_men:
                men_data = filtered_df[filtered_df['SEXO'] == 'HOMBRE'].groupby('grupo_edad')['FACTOR DE EXPANSION'].sum().sort_index()
            else:
                men_data = pd.Series()
                
            if has_women:
                women_data = filtered_df[filtered_df['SEXO'] == 'MUJER'].groupby('grupo_edad')['FACTOR DE EXPANSION'].sum().sort_index()
            else:
                women_data = pd.Series()
            
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
                xaxis_title="Población (Factor de Expansión)",
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
            # Utilizar FACTOR DE EXPANSION para la distribución educativa
            education_counts = filtered_df.groupby('NIVEL EDUCATIVO')['FACTOR DE EXPANSION'].sum()
            colors = get_purple_palette(len(education_counts))
            
            fig = px.pie(values=education_counts.values,
                names=education_counts.index,
                color_discrete_sequence=colors,
                hole=0.4)
            
            fig.update_layout(
                title="Distribución ponderada por Factor de Expansión"
            )
            
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
            
            # Box plot por género - usando pesos con FACTOR DE EXPANSION
            if valid_genders:
                for i, gender in enumerate(valid_genders):
                    gender_data = filtered_df[filtered_df['SEXO'] == gender]
                    data = gender_data['P2'].dropna()
                    weights = gender_data['FACTOR DE EXPANSION'].dropna()
                    
                    if not data.empty and not weights.empty:
                        # Repetir valores según el factor de expansión (redondeado)
                        expanded_data = []
                        for val, weight in zip(data, weights):
                            # Usar un factor de escala más pequeño para manejar los pesos
                            scaled_weight = max(1, round(weight / 10))
                            expanded_data.extend([val] * scaled_weight)
                        
                        if expanded_data:
                            fig.add_trace(
                                go.Box(y=expanded_data, name=gender, marker_color=get_purple_palette(len(valid_genders))[i]),
                                row=1, col=1
                            )
            
            # Box plot por educación - usando pesos con FACTOR DE EXPANSION
            if valid_education:
                for i, edu in enumerate(valid_education):
                    edu_data = filtered_df[filtered_df['NIVEL EDUCATIVO'] == edu]
                    data = edu_data['P2'].dropna()
                    weights = edu_data['FACTOR DE EXPANSION'].dropna()
                    
                    if not data.empty and not weights.empty:
                        # Repetir valores según el factor de expansión (redondeado)
                        expanded_data = []
                        for val, weight in zip(data, weights):
                            # Usar un factor de escala más pequeño para manejar los pesos
                            scaled_weight = max(1, round(weight / 10))
                            expanded_data.extend([val] * scaled_weight)
                        
                        if expanded_data:
                            fig.add_trace(
                                go.Box(y=expanded_data, name=edu, marker_color=get_purple_palette(len(valid_education))[i]),
                                row=1, col=2
                            )
            
            # Box plot por edad - usando pesos con FACTOR DE EXPANSION
            if valid_age_groups:
                for i, age in enumerate(valid_age_groups):
                    age_data = filtered_df[filtered_df['grupo_edad'] == age]
                    data = age_data['P2'].dropna()
                    weights = age_data['FACTOR DE EXPANSION'].dropna()
                    
                    if not data.empty and not weights.empty:
                        # Repetir valores según el factor de expansión (redondeado)
                        expanded_data = []
                        for val, weight in zip(data, weights):
                            # Usar un factor de escala más pequeño para manejar los pesos
                            scaled_weight = max(1, round(weight / 10))
                            expanded_data.extend([val] * scaled_weight)
                        
                        if expanded_data:
                            fig.add_trace(
                                go.Box(y=expanded_data, name=age, marker_color=get_purple_palette(len(valid_age_groups))[i]),
                                row=1, col=3
                            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                font=dict(size=10),
                title="Distribución de ingresos ponderada por Factor de Expansión"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos de ingresos para generar las gráficas.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No hay datos de ingresos disponibles para el análisis.")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
import warnings
warnings.filterwarnings('ignore')

def cramers_v(x, y, weights=None):
    """
    Calcula el coeficiente de Cramér's V para variables categóricas
    con soporte para pesos (factor de expansión)
    """
    try:
        if weights is not None:
            # Crear tabla de contingencia ponderada
            crosstab = pd.crosstab(x, y, weights, aggfunc='sum')
        else:
            crosstab = pd.crosstab(x, y)
        
        # Verificar que la tabla no esté vacía
        if crosstab.empty or crosstab.sum().sum() == 0:
            return 0.0
        
        # Calcular chi-cuadrado
        chi2, _, _, _ = chi2_contingency(crosstab)
        n = crosstab.sum().sum()
        
        # Calcular Cramér's V
        min_dim = min(crosstab.shape[0] - 1, crosstab.shape[1] - 1)
        if min_dim == 0:
            return 0.0
        
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        return min(cramers_v, 1.0)  # Limitar a 1.0
    except:
        return 0.0

def calculate_association_matrix(df, variables, weights=None):
    """
    Calcula matriz de asociación para variables categóricas
    """
    n_vars = len(variables)
    association_matrix = np.zeros((n_vars, n_vars))
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                association_matrix[i, j] = 1.0
            elif i < j:  # Solo calcular la mitad superior
                # Remover valores nulos
                valid_mask = df[var1].notna() & df[var2].notna()
                if weights is not None:
                    valid_mask = valid_mask & df[weights].notna()
                
                if valid_mask.sum() > 0:
                    x = df.loc[valid_mask, var1]
                    y = df.loc[valid_mask, var2]
                    w = df.loc[valid_mask, weights] if weights is not None else None
                    
                    assoc_value = cramers_v(x, y, w)
                    association_matrix[i, j] = assoc_value
                    association_matrix[j, i] = assoc_value  # Matriz simétrica
    
    return association_matrix

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
import warnings
warnings.filterwarnings('ignore')

def cramers_v(x, y, weights=None):
    """
    Calcula el coeficiente de Cramér's V para variables categóricas
    con soporte para pesos (factor de expansión)
    """
    try:
        if weights is not None:
            # Crear tabla de contingencia ponderada
            crosstab = pd.crosstab(x, y, weights, aggfunc='sum')
        else:
            crosstab = pd.crosstab(x, y)
        
        # Verificar que la tabla no esté vacía
        if crosstab.empty or crosstab.sum().sum() == 0:
            return 0.0
        
        # Calcular chi-cuadrado
        chi2, _, _, _ = chi2_contingency(crosstab)
        n = crosstab.sum().sum()
        
        # Calcular Cramér's V
        min_dim = min(crosstab.shape[0] - 1, crosstab.shape[1] - 1)
        if min_dim == 0:
            return 0.0
        
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        return min(cramers_v, 1.0)  # Limitar a 1.0
    except:
        return 0.0

def calculate_association_matrix(df, variables, weights=None):
    """
    Calcula matriz de asociación para variables categóricas
    """
    n_vars = len(variables)
    association_matrix = np.zeros((n_vars, n_vars))
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                association_matrix[i, j] = 1.0
            elif i < j:  # Solo calcular la mitad superior
                # Remover valores nulos
                valid_mask = df[var1].notna() & df[var2].notna()
                if weights is not None:
                    valid_mask = valid_mask & df[weights].notna()
                
                if valid_mask.sum() > 0:
                    x = df.loc[valid_mask, var1]
                    y = df.loc[valid_mask, var2]
                    w = df.loc[valid_mask, weights] if weights is not None else None
                    
                    assoc_value = cramers_v(x, y, w)
                    association_matrix[i, j] = assoc_value
                    association_matrix[j, i] = assoc_value  # Matriz simétrica
    
    return association_matrix

# PÁGINA: PARTICIPACIÓN CULTURAL
if page == "🎪 Participación Cultural":
    st.markdown('<div class="section-header"><h2>🎪 Análisis de Participación Cultural</h2></div>', unsafe_allow_html=True)
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
            'Casas de Cultura': 'ASISTENCIA CASAS DE CULTURA',
            'Centros Culturales': 'ASISTENCIA CENTROS CUTURALES',
            'Museos': 'ASISTENCIA MUSEOS',
            'Exposiciones': 'ASISTENCIA EXPOSICIONES',
            'Monumentos': 'ASISTENCIA MONUMENTOS',
            'Cursos': 'ASISTENCIA CURSOS'
        }
        
        # Verificar que hay datos de género
        if not df['SEXO'].isna().all():
            # Calcular participación por género ponderada por FACTOR DE EXPANSION
            gender_participation = {}
            
            for activity, col_name in cultural_activities.items():
                if col_name in df.columns:
                    # Para cada género, calcular la suma ponderada de participación
                    activity_by_gender = {}
                    for gender in df['SEXO'].dropna().unique():
                        gender_df = df[df['SEXO'] == gender]
                        yes_responses = gender_df[gender_df[col_name] == 'SI']['FACTOR DE EXPANSION'].sum()
                        total_weight = gender_df['FACTOR DE EXPANSION'].sum()
                        participation_pct = (yes_responses / total_weight * 100) if total_weight > 0 else 0
                        activity_by_gender[gender] = participation_pct
                    
                    gender_participation[activity] = activity_by_gender
            
            if gender_participation:
                # Convertir a DataFrame para graficación
                gender_participation_df = pd.DataFrame()
                
                for activity, gender_data in gender_participation.items():
                    temp_df = pd.DataFrame.from_dict(gender_data, orient='index', columns=[activity])
                    if gender_participation_df.empty:
                        gender_participation_df = temp_df
                    else:
                        gender_participation_df = gender_participation_df.join(temp_df, how='outer')
                
                # Transponer para el formato correcto
                gender_df = gender_participation_df.T
                
                if not gender_df.empty:
                    fig = px.bar(gender_df, 
                                barmode='group',
                                color_discrete_sequence=['#6a0dad', '#ba55d3'],
                                title="Participación Ponderada por Factor de Expansión")
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
            # Calcular participación cultural por etnia ponderada por FACTOR DE EXPANSION
            ethnicity_participation = {}
            
            for ethnicity in df['ETNIA'].dropna().unique():
                ethnicity_df = df[df['ETNIA'] == ethnicity]
                # Calcular el promedio ponderado del índice cultural
                avg_participation = np.average(
                    ethnicity_df['indice_cultural'].dropna(),
                    weights=ethnicity_df['FACTOR DE EXPANSION'].dropna()
                ) if not ethnicity_df['indice_cultural'].dropna().empty else 0
                ethnicity_participation[ethnicity] = avg_participation
            
            # Convertir a Series para graficación
            ethnicity_participation = pd.Series(ethnicity_participation).sort_values(ascending=True)
            
            if not ethnicity_participation.empty:
                colors = get_purple_palette(len(ethnicity_participation))
                
                fig = px.bar(x=ethnicity_participation.values,
                            y=ethnicity_participation.index,
                            orientation='h',
                            color=ethnicity_participation.values,
                            color_continuous_scale='Purples',
                            title="Participación Cultural Ponderada por Factor de Expansión")
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
    
    # Análisis de asociaciones para variables categóricas
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.subheader("🔗 Matriz de Asociaciones: Actividades Culturales")
    st.markdown("*Usando Coeficiente de Cramér's V para variables categóricas*")
    
    # Identificar variables culturales categóricas
    cultural_categorical_vars = []
    
    # Variables de asistencia (P2)
    asistencia_vars = [col for col in df.columns if col.lower().startswith('asistencia')]
    cultural_categorical_vars.extend(asistencia_vars)
    
    # Variables de frecuencia de actividades (P3, P4, P5)
    frecuencia_vars = [col for col in df.columns if any(x in col.lower() for x in ['p3', 'p4', 'p5']) 
                       and not col.lower().endswith('_num')]
    cultural_categorical_vars.extend(frecuencia_vars)
    
    # Variables de práctica cultural
    practica_vars = [col for col in df.columns if 'practica' in col.lower() and 'cultural' in col.lower()]
    cultural_categorical_vars.extend(practica_vars)
    
    # Variables de lectura
    lectura_vars = [col for col in df.columns if 'lectura' in col.lower() and not col.lower().endswith('_num')]
    cultural_categorical_vars.extend(lectura_vars)
    
    # Remover duplicados y verificar que existen en el DataFrame
    cultural_categorical_vars = list(set([var for var in cultural_categorical_vars if var in df.columns]))
    
    if len(cultural_categorical_vars) >= 2:
        # Filtrar variables con suficiente variación
        valid_vars = []
        for var in cultural_categorical_vars:
            if df[var].nunique() > 1 and df[var].notna().sum() > 10:
                valid_vars.append(var)
        
        if len(valid_vars) >= 2:
            # Limitar a máximo 10 variables para mejor visualización
            if len(valid_vars) > 10:
                # Seleccionar las variables con mayor variación
                var_info = [(var, df[var].nunique()) for var in valid_vars]
                var_info.sort(key=lambda x: x[1], reverse=True)
                valid_vars = [var for var, _ in var_info[:10]]
            
            st.info(f"Analizando {len(valid_vars)} variables culturales categóricas")
            
            # Calcular matriz de asociación
            try:
                association_matrix = calculate_association_matrix(df, valid_vars, 'FACTOR DE EXPANSION')
                
                # Crear nombres más legibles y únicos
                readable_names = []
                name_counts = {}
                
                for var in valid_vars:
                    if 'asistencia' in var.lower():
                        base_name = var.replace('ASISTENCIA ', '').replace('_', ' ').title()
                        if 'biblioteca' in var.lower():
                            name = 'Bibliotecas'
                        elif 'museo' in var.lower():
                            name = 'Museos'
                        elif 'casa' in var.lower():
                            name = 'Casas de Cultura'
                        elif 'centro' in var.lower():
                            name = 'Centros Culturales'
                        elif 'exposicion' in var.lower():
                            name = 'Exposiciones'
                        elif 'monumento' in var.lower():
                            name = 'Monumentos'
                        elif 'curso' in var.lower():
                            name = 'Cursos'
                        else:
                            name = base_name
                    elif 'p3' in var.lower():
                        name = 'Teatro/Danza'
                    elif 'p4' in var.lower():
                        name = 'Conciertos'
                    elif 'p5' in var.lower():
                        name = 'Música en Bares'
                    elif 'practica' in var.lower():
                        name = 'Práctica Cultural'
                    elif 'lectura' in var.lower():
                        if 'libro' in var.lower():
                            name = 'Lectura Libros'
                        elif 'revista' in var.lower():
                            name = 'Lectura Revistas'
                        elif 'periodico' in var.lower():
                            name = 'Lectura Periódicos'
                        else:
                            name = f"Lectura ({var.split('_')[-1] if '_' in var else 'General'})"
                    else:
                        name = var.replace('_', ' ').title()
                    
                    # Asegurar nombres únicos
                    original_name = name
                    counter = 1
                    while name in name_counts:
                        name = f"{original_name} ({counter})"
                        counter += 1
                    
                    name_counts[name] = True
                    readable_names.append(name)
                
                # Crear DataFrame para la visualización
                association_df = pd.DataFrame(
                    association_matrix,
                    index=readable_names,
                    columns=readable_names
                )
                
                # Crear heatmap
                fig = px.imshow(
                    association_df,
                    color_continuous_scale='Viridis',
                    aspect="auto",
                    title="Matriz de Asociaciones (Cramér's V) - Variables Categóricas<br><sub>Ponderado por Factor de Expansión</sub>",
                    labels=dict(color="Cramér's V")
                )
                
                fig.update_layout(
                    width=800,
                    height=600,
                    font=dict(size=10),
                    xaxis_title="",
                    yaxis_title="",
                    coloraxis_colorbar=dict(
                        title="Cramér's V<br>(0 = Sin asociación<br>1 = Asociación perfecta)"
                    )
                )
                
                # Añadir valores en las celdas
                fig.update_traces(
                    texttemplate="%{z:.2f}",
                    textfont={"size": 8}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretación
                st.markdown("### 📊 Interpretación de los Resultados:")
                st.markdown("""
                - **Cramér's V = 0**: No hay asociación entre las variables
                - **Cramér's V = 0.1-0.3**: Asociación débil
                - **Cramér's V = 0.3-0.6**: Asociación moderada  
                - **Cramér's V = 0.6-1.0**: Asociación fuerte
                """)
                
                # Mostrar las asociaciones más fuertes
                upper_triangle = np.triu(association_matrix, k=1)
                strong_associations = []
                
                for i in range(len(valid_vars)):
                    for j in range(i+1, len(valid_vars)):
                        if upper_triangle[i, j] > 0.3:  # Asociaciones moderadas o fuertes
                            strong_associations.append({
                                'Variable 1': readable_names[i],
                                'Variable 2': readable_names[j],
                                'Cramér\'s V': upper_triangle[i, j]
                            })
                
                if strong_associations:
                    st.markdown("### 🔍 Asociaciones Más Fuertes (Cramér's V > 0.3):")
                    strong_df = pd.DataFrame(strong_associations)
                    strong_df = strong_df.sort_values('Cramér\'s V', ascending=False)
                    st.dataframe(strong_df, use_container_width=True)
                else:
                    st.info("No se encontraron asociaciones fuertes (Cramér's V > 0.3) entre las variables analizadas.")
                    
            except Exception as e:
                st.error(f"Error al calcular las asociaciones: {str(e)}")
                st.info("Esto puede deberse a datos insuficientes o variables con muy poca variación.")
        else:
            st.warning("No hay suficientes variables categóricas con variación para el análisis de asociación.")
    else:
        st.warning("No se encontraron suficientes variables culturales categóricas para el análisis.")
    
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
        # Calcular tasa de participación ponderada por FACTOR DE EXPANSION
        if 'FACTOR DE EXPANSION' in df.columns:
            yes_weight = df[df[activity_col] == 'SI']['FACTOR DE EXPANSION'].sum()
            total_weight = df['FACTOR DE EXPANSION'].sum()
            participation_rate = (yes_weight / total_weight * 100) if total_weight > 0 else 0
        else:
            participation_rate = (df[activity_col] == 'SI').mean() * 100
            
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tasa de Participación", f"{participation_rate:.1f}%")
        st.markdown("<small>Ponderada por Factor de Expansión</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Calcular total de participantes expandido usando FACTOR DE EXPANSION
        if 'FACTOR DE EXPANSION' in df.columns:
            total_participants = df[df[activity_col] == 'SI']['FACTOR DE EXPANSION'].sum()
        else:
            total_participants = (df[activity_col] == 'SI').sum()
            
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Participantes", f"{int(total_participants):,}")
        st.markdown("<small>Expandido a la población</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Calcular el grupo demográfico con mayor participación ponderada
        if 'FACTOR DE EXPANSION' in df.columns:
            # Calcular tasa de participación ponderada por grupo de edad
            age_participation_rates = {}
            for age in df['grupo_edad'].dropna().unique():
                age_df = df[df['grupo_edad'] == age]
                yes_weight = age_df[age_df[activity_col] == 'SI']['FACTOR DE EXPANSION'].sum()
                total_weight = age_df['FACTOR DE EXPANSION'].sum()
                rate = (yes_weight / total_weight * 100) if total_weight > 0 else 0
                age_participation_rates[age] = rate
            
            # Encontrar el grupo con mayor tasa de participación
            max_group = max(age_participation_rates.items(), key=lambda x: x[1])[0]
        else:
            max_group = df.groupby('grupo_edad')[activity_col].apply(lambda x: (x == 'SI').mean()).idxmax()
            
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Grupo Más Activo", max_group)
        st.markdown("<small>Según Factor de Expansión</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Gráficos de análisis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader(f"📊 {selected_activity} por Grupo de Edad")
        
        # Calcular participación por edad ponderada por FACTOR DE EXPANSION
        age_participation = {}
        for age in df['grupo_edad'].dropna().unique():
            age_df = df[df['grupo_edad'] == age]
            yes_weight = age_df[age_df[activity_col] == 'SI']['FACTOR DE EXPANSION'].sum()
            total_weight = age_df['FACTOR DE EXPANSION'].sum()
            rate = (yes_weight / total_weight * 100) if total_weight > 0 else 0
            age_participation[age] = rate
        
        # Convertir a DataFrame para graficación
        age_participation_df = pd.DataFrame({
            'grupo_edad': list(age_participation.keys()),
            'participacion': list(age_participation.values())
        })
        
        fig = px.bar(age_participation_df,
                    x='grupo_edad',
                    y='participacion',
                    color='participacion',
                    color_continuous_scale='Purples',
                    title="Participación Ponderada por Factor de Expansión")
        fig.update_layout(
            xaxis_title="Grupo de Edad",
            yaxis_title="Porcentaje de Participación (%)",
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader(f"📈 {selected_activity} por Nivel Educativo")
        
        # Calcular participación por nivel educativo ponderada por FACTOR DE EXPANSION
        edu_participation = {}
        for edu in df['NIVEL EDUCATIVO'].dropna().unique():
            edu_df = df[df['NIVEL EDUCATIVO'] == edu]
            yes_weight = edu_df[edu_df[activity_col] == 'SI']['FACTOR DE EXPANSION'].sum()
            total_weight = edu_df['FACTOR DE EXPANSION'].sum()
            rate = (yes_weight / total_weight * 100) if total_weight > 0 else 0
            edu_participation[edu] = rate
        
        # Convertir a DataFrame para graficación
        edu_participation_df = pd.DataFrame({
            'nivel_educativo': list(edu_participation.keys()),
            'participacion': list(edu_participation.values())
        }).sort_values('participacion', ascending=False)
        
        fig = px.bar(edu_participation_df,
                    x='nivel_educativo',
                    y='participacion',
                    color='participacion',
                    color_continuous_scale='Purples',
                    title="Participación por Nivel Educativo (Ponderada)")
        fig.update_layout(
            xaxis_title="Nivel Educativo",
            yaxis_title="Porcentaje de Participación (%)",
            font=dict(size=12),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)