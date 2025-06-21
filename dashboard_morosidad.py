# =============================================
# IMPORTACIÓN DE LIBRERÍAS CON VERIFICACIÓN EXTENDIDA
# =============================================
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Verificación de entorno Python
if sys.version_info >= (3, 12):
    print("⚠️ Advertencia: Python 3.12+ puede tener problemas de compatibilidad", file=sys.stderr)

# Lista de paquetes críticos
CRITICAL_PACKAGES = [
    ('streamlit', '1.36.0'),
    ('pandas', '2.2.2'),
    ('numpy', '2.0.0'),
    ('plotly', '5.22.0'),
    ('scikit-learn', '1.5.0')
]

def check_packages():
    missing = []
    wrong_version = []
    
    for package, version in CRITICAL_PACKAGES:
        try:
            mod = __import__(package)
            current_version = getattr(mod, '__version__', '0.0.0')
            if not current_version.startswith(version.split('.')[0] + '.'):  # Compara versión mayor
                wrong_version.append(f"{package}=={version} (instalado: {current_version})")
        except ImportError:
            missing.append(package)
    
    return missing, wrong_version

missing, wrong_version = check_packages()

if missing or wrong_version:
    try:
        import streamlit as st
        st.error("🚨 Error Crítico en el Entorno")
        if missing:
            st.error(f"Paquetes faltantes: {', '.join(missing)}")
            st.code("pip install " + " ".join(missing), language="bash")
        if wrong_version:
            st.error(f"Versiones incorrectas: {', '.join(wrong_version)}")
            st.code("pip install --upgrade " + " ".join([p.split('==')[0] for p in wrong_version]), language="bash")
        
        st.markdown("""
        **Solución inmediata:**
        1. Verifica tu archivo `requirements.txt`
        2. Asegúrate de tener `runtime.txt` con `python-3.11.9`
        3. Espera 5 minutos tras actualizar los archivos
        """)
        st.stop()
    except:
        print("ERROR: Paquetes faltantes:", missing, file=sys.stderr)
        print("ERROR: Versiones incorrectas:", wrong_version, file=sys.stderr)
        sys.exit(1)

# Resto de tus importaciones
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================
# FUNCIONES AUXILIARES (modularizadas)
# =============================================
def format_currency(value):
    """Formatea valores numéricos como moneda"""
    if pd.isna(value):
        return "N/A"
    return "${:,.2f}".format(value) if value % 1 else "${:,.0f}".format(value)

def format_percent(value):
    """Formatea valores como porcentaje"""
    if pd.isna(value):
        return "N/A"
    return "{:.1%}".format(value)

def load_sample_data():
    """Genera datos de ejemplo para prueba"""
    n = 500
    dates = pd.date_range('2023-01-01', periods=12, freq='M')
    return (
        pd.DataFrame({
            'Codigo': np.random.choice(['C001', 'C002', 'C003', 'C004', 'C005'], n),
            'Nombre Cliente': np.random.choice(['Cliente A', 'Cliente B', 'Cliente C'], n),
            'Fecha_fatura': np.random.choice(dates, n),
            'Fecha_vencimiento': np.random.choice(dates, n) + pd.to_timedelta(np.random.randint(30, 90, n), unit='d'),
            'Inicial': np.random.uniform(1000, 50000, n).round(2),
            'Balance': np.random.uniform(0, 50000, n).round(2),
            'NCF': [f'NCF{str(i).zfill(6)}' for i in range(n)],
            'Documento': [f'DOC{str(i).zfill(5)}' for i in range(n)],
        }),
        pd.DataFrame({
            'Codigo': np.random.choice(['C001', 'C002', 'C003', 'C004', 'C005'], n),
            'Fecha_fatura': np.random.choice(dates, n),
            'Pagado': np.random.uniform(1000, 50000, n).round(2),
            'Efectivo': np.random.uniform(0, 5000, n).round(2),
            'Cheque': np.random.uniform(0, 5000, n).round(2),
            'Tarjeta': np.random.uniform(0, 5000, n).round(2),
            'Transferencia': np.random.uniform(0, 5000, n).round(2),
        })
    )

@st.cache_data(ttl=3600, show_spinner="Cargando datos...")
def load_and_process_data():
    """Carga y procesa los datos optimizados"""
    try:
        # Cargar datos de ejemplo (reemplazar con tu fuente real)
        estado_cuenta, comportamiento_pago = load_sample_data()
        
        # Procesamiento mínimo necesario
        date_cols = ['Fecha_fatura', 'Fecha_vencimiento']
        for col in date_cols:
            estado_cuenta[col] = pd.to_datetime(estado_cuenta[col], errors='coerce')
        
        # Calcular días de atraso
        estado_cuenta['Dias'] = (pd.to_datetime('today') - estado_cuenta['Fecha_vencimiento']).dt.days
        
        # Clasificación de morosidad (optimizada)
        conditions = [
            estado_cuenta['Dias'] > 120,
            estado_cuenta['Dias'] > 90,
            estado_cuenta['Dias'] > 60,
            estado_cuenta['Dias'] > 30
        ]
        choices = [
            'Morosidad Severa (+120 días)',
            'Morosidad Alta (91-120 días)',
            'Morosidad Moderada (61-90 días)',
            'Alerta Temprana (31-60 días)'
        ]
        estado_cuenta['Estado_Morosidad'] = np.select(conditions, choices, 'Al día (0-30 días)')
        
        return estado_cuenta, comportamiento_pago
    
    except Exception as e:
        st.error(f"Error al procesar datos: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def train_model(_estado_cuenta):
    """Entrena el modelo predictivo con caché"""
    try:
        X = _estado_cuenta[['Dias', 'Inicial', 'Balance']].fillna(0)
        y = (_estado_cuenta['Dias'] > 60).astype(int)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X, y)
        
        _estado_cuenta['Probabilidad_Morosidad'] = model.predict_proba(X)[:, 1]
        _estado_cuenta['Segmento_Riesgo'] = pd.cut(
            _estado_cuenta['Probabilidad_Morosidad'],
            bins=[0, 0.3, 0.6, 0.8, 1],
            labels=['Bajo (0-30%)', 'Moderado (30-60%)', 'Alto (60-80%)', 'Extremo (80-100%)']
        )
        
        return _estado_cuenta, model
    except:
        # Fallback si el modelo falla - VERSIÓN CORREGIDA
        _estado_cuenta['Probabilidad_Morosidad'] = np.where(
            _estado_cuenta['Dias'] > 60, 0.85,
            np.where(
                _estado_cuenta['Dias'] > 30, 0.5,
                np.where(
                    _estado_cuenta['Dias'] > 15, 0.3,
                    0.1
                )
            )
        )
        
        _estado_cuenta['Segmento_Riesgo'] = pd.cut(
            _estado_cuenta['Probabilidad_Morosidad'],
            bins=[0, 0.3, 0.6, 0.8, 1],
            labels=['Bajo (0-30%)', 'Moderado (30-60%)', 'Alto (60-80%)', 'Extremo (80-100%)']
        )
        return _estado_cuenta, None

# =============================================
# CARGAR DATOS Y MODELO
# =============================================
estado_cuenta, comportamiento_pago = load_and_process_data()
estado_cuenta, model = train_model(estado_cuenta)

# Verificación de datos
if estado_cuenta.empty:
    st.error("No se pudieron cargar los datos. Verifica la fuente de datos.")
    st.stop()

# =============================================
# INTERFAZ PRINCIPAL
# =============================================
st.title("📊 Dashboard de Análisis de Morosidad - IDEMEFA")
st.markdown("Análisis de comportamiento de pagos y riesgo crediticio")

# Crear pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Resumen Ejecutivo", 
    "🔍 Análisis de Morosidad",
    "🔮 Predicción de Riesgo",
    "👤 Perfil de Cliente"
])

# =============================================
# PESTAÑA 1: RESUMEN EJECUTIVO (optimizado)
# =============================================
with tab1:
    st.header("📌 Resumen Ejecutivo", divider="blue")
    
    # KPIs calculados una sola vez
    total_cartera = estado_cuenta['Balance'].sum()
    morosos_mask = estado_cuenta['Dias'] > 60
    total_morosidad = estado_cuenta.loc[morosos_mask, 'Balance'].sum()
    porcentaje_morosidad = (total_morosidad / total_cartera) if total_cartera > 0 else 0
    
    cols = st.columns(4)
    cols[0].metric("📊 Total en Cartera", format_currency(total_cartera))
    cols[1].metric("⚠️ Total en Morosidad", format_currency(total_morosidad), 
                  f"{porcentaje_morosidad:.1%} de la cartera")
    cols[2].metric("👥 Clientes Morosos", estado_cuenta.loc[morosos_mask, 'Codigo'].nunique())
    cols[3].metric("⏳ DSO Promedio", 
                  f"{(estado_cuenta['Balance'] * estado_cuenta['Dias']).sum() / total_cartera:.0f} días")
    
    # Gráficos optimizados
    st.subheader("📈 Distribución de Morosidad", divider="gray")
    
    fig = px.pie(
        estado_cuenta, 
        names='Estado_Morosidad', 
        values='Balance',
        title='Distribución por Estado de Morosidad',
        hole=0.3
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# PESTAÑA 2: ANÁLISIS DE MOROSIDAD
# =============================================
with tab2:
    st.header("🔍 Análisis Detallado de Morosidad", divider="blue")
    
    # Primero creamos la máscara booleana
    morosos_mask = estado_cuenta['Dias'] > 60
    
    # Luego aplicamos el filtro - versión corregida
    top_morosos = (
        estado_cuenta[morosos_mask]
        .groupby('Nombre Cliente')['Balance']
        .sum()
        .nlargest(10)
        .reset_index()  # Añadido para mejor visualización
    )
    
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(
            px.bar(
                top_morosos,
                x='Nombre Cliente',
                y='Balance',
                title='Top 10 Clientes Morosos',
                labels={'Balance': 'Monto en Morosidad', 'Nombre Cliente': 'Cliente'}
            ),
            use_container_width=True
        )
    
    with cols[1]:
        # Análisis por día de semana
        estado_cuenta['Dia_Semana'] = estado_cuenta['Fecha_vencimiento'].dt.day_name()
        dias_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        st.plotly_chart(
            px.bar(
                estado_cuenta.groupby('Dia_Semana')['Balance']
                .sum()
                .reindex(dias_order)
                .reset_index(),
                x='Dia_Semana',
                y='Balance',
                title='Morosidad por Día de Vencimiento'
            ),
            use_container_width=True
        )

# =============================================
# PESTAÑA 3: PREDICCIÓN DE RIESGO
# =============================================
with tab3:
    st.header("🔮 Predicción de Riesgo de Morosidad", divider="blue")
    
    if model:
        # Importancia de características (calculada una vez)
        feature_imp = pd.DataFrame({
            'Variable': ['Dias', 'Inicial', 'Balance'],
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        st.plotly_chart(
            px.bar(
                feature_imp,
                x='Variable',
                y='Importancia',
                title='Importancia de Variables en el Modelo'
            ),
            use_container_width=True
        )
    
    # Segmentos de riesgo (ya calculados)
    st.plotly_chart(
        px.box(
            estado_cuenta,
            x='Segmento_Riesgo',
            y='Dias',
            color='Segmento_Riesgo',
            title='Distribución de Días por Segmento de Riesgo'
        ),
        use_container_width=True
    )

# =============================================
# PESTAÑA 4: PERFIL DE CLIENTE (optimizado)
# =============================================
with tab4:
    st.header("👤 Perfil de Cliente", divider="blue")
    
    cliente_options = estado_cuenta[['Codigo', 'Nombre Cliente']].drop_duplicates()
    cliente_seleccionado = st.selectbox(
        "Seleccionar Cliente",
        options=cliente_options['Codigo'],
        format_func=lambda x: f"{x} - {cliente_options[cliente_options['Codigo'] == x]['Nombre Cliente'].iloc[0]}"
    )
    
    # Datos del cliente (filtrados una vez)
    cliente_data = estado_cuenta[estado_cuenta['Codigo'] == cliente_seleccionado]
    cliente_pagos = comportamiento_pago[comportamiento_pago['Codigo'] == cliente_seleccionado]
    
    cols = st.columns(3)
    cols[0].metric("📅 Facturas Pendientes", len(cliente_data))
    cols[1].metric("💰 Balance Total", format_currency(cliente_data['Balance'].sum()))
    cols[2].metric("⚠️ Riesgo Promedio", 
                  format_percent(cliente_data['Probabilidad_Morosidad'].mean()))
    
    if not cliente_pagos.empty:
        st.plotly_chart(
            px.line(
                cliente_pagos.sort_values('Fecha_fatura'),
                x='Fecha_fatura',
                y='Pagado',
                title='Historial de Pagos',
                markers=True
            ),
            use_container_width=True
        )

# =============================================
# FILTROS GLOBALES (sidebar optimizado)
# =============================================
with st.sidebar:
    st.title("⚙️ Filtros")
    
    min_date = estado_cuenta['Fecha_fatura'].min().date()
    max_date = estado_cuenta['Fecha_fatura'].max().date()
    
    fecha_inicio, fecha_fin = st.date_input(
        "Rango de fechas",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    estados_seleccionados = st.multiselect(
        "Estados de morosidad",
        options=estado_cuenta['Estado_Morosidad'].unique(),
        default=estado_cuenta['Estado_Morosidad'].unique()
    )

# Aplicar filtros (reactivo)
if fecha_inicio and fecha_fin and estados_seleccionados:
    filtered_data = estado_cuenta[
        (estado_cuenta['Fecha_fatura'].dt.date >= fecha_inicio) &
        (estado_cuenta['Fecha_fatura'].dt.date <= fecha_fin) &
        (estado_cuenta['Estado_Morosidad'].isin(estados_seleccionados))
    ]
    # Actualizar los datos mostrados
    estado_cuenta = filtered_data

# =============================================
# FOOTER
# =============================================
st.sidebar.markdown("---")
st.sidebar.info("""
    **Dashboard de Morosidad**  
    Versión optimizada - 2024
""")
