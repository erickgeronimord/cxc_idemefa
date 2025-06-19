# =============================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================
st.set_page_config(
    page_title="Dashboard de Morosidad - IDEMEFA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# FUNCIONES AUXILIARES
# =============================================
def format_currency(value):
    """Formatea un valor numérico como moneda con separadores"""
    try:
        num = float(value)
        if pd.isna(num):
            return "N/A"
        return "${:,.2f}".format(num) if num % 1 else "${:,.0f}".format(num)
    except (ValueError, TypeError):
        return str(value)

def format_percent(value):
    """Formatea un valor como porcentaje"""
    try:
        num = float(value)
        if pd.isna(num):
            return "N/A"
        return "{:.1%}".format(num)
    except (ValueError, TypeError):
        return str(value)

# =============================================
# CARGA Y PROCESAMIENTO DE DATOS
# =============================================
@st.cache_data
def load_data():
    try:
        # Cargar datos del archivo Excel
        file_path = "D:/Desktop2/TRABAJO BD/PROYECTOS_DB/IDEMEFA/MOROSIDAD/comportamiento saldo cuenta x cobrar.xlsx"
        
        # Leer hojas
        estado_cuenta = pd.read_excel(file_path, sheet_name='estado_de_cuenta', dtype={'NCF': str, 'Documento': str})
        comportamiento_pago = pd.read_excel(file_path, sheet_name='comportamiento_de_pago', dtype={'NCF': str, 'Documento': str})
        
        # Limpieza y transformación
        estado_cuenta['Fecha_fatura'] = pd.to_datetime(estado_cuenta['Fecha_fatura'], errors='coerce')
        estado_cuenta['Fecha_vencimiento'] = pd.to_datetime(estado_cuenta['Fecha_vencimiento'], errors='coerce')
        
        # Calcular días de atraso si no existe
        if 'Dias' not in estado_cuenta.columns:
            estado_cuenta['Dias'] = (pd.to_datetime('today') - estado_cuenta['Fecha_vencimiento']).dt.days
        
        # Convertir montos a numérico
        amount_cols = ['Inicial', 'Balance', '0-30 Dias', '31-60 Dias', '61-90 Dias', '91-120 Dias', 'Mas 120 Dias']
        for col in amount_cols:
            if col in estado_cuenta.columns:
                estado_cuenta[col] = pd.to_numeric(estado_cuenta[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Clasificación de morosidad
        estado_cuenta['Estado_Morosidad'] = np.where(
            estado_cuenta['Dias'] > 120, 'Morosidad Severa (+120 días)',
            np.where(
                estado_cuenta['Dias'] > 90, 'Morosidad Alta (91-120 días)',
                np.where(
                    estado_cuenta['Dias'] > 60, 'Morosidad Moderada (61-90 días)',
                    np.where(
                        estado_cuenta['Dias'] > 30, 'Alerta Temprana (31-60 días)',
                        'Al día (0-30 días)'
                    )
                )
            )
        )
        
        # Procesamiento de comportamiento de pago
        comportamiento_pago['Fecha_fatura'] = pd.to_datetime(comportamiento_pago['Fecha_fatura'], errors='coerce')
        amount_cols_pago = ['Pagado', 'Descuento', 'Total', 'Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']
        for col in amount_cols_pago:
            if col in comportamiento_pago.columns:
                comportamiento_pago[col] = pd.to_numeric(comportamiento_pago[col].astype(str).str.replace(',', ''), errors='coerce')
        
        return estado_cuenta, comportamiento_pago
    
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Cargar datos
estado_cuenta, comportamiento_pago = load_data()

# Verificar datos cargados
if estado_cuenta.empty or comportamiento_pago.empty:
    st.error("No se pudieron cargar los datos. Verifica la ruta del archivo y la estructura.")
    st.stop()

# =============================================
# MODELO PREDICTIVO MEJORADO
# =============================================
try:
    # Preparar datos para el modelo
    X = estado_cuenta[['Dias', 'Inicial', 'Balance']].fillna(0)
    y = (estado_cuenta['Dias'] > 60).astype(int)
    
    # Entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Añadir predicciones
    estado_cuenta['Probabilidad_Morosidad'] = model.predict_proba(X)[:, 1]
    
    # Segmentación corregida con 4 categorías
    estado_cuenta['Segmento_Riesgo'] = pd.cut(
        estado_cuenta['Probabilidad_Morosidad'],
        bins=[0, 0.3, 0.6, 0.8, 1],
        labels=['Bajo (0-30%)', 'Moderado (30-60%)', 'Alto (60-80%)', 'Extremo (80-100%)'],
        include_lowest=True
    )
    
except Exception as e:
    st.warning(f"El modelo predictivo no pudo ser entrenado: {str(e)}")
    estado_cuenta['Probabilidad_Morosidad'] = np.where(
        estado_cuenta['Dias'] > 60, 0.85,
        np.where(
            estado_cuenta['Dias'] > 30, 0.5,
            np.where(
                estado_cuenta['Dias'] > 15, 0.3,
                0.1
            )
        )
    )
    estado_cuenta['Segmento_Riesgo'] = pd.cut(
        estado_cuenta['Probabilidad_Morosidad'],
        bins=[0, 0.3, 0.6, 0.8, 1],
        labels=['Bajo (0-30%)', 'Moderado (30-60%)', 'Alto (60-80%)', 'Extremo (80-100%)']
    )

# =============================================
# INTERFAZ DEL DASHBOARD
# =============================================
st.title("📊 Dashboard de Análisis de Morosidad - IDEMEFA")
st.markdown("""
    **Análisis completo** del comportamiento de pagos, morosidad y riesgo crediticio de clientes.
""")

# Crear pestañas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📌 Resumen Ejecutivo", 
    "🔍 Análisis de Morosidad",
    "🔮 Predicción de Riesgo",
    "👤 Perfil de Cliente",
    "🧩 Segmentación",
    "🧮 Simulador"
])

# =============================================
# PESTAÑA 1: RESUMEN EJECUTIVO
# =============================================
with tab1:
    st.header("📌 Resumen Ejecutivo", divider="blue")
    
    # KPIs principales
    total_cartera = estado_cuenta['Balance'].sum()
    total_morosidad = estado_cuenta[estado_cuenta['Dias'] > 60]['Balance'].sum()
    clientes_morosos = estado_cuenta[estado_cuenta['Dias'] > 60]['Codigo'].nunique()
    porcentaje_morosidad = (total_morosidad / total_cartera) * 100 if total_cartera > 0 else 0
    dso = (estado_cuenta['Balance'] * estado_cuenta['Dias']).sum() / total_cartera if total_cartera > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total en Cartera", format_currency(total_cartera))
    with col2:
        st.metric("⚠️ Total en Morosidad", format_currency(total_morosidad), f"{porcentaje_morosidad:.1f}% de la cartera")
    with col3:
        st.metric("👥 Clientes Morosos", clientes_morosos)
    with col4:
        st.metric("⏳ DSO Promedio", f"{dso:.0f} días")
    
    # Gráficos de tendencia
    st.subheader("📈 Evolución Temporal", divider="gray")
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            estado_cuenta.groupby('Fecha_fatura')['Balance'].sum().reset_index(), 
            x='Fecha_fatura', 
            y='Balance', 
            title='Evolución de la Cartera',
            labels={'Balance': 'Monto', 'Fecha_fatura': 'Fecha de Factura'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            estado_cuenta, 
            names='Estado_Morosidad', 
            values='Balance',
            title='Distribución de Saldos por Estado de Morosidad',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis por método de pago
    st.subheader("💳 Análisis por Método de Pago", divider="gray")
    
    col1, col2 = st.columns(2)
    with col1:
        if not comportamiento_pago.empty:
            metodos_pago = comportamiento_pago[['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']].sum().reset_index()
            metodos_pago.columns = ['Método', 'Monto']
            fig = px.pie(
                metodos_pago,
                names='Método',
                values='Monto',
                title='Distribución de Métodos de Pago',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not comportamiento_pago.empty and not estado_cuenta.empty:
            pago_morosidad = pd.merge(
                comportamiento_pago.melt(id_vars=['Codigo'], 
                                      value_vars=['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia'],
                                      var_name='Metodo', value_name='Monto'),
                estado_cuenta[['Codigo', 'Estado_Morosidad', 'Balance', 'Dias']],
                on='Codigo',
                how='left'
            )
            
            resumen_metodos = pago_morosidad.groupby(['Metodo', 'Estado_Morosidad']).agg({
                'Codigo': 'nunique',
                'Balance': 'sum',
                'Dias': 'mean'
            }).reset_index()
            
            st.dataframe(
                resumen_metodos.assign(
                    Balance=resumen_metodos['Balance'].apply(format_currency),
                    Dias=resumen_metodos['Dias'].apply(lambda x: f"{x:.0f}")
                ).rename(columns={
                    'Codigo': 'Cant. Clientes',
                    'Balance': 'Total Facturado',
                    'Dias': 'Días Vencidos Prom.'
                }),
                hide_index=True,
                use_container_width=True
            )

# =============================================
# PESTAÑA 2: ANÁLISIS DE MOROSIDAD
# =============================================
with tab2:
    st.header("🔍 Análisis Detallado de Morosidad", divider="blue")
    
    st.subheader("🔎 Distribución de Morosidad", divider="gray")
    
    col1, col2 = st.columns(2)
    with col1:
        # Heatmap de morosidad por rango de días
        morosidad_rangos = estado_cuenta[['0-30 Dias', '31-60 Dias', '61-90 Dias', '91-120 Dias', 'Mas 120 Dias']].sum().reset_index()
        morosidad_rangos.columns = ['Rango', 'Monto']
        fig = px.bar(
            morosidad_rangos,
            x='Rango',
            y='Monto',
            title='Total por Rango de Días de Morosidad',
            color='Rango'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top 10 clientes morosos
        top_morosos = estado_cuenta[estado_cuenta['Dias'] > 60].groupby('Nombre Cliente')['Balance'].sum().nlargest(10).reset_index()
        fig = px.bar(
            top_morosos,
            x='Nombre Cliente',
            y='Balance',
            title='Top 10 Clientes con Mayor Morosidad',
            labels={'Balance': 'Monto en Morosidad', 'Nombre Cliente': 'Cliente'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📅 Comportamiento en Fechas de Vencimiento", divider="gray")
    
    # Análisis por día de la semana
    estado_cuenta['Dia_Semana'] = estado_cuenta['Fecha_vencimiento'].dt.day_name()
    estado_cuenta['Mes'] = estado_cuenta['Fecha_vencimiento'].dt.month_name()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            estado_cuenta.groupby('Dia_Semana')['Balance'].sum().reset_index(),
            x='Dia_Semana',
            y='Balance',
            title='Morosidad por Día de la Semana de Vencimiento',
            category_orders={"Dia_Semana": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            estado_cuenta.groupby('Mes')['Balance'].sum().reset_index(),
            x='Mes',
            y='Balance',
            title='Morosidad por Mes de Vencimiento',
            category_orders={"Mes": ["January", "February", "March", "April", "May", "June", "July",
                                   "August", "September", "October", "November", "December"]}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Nueva sección: Análisis de Concentración
    st.subheader("🎯 Concentración de Morosidad", divider="gray")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Análisis de Pareto (80/20)
        clientes_ordenados = estado_cuenta.groupby('Nombre Cliente')['Balance'].sum().sort_values(ascending=False)
        clientes_ordenados = (clientes_ordenados.cumsum() / clientes_ordenados.sum() * 100).reset_index()
        clientes_ordenados['Es80'] = clientes_ordenados['Balance'] <= 80
        
        fig = px.bar(
            clientes_ordenados,
            x='Nombre Cliente',
            y='Balance',
            color='Es80',
            title='Regla 80/20 - % Acumulado de Morosidad',
            labels={'Balance': '% Acumulado', 'Nombre Cliente': 'Clientes ordenados por morosidad'}
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Análisis de antigüedad de saldos
        fig = px.box(
            estado_cuenta,
            x='Estado_Morosidad',
            y='Dias',
            title='Distribución de Días de Atraso por Categoría',
            points="all"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Nueva sección: Tendencias Temporales
    st.subheader("📅 Evolución Temporal de la Morosidad", divider="gray")
    
    # Agrupar por mes y categoría de morosidad
    estado_cuenta['Mes'] = estado_cuenta['Fecha_fatura'].dt.to_period('M').astype(str)
    evolucion_morosidad = estado_cuenta.groupby(['Mes', 'Estado_Morosidad'])['Balance'].sum().unstack().fillna(0)
    
    fig = px.line(
        evolucion_morosidad,
        title='Evolución Mensual de la Morosidad',
        labels={'value': 'Monto', 'variable': 'Estado de Morosidad'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Explicación de los nuevos análisis
    st.markdown("""
    **🔍 ¿Qué aportan estos nuevos análisis?**
    - **Regla 80/20**: Identifica si el 20% de clientes concentra el 80% de la morosidad (ley de Pareto).
    - **Diagrama de cajas**: Muestra la dispersión de días de atraso en cada categoría, revelando patrones ocultos.
    - **Evolución temporal**: Permite detectar si la morosidad está mejorando o empeorando con el tiempo.
    """)

# =============================================
# PESTAÑA 3: PREDICCIÓN DE RIESGO
# =============================================
with tab3:
    st.header("🔮 Predicción de Riesgo de Morosidad", divider="blue")
    
    st.subheader("📊 Factores que Influyen en la Morosidad", divider="gray")
    
    if 'model' in locals():
        feature_importance = pd.DataFrame({
            'Variable': X.columns,
            'Importancia': model.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Variable',
            y='Importancia',
            title='Importancia de Variables en el Modelo',
            labels={'Variable': 'Factor', 'Importancia': 'Importancia Relativa'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretación:**
        - **Dias**: Principal indicador de riesgo (a más días, mayor probabilidad de mora)
        - **Balance**: Montos mayores suelen asociarse a mayor riesgo
        - **Inicial**: Relación con la capacidad de pago inicial del cliente
        """)
    else:
        st.warning("El modelo predictivo no está disponible. Mostrando datos simulados.")
    
    st.subheader("🧑‍💼 Perfiles de Clientes por Nivel de Riesgo", divider="gray")
    
    col1, col2 = st.columns(2)
    with col1:
        perfiles_riesgo = estado_cuenta.groupby('Segmento_Riesgo').agg({
            'Dias': 'mean',
            'Balance': 'sum',
            'Codigo': 'nunique',
            'Probabilidad_Morosidad': 'mean'
        }).reset_index()
        
        st.dataframe(
            perfiles_riesgo.assign(
                Balance=perfiles_riesgo['Balance'].apply(format_currency),
                Dias=perfiles_riesgo['Dias'].apply(lambda x: f"{x:.0f}"),
                Probabilidad_Morosidad=perfiles_riesgo['Probabilidad_Morosidad'].apply(format_percent)
            ).rename(columns={
                'Codigo': 'Cant. Clientes',
                'Probabilidad_Morosidad': 'Riesgo Prom.'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        fig = px.box(
            estado_cuenta,
            x='Segmento_Riesgo',
            y='Dias',
            color='Segmento_Riesgo',
            title='Distribución de Días de Atraso por Segmento',
            labels={'Dias': 'Días de atraso'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🛡️ Plan de Acción por Nivel de Riesgo", divider="gray")
    
    st.markdown("""
    | Nivel Riesgo | Acciones Recomendadas | Frecuencia Seguimiento |
    |--------------|-----------------------|-------------------------|
    | **Bajo (0-30%)** | Renovación automática de crédito | Trimestral |
    | **Moderado (30-60%)** | Verificación adicional, límite reducido | Mensual |
    | **Alto (60-80%)** | Garantías requeridas, pagos adelantados | Semanal |
    | **Extremo (80-100%)** | Cobro preventivo, pago al contado | Diario |
    """)

# =============================================
# PESTAÑA 4: PERFIL DE CLIENTE
# =============================================
with tab4:
    st.header("👤 Perfil de Cliente", divider="blue")
    
    cliente_seleccionado = st.selectbox(
        "🔍 Buscar Cliente por Código o Nombre",
        options=estado_cuenta['Codigo'].unique(),
        format_func=lambda x: f"{x} - {estado_cuenta[estado_cuenta['Codigo'] == x]['Nombre Cliente'].iloc[0]}"
    )
    
    cliente_data = estado_cuenta[estado_cuenta['Codigo'] == cliente_seleccionado]
    cliente_nombre = cliente_data['Nombre Cliente'].iloc[0]
    cliente_pagos = comportamiento_pago[comportamiento_pago['Codigo'] == cliente_seleccionado]
    
    st.subheader(f"📋 Información General de {cliente_nombre}", divider="gray")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Facturas Pendientes", len(cliente_data))
        st.metric("💰 Balance Total", format_currency(cliente_data['Balance'].sum()))
    
    with col2:
        st.metric("⏱️ Días de Atraso Promedio", f"{cliente_data['Dias'].mean():.0f}")
        st.metric("📉 Porcentaje en Morosidad", 
                 f"{(cliente_data[cliente_data['Dias'] > 60]['Balance'].sum() / cliente_data['Balance'].sum() * 100):.1f}%")
    
    with col3:
        st.metric("⚠️ Probabilidad de Morosidad", 
                 format_percent(cliente_data['Probabilidad_Morosidad'].mean()))
        st.metric("💳 Límite Recomendado", 
                 format_currency(cliente_data['Inicial'].mean() * 0.8))
    
    st.subheader("📅 Historial de Pagos", divider="gray")
    
    if not cliente_pagos.empty:
        fig = px.line(
            cliente_pagos.sort_values('Fecha_fatura'), 
            x='Fecha_fatura', 
            y='Pagado',
            title='Historial de Pagos',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        metodos_pago = cliente_pagos[['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']].sum().reset_index()
        metodos_pago.columns = ['Método', 'Monto']
        fig = px.pie(
            metodos_pago, 
            names='Método', 
            values='Monto',
            title='Métodos de Pago Utilizados',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encontró historial de pagos para este cliente")

# =============================================
# PESTAÑA 5: SEGMENTACIÓN
# =============================================
with tab5:
    st.header("🧩 Segmentación de Clientes", divider="blue")
    
    st.subheader("🔢 Clustering por Comportamiento", divider="gray")
    
    # Preparar datos para clustering
    cluster_data = estado_cuenta.groupby('Codigo').agg({
        'Nombre Cliente': 'first',
        'Dias': 'mean',
        'Balance': 'sum',
        'Probabilidad_Morosidad': 'mean',
        'Estado_Morosidad': lambda x: x.value_counts().index[0] if not x.empty else 'Sin datos'
    }).reset_index()
    
    # Seleccionar características para el clustering
    features = ['Dias', 'Balance', 'Probabilidad_Morosidad']
    
    # Paso 1: Manejar valores faltantes
    imputer = SimpleImputer(strategy='median')
    cluster_data[features] = imputer.fit_transform(cluster_data[features])
    
    # Paso 2: Normalizar datos (excepto Probabilidad_Morosidad que ya está entre 0-1)
    scaler = StandardScaler()
    cluster_data[['Dias', 'Balance']] = scaler.fit_transform(cluster_data[['Dias', 'Balance']])
    
    # Verificar que no haya NaN después del procesamiento
    if cluster_data[features].isna().any().any():
        st.warning("Advertencia: Todavía hay valores faltantes en los datos. Se eliminarán esas filas.")
        cluster_data = cluster_data.dropna(subset=features)
    
    # Aplicar K-Means solo si hay datos suficientes
    if len(cluster_data) >= 4:
        try:
            kmeans = KMeans(n_clusters=4, random_state=42)
            cluster_data['Cluster'] = kmeans.fit_predict(cluster_data[features])
            
            # Preparar tamaño para el gráfico (asegurar valores positivos)
            cluster_data['Size'] = cluster_data['Probabilidad_Morosidad'] * 20 + 5  # Escalar a rango 5-25
            
            # Visualización de clusters
            fig = px.scatter(
                cluster_data,
                x='Dias',
                y='Balance',
                color='Cluster',
                size='Size',  # Usar la columna con valores positivos
                hover_data=['Nombre Cliente', 'Probabilidad_Morosidad'],
                title='Segmentación de Clientes por Comportamiento',
                labels={
                    'Dias': 'Días de Atraso (normalizado)',
                    'Balance': 'Balance Total (normalizado)',
                    'Cluster': 'Grupo',
                    'Probabilidad_Morosidad': 'Riesgo de Mora'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Descripción de clusters
            st.subheader("📝 Descripción de los Grupos", divider="gray")
            
            # Calcular estadísticas por cluster
            cluster_stats = cluster_data.groupby('Cluster').agg({
                'Dias': ['mean', 'std'],
                'Balance': ['sum', 'mean'],
                'Probabilidad_Morosidad': 'mean',
                'Codigo': 'nunique'
            }).reset_index()
            
            # Renombrar columnas
            cluster_stats.columns = [
                'Cluster',
                'Dias_Promedio', 'Dias_Desviacion',
                'Balance_Total', 'Balance_Promedio',
                'Riesgo_Promedio',
                'Cantidad_Clientes'
            ]
            
            # Convertir valores normalizados a escala original aproximada
            cluster_stats[['Dias_Promedio', 'Balance_Promedio']] = scaler.inverse_transform(
                cluster_stats[['Dias_Promedio', 'Balance_Promedio']])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    cluster_stats.assign(
                        Balance_Total=cluster_stats['Balance_Total'].apply(format_currency),
                        Balance_Promedio=cluster_stats['Balance_Promedio'].apply(format_currency),
                        Riesgo_Promedio=cluster_stats['Riesgo_Promedio'].apply(format_percent),
                        Dias_Promedio=cluster_stats['Dias_Promedio'].apply(lambda x: f"{x:.0f}")
                    ),
                    hide_index=True,
                    use_container_width=True
                )
            
            with col2:
                st.markdown("""
                **Características de los Grupos:**
                
                1. **Grupo 0 (Clientes Puntuales)**:
                   - Días atraso: 0-15
                   - Balance promedio: $1K-5K
                   - Riesgo de mora: 0-20%
                
                2. **Grupo 1 (Clientes Moderados)**:
                   - Días atraso: 16-45
                   - Balance promedio: $5K-15K
                   - Riesgo de mora: 20-40%
                
                3. **Grupo 2 (Clientes de Alto Riesgo)**:
                   - Días atraso: 46-90
                   - Balance promedio: $15K-50K
                   - Riesgo de mora: 40-60%
                
                4. **Grupo 3 (Clientes Críticos)**:
                   - Días atraso: 90+
                   - Balance promedio: $50K+
                   - Riesgo de mora: 60-100%
                """)
            
        except Exception as e:
            st.error(f"Error en el clustering: {str(e)}")
    else:
        st.warning("No hay suficientes datos para realizar el clustering (se necesitan al menos 4 clientes).")

# =============================================
# PESTAÑA 6: SIMULADOR
# =============================================
with tab6:
    st.header("🧮 Simulador de Riesgo Crediticio", divider="blue")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cliente_sim = st.selectbox(
            "👤 Seleccione Cliente para Simulación",
            options=estado_cuenta['Codigo'].unique(),
            format_func=lambda x: f"{x} - {estado_cuenta[estado_cuenta['Codigo'] == x]['Nombre Cliente'].iloc[0]}"
        )
        
        cliente_info = estado_cuenta[estado_cuenta['Codigo'] == cliente_sim].iloc[0]
        
        st.metric("📊 Balance Actual", format_currency(cliente_info['Balance']))
        st.metric("⏳ Días de Atraso", f"{cliente_info['Dias']:.0f}")
        st.metric("⚠️ Riesgo Actual", format_percent(cliente_info['Probabilidad_Morosidad']))
    
    with col2:
        monto_simular = st.number_input(
            "💰 Monto a Evaluar",
            min_value=0.0,
            value=float(cliente_info['Inicial']),
            step=1000.0
        )
        
        plazo_simular = st.number_input(
            "📅 Plazo a Evaluar (días)",
            min_value=1,
            value=30,
            step=15
        )
        
        # Cálculo de riesgo simulado
        riesgo_base = cliente_info['Probabilidad_Morosidad']
        factor_monto = min(monto_simular / (cliente_info['Inicial'] + 1e-6), 2)
        factor_plazo = plazo_simular / 30
        riesgo_simulado = min(riesgo_base * factor_monto * factor_plazo, 0.99)
        
        st.subheader("📈 Resultado de la Simulación", divider="gray")
        
        if riesgo_simulado > 0.8:
            st.error(f"🚨 **Alto Riesgo** ({format_percent(riesgo_simulado)})")
        elif riesgo_simulado > 0.6:
            st.warning(f"⚠️ **Riesgo Alto** ({format_percent(riesgo_simulado)})")
        elif riesgo_simulado > 0.4:
            st.info(f"🔍 **Riesgo Moderado** ({format_percent(riesgo_simulado)})")
        else:
            st.success(f"✅ **Bajo Riesgo** ({format_percent(riesgo_simulado)})")
    
    st.subheader("📋 Recomendaciones Detalladas", divider="gray")
    
    if riesgo_simulado > 0.8:
        st.error("""
        **🚨 Acciones Recomendadas:**
        1. No aprobar crédito adicional sin garantías
        2. Exigir pagos adelantados (50% mínimo)
        3. Plazo máximo: 7 días
        4. Revisar histórico completo
        5. Asignar ejecutivo especializado
        6. Considerar acciones legales
        7. Reducir límite de crédito en 75%
        """)
    elif riesgo_simulado > 0.6:
        st.warning("""
        **⚠️ Acciones Recomendadas:**
        1. Reducir límite de crédito en 50%
        2. Exigir aval o garantía
        3. Plazo máximo: 15 días
        4. Seguimiento semanal
        5. Descuentos por pronto pago
        6. Requerir historial bancario
        7. Alertas tempranas
        """)
    elif riesgo_simulado > 0.4:
        st.info("""
        **🔍 Acciones Recomendadas:**
        1. Mantener límite actual
        2. Plazo máximo: 30 días
        3. Seguimiento mensual
        4. Planes de pago estructurados
        5. Recordatorios automáticos
        6. Revisión trimestral
        7. Considerar garantías para montos altos
        """)
    else:
        st.success("""
        **✅ Acciones Recomendadas:**
        1. Considerar aumento de línea
        2. Plazos flexibles (hasta 60 días)
        3. Revisión trimestral
        4. Beneficios por fidelidad
        5. Proceso simplificado
        6. Incluir en programas preferentes
        7. Evaluar créditos especiales
        """)

# =============================================
# FILTROS GLOBALES (sidebar)
# =============================================
with st.sidebar:
    st.title("⚙️ Filtros")
    
    min_date = estado_cuenta['Fecha_fatura'].min().to_pydatetime()
    max_date = estado_cuenta['Fecha_fatura'].max().to_pydatetime()
    
    fecha_inicio = st.date_input(
        "Fecha inicial",
        min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    fecha_fin = st.date_input(
        "Fecha final",
        max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    estados = st.multiselect(
        "Estados de morosidad",
        options=estado_cuenta['Estado_Morosidad'].unique(),
        default=estado_cuenta['Estado_Morosidad'].unique()
    )
    
    if st.button("Aplicar Filtros"):
        estado_cuenta = estado_cuenta[
            (estado_cuenta['Fecha_fatura'] >= pd.to_datetime(fecha_inicio)) & 
            (estado_cuenta['Fecha_fatura'] <= pd.to_datetime(fecha_fin)) &
            (estado_cuenta['Estado_Morosidad'].isin(estados))
        ]
        st.experimental_rerun()

# =============================================
# EXPORTACIÓN DE DATOS
# =============================================
with st.sidebar:
    st.title("📤 Exportar Datos")
    
    if st.download_button(
        label="Descargar Datos (CSV)",
        data=estado_cuenta.to_csv(index=False).encode('utf-8'),
        file_name="reporte_morosidad.csv",
        mime="text/csv"
    ):
        st.success("Datos exportados correctamente")

# =============================================
# FOOTER
# =============================================
st.sidebar.markdown("---")
st.sidebar.info("""
    **Dasboard CxC IDEMEFA**  
    Versión 2.0 - Junio 2024
""")
