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
import requests
from io import BytesIO
import tempfile
import os
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

def download_google_drive_file(url):
    """Descarga un archivo de Google Drive dado un enlace compartido"""
    try:
        # Transformar el enlace compartido a enlace de descarga directa
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Descargar el archivo
        response = requests.get(download_url)
        response.raise_for_status()
        
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
            
        return tmp_path
    except Exception as e:
        st.error(f"Error al descargar el archivo: {str(e)}")
        return None

# =============================================
# CARGA Y PROCESAMIENTO DE DATOS
# =============================================
@st.cache_data
def load_data():
    try:
        # URL del archivo en Google Drive (acceso público)
        drive_url = "https://docs.google.com/spreadsheets/d/16VCwSt7qsHblFoFO2H331s5pJDEPHtRa/edit?usp=sharing"
        
        # Descargar el archivo temporalmente
        with st.spinner('Loading data...'):
            file_path = download_google_drive_file(drive_url)
            
            if file_path is None:
                return pd.DataFrame(), pd.DataFrame()
            
            # Leer el archivo Excel
            estado_cuenta = pd.read_excel(file_path, sheet_name='estado_de_cuenta', dtype={'NCF': str, 'Documento': str})
            comportamiento_pago = pd.read_excel(file_path, sheet_name='comportamiento_de_pago', dtype={'NCF': str, 'Documento': str})
            
            # Eliminar el archivo temporal
            os.unlink(file_path)
        
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
        st.error(f"Error al procesar los datos: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# =============================================
# CARGAR LOS DATOS
# =============================================
estado_cuenta, comportamiento_pago = load_data()

# Verificar si los datos se cargaron correctamente
if estado_cuenta.empty or comportamiento_pago.empty:
    st.error("No se pudieron cargar los datos. Verifica la ruta del archivo y la estructura.")
    st.stop()

# =============================================
# MODELO PREDICTIVO
# =============================================
try:
    # Preparar datos para el modelo
    X = estado_cuenta[['Dias', 'Inicial', 'Balance']].fillna(0)
    y = (estado_cuenta['Dias'] > 60).astype(int)  # Morosidad según definición
    
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
    **Análisis** del comportamiento de pagos, morosidad y riesgo crediticio de los clientes.
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
    
    # KPIs PRINCIPALES
    total_cartera = estado_cuenta['Balance'].sum()
    total_morosidad = estado_cuenta[estado_cuenta['Dias'] > 30]['Balance'].sum()
    clientes_morosos = estado_cuenta[estado_cuenta['Dias'] > 30]['Codigo'].nunique()
    porcentaje_morosidad = (total_morosidad / total_cartera) if total_cartera > 0 else 0
    dso = (estado_cuenta['Balance'] * estado_cuenta['Dias']).sum() / total_cartera if total_cartera > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total en Cartera", f"${total_cartera:,.2f}")
    with col2:
        st.metric("⚠️ Total en Morosidad", 
                f"${total_morosidad:,.2f}", 
                f"{porcentaje_morosidad:.1%} de la cartera")
    with col3:
        st.metric("👥 Clientes Morosos", clientes_morosos)
    with col4:
        st.metric("⏳ DSO Promedio", f"{dso:.0f} días")
    
    # GRÁFICOS DE TENDENCIA
    st.subheader("📈 Evolución Temporal", divider="gray")
    
    col1, col2 = st.columns(2)
    with col1:
        # Gráfico de evolución mensual
        estado_cuenta['Mes'] = estado_cuenta['Fecha_fatura'].dt.to_period('M').astype(str)
        evolucion_mensual = estado_cuenta.groupby('Mes')['Balance'].sum().reset_index()
        
        fig = px.line(
            evolucion_mensual,
            x='Mes',
            y='Balance',
            title='Evolución Mensual de la Cartera',
            labels={'Balance': 'Saldo ($)', 'Mes': 'Periodo'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribución por estado de morosidad
        distrib_morosidad = estado_cuenta.groupby('Estado_Morosidad')['Balance'].sum().reset_index()
        
        fig = px.pie(
            distrib_morosidad,
            names='Estado_Morosidad',
            values='Balance',
            title='Distribución por Estado de Morosidad',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ANÁLISIS POR MÉTODO DE PAGO (VERSIÓN CORREGIDA)
    st.subheader("💳 Análisis de Pagos Realizados", divider="gray")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not comportamiento_pago.empty and 'Pagado' in comportamiento_pago.columns:
            # Limpieza de datos de pagos
            comportamiento_pago['Pagado'] = (
                comportamiento_pago['Pagado']
                .astype(str)
                .str.replace(',', '')
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
            )
            # Gráfico de distribución de montos pagados
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
        if not comportamiento_pago.empty and 'Pagado' in comportamiento_pago.columns:
            # Resumen por método de pago
            metodos_pago = ['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']
            resumen = []
            
            for metodo in metodos_pago:
                if metodo in comportamiento_pago.columns:
                    # Limpiar método específico
                    comportamiento_pago[metodo] = (
                        comportamiento_pago[metodo]
                        .astype(str)
                        .str.replace(',', '')
                        .apply(pd.to_numeric, errors='coerce')
                        .fillna(0)
                    )
                    
                    # Filtrar solo transacciones con este método
                    mask = (comportamiento_pago[metodo] > 0)
                    total_pagado = comportamiento_pago.loc[mask, 'Pagado'].sum()
                    num_transacciones = mask.sum()
                    
                    resumen.append({
                        'Método': metodo,
                        'Total Pagado': total_pagado,
                        'Transacciones': num_transacciones
                    })
            
            # Crear y mostrar tabla
            df_resumen = pd.DataFrame(resumen)
            
            st.dataframe(
                df_resumen.assign(
                    **{
                        'Total Pagado': df_resumen['Total Pagado'].apply(lambda x: f"${x:,.2f}"),
                        '% del Total': df_resumen['Total Pagado'] / df_resumen['Total Pagado'].sum()
                    }
                ).sort_values('Total Pagado', ascending=False),
                column_config={
                    '% del Total': st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=1
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Estadísticas adicionales
            with st.expander("📌 Estadísticas Detalladas"):
                st.write(f"**Total general pagado:** ${comportamiento_pago['Pagado'].sum():,.2f}")
                st.write(f"**Pago promedio:** ${comportamiento_pago['Pagado'].mean():,.2f}")
                st.write(f"**Transacciones registradas:** {len(comportamiento_pago)}")
                st.write(f"**Clientes únicos:** {comportamiento_pago['Codigo'].nunique()}")
    
    # ANÁLISIS ADICIONAL
    st.subheader("🔍 Análisis Complementario", divider="gray")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 clientes con mayor saldo
        top_clientes = (
            estado_cuenta.groupby('Nombre Cliente')
            .agg({'Balance': 'sum', 'Dias': 'mean'})
            .nlargest(10, 'Balance')
            .reset_index()
        )
        
        fig = px.bar(
            top_clientes,
            x='Nombre Cliente',
            y='Balance',
            title='Top 10 Clientes por Saldo',
            labels={'Balance': 'Saldo ($)', 'Nombre Cliente': 'Cliente'},
            hover_data=['Dias']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribución por días de atraso
        fig = px.box(
            estado_cuenta,
            x='Dias',
            title='Distribución de Días de Atraso',
            labels={'Dias': 'Días de atraso'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💳 Análisis Detallado de Métodos de Pago", divider="gray")
    
    if not comportamiento_pago.empty and 'Pagado' in comportamiento_pago.columns:
        # Limpieza de datos
        comportamiento_pago['Pagado'] = (
            comportamiento_pago['Pagado']
            .astype(str)
            .str.replace(',', '')
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
        )
        # Preparar datos para la tabla
        metodos_pago = ['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']
        resultados = []
        
        for metodo in metodos_pago:
            if metodo in comportamiento_pago.columns:
                # Limpiar método específico
                comportamiento_pago[metodo] = (
                    comportamiento_pago[metodo]
                    .astype(str)
                    .str.replace(',', '')
                    .apply(pd.to_numeric, errors='coerce')
                    .fillna(0)
                )
                # Filtrar solo transacciones con este método
                mask_metodo = (comportamiento_pago[metodo] > 0)
                df_metodo = comportamiento_pago[mask_metodo].copy()
                
                # Obtener estado de morosidad si existe
                if not estado_cuenta.empty:
                    df_metodo = df_metodo.merge(
                        estado_cuenta[['Codigo', 'Estado_Morosidad']].drop_duplicates(),
                        on='Codigo',
                        how='left'
                    )
                    df_metodo['Estado_Morosidad'] = df_metodo['Estado_Morosidad'].fillna('Sin datos')
                else:
                    df_metodo['Estado_Morosidad'] = 'Sin datos'
                
                # Agrupar por estado de morosidad
                grupo = df_metodo.groupby('Estado_Morosidad').agg({
                    'Pagado': 'sum',
                    'Codigo': 'nunique',
                    metodo: 'count'
                }).reset_index()
                
                for _, row in grupo.iterrows():
                    resultados.append({
                        'Método': metodo,
                        'Estado Morosidad': row['Estado_Morosidad'],
                        'Total Pagado': row['Pagado'],
                        'Transacciones': row[metodo],
                        'Clientes': row['Codigo']
                    })
        
        # Crear DataFrame final
        if resultados:
            df_resultados = pd.DataFrame(resultados)
            total_general = df_resultados['Total Pagado'].sum()
            
            # Formatear tabla
            st.dataframe(
                df_resultados.assign(
                    **{
                        'Total Pagado': df_resultados['Total Pagado'].apply(lambda x: f"${x:,.2f}"),
                        '% del Total': df_resultados['Total Pagado'] / total_general
                    }
                ).sort_values(['Método', 'Total Pagado'], ascending=[True, False]),
                column_config={
                    '% del Total': st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=1,
                        width="medium"
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=500
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
    
    # Obtener todos los clientes únicos de AMBAS fuentes de datos
    clientes_estado = estado_cuenta[['Codigo', 'Nombre Cliente']].drop_duplicates()
    clientes_pago = comportamiento_pago[['Codigo', 'Nombre Cliente']].drop_duplicates()
    
    # Combinar ambos DataFrames y eliminar duplicados
    todos_clientes = pd.concat([clientes_estado, clientes_pago]).drop_duplicates('Codigo')
    
    # Crear opciones para el selectbox
    opciones_clientes = {
        row['Codigo']: f"{row['Codigo']} - {row['Nombre Cliente']}" 
        for _, row in todos_clientes.iterrows()
    }
    
    cliente_seleccionado = st.selectbox(
        "🔍 Buscar Cliente por Código o Nombre",
        options=list(opciones_clientes.keys()),
        format_func=lambda x: opciones_clientes[x]
    )
    
    # Obtener datos del cliente seleccionado de ambas fuentes
    cliente_data_estado = estado_cuenta[estado_cuenta['Codigo'] == cliente_seleccionado]
    cliente_data_pago = comportamiento_pago[comportamiento_pago['Codigo'] == cliente_seleccionado]
    
    # Determinar el nombre del cliente (puede venir de cualquiera de las dos fuentes)
    cliente_nombre = ""
    if not cliente_data_estado.empty:
        cliente_nombre = cliente_data_estado['Nombre Cliente'].iloc[0]
    elif not cliente_data_pago.empty:
        cliente_nombre = cliente_data_pago['Nombre Cliente'].iloc[0]
    else:
        st.error("No se encontró información para este cliente")
        st.stop()
    
    st.subheader(f"📋 Información General de {cliente_nombre}", divider="gray")
    
    # Mostrar diferentes métricas dependiendo de qué datos están disponibles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not cliente_data_estado.empty:
            st.metric("📅 Facturas Pendientes", len(cliente_data_estado))
            st.metric("💰 Balance Total", format_currency(cliente_data_estado['Balance'].sum()))
        else:
            st.metric("📅 Facturas Pendientes", "N/A")
            st.metric("💰 Balance Total", "N/A")
    
    with col2:
        if not cliente_data_estado.empty:
            st.metric("⏱️ Días de Atraso Promedio", f"{cliente_data_estado['Dias'].mean():.0f}")
            morosidad = (cliente_data_estado[cliente_data_estado['Dias'] > 60]['Balance'].sum() / 
                        cliente_data_estado['Balance'].sum() * 100) if cliente_data_estado['Balance'].sum() > 0 else 0
            st.metric("📉 Porcentaje en Morosidad", f"{morosidad:.1f}%")
        else:
            st.metric("⏱️ Días de Atraso Promedio", "N/A")
            st.metric("📉 Porcentaje en Morosidad", "N/A")
    
    with col3:
        if not cliente_data_estado.empty:
            st.metric("⚠️ Probabilidad de Morosidad", 
                     format_percent(cliente_data_estado['Probabilidad_Morosidad'].mean()))
            st.metric("💳 Límite Recomendado", 
                     format_currency(cliente_data_estado['Inicial'].mean() * 0.8))
        else:
            st.metric("⚠️ Probabilidad de Morosidad", "N/A")
            st.metric("💳 Límite Recomendado", "N/A")
    
    st.subheader("📅 Historial de Pagos", divider="gray")
    
    if not cliente_data_pago.empty:
        fig = px.line(
            cliente_data_pago.sort_values('Fecha_fatura'), 
            x='Fecha_fatura', 
            y='Pagado',
            title='Historial de Pagos',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        metodos_pago = cliente_data_pago[['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']].sum().reset_index()
        metodos_pago.columns = ['Método', 'Monto']
        fig = px.pie(
            metodos_pago, 
            names='Método', 
            values='Monto',
            title='Métodos de Pago Utilizados',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabla con últimos 5 pagos
        st.write("**Últimos Pagos Realizados:**")
        st.dataframe(
            cliente_data_pago.sort_values('Fecha_fatura', ascending=False).head(5)[[
                'Fecha_fatura', 'Pagado', 'Efectivo', 'Cheque', 'Tarjeta', 'Transferencia'
            ]].rename(columns={
                'Fecha_fatura': 'Fecha',
                'Pagado': 'Total Pagado'
            }),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No se encontró historial de pagos para este cliente")
    
    # Mostrar mensaje si no hay datos de estado de cuenta
    if cliente_data_estado.empty:
        st.info("ℹ️ Este cliente no tiene facturas pendientes en el estado de cuenta")

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
    
    # Obtener todos los clientes únicos de AMBAS fuentes de datos
    clientes_estado = estado_cuenta[['Codigo', 'Nombre Cliente']].drop_duplicates()
    clientes_pago = comportamiento_pago[['Codigo', 'Nombre Cliente']].drop_duplicates()
    todos_clientes = pd.concat([clientes_estado, clientes_pago]).drop_duplicates('Codigo')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Crear opciones para el selectbox
        opciones_clientes = {
            row['Codigo']: f"{row['Codigo']} - {row['Nombre Cliente']}" 
            for _, row in todos_clientes.iterrows()
        }
        
        cliente_sim = st.selectbox(
            "👤 Seleccione Cliente para Simulación",
            options=list(opciones_clientes.keys()),
            format_func=lambda x: opciones_clientes[x]
        )
        
        # Obtener datos del cliente de estado_cuenta (si existen)
        cliente_info_estado = estado_cuenta[estado_cuenta['Codigo'] == cliente_sim]
        
        # Obtener datos del cliente de comportamiento_pago (si existen)
        cliente_info_pago = comportamiento_pago[comportamiento_pago['Codigo'] == cliente_sim]
        
        # Determinar el nombre del cliente
        cliente_nombre = ""
        if not cliente_info_estado.empty:
            cliente_nombre = cliente_info_estado['Nombre Cliente'].iloc[0]
        elif not cliente_info_pago.empty:
            cliente_nombre = cliente_info_pago['Nombre Cliente'].iloc[0]
        
        # Mostrar métricas con valores por defecto si no hay datos en estado_cuenta
        balance_actual = cliente_info_estado['Balance'].sum() if not cliente_info_estado.empty else 0
        dias_atraso = cliente_info_estado['Dias'].mean() if not cliente_info_estado.empty else 0
        riesgo_actual = cliente_info_estado['Probabilidad_Morosidad'].mean() if not cliente_info_estado.empty else 0.3  # Valor por defecto
        
        st.metric("📊 Balance Actual", format_currency(balance_actual))
        st.metric("⏳ Días de Atraso", f"{dias_atraso:.0f}")
        st.metric("⚠️ Riesgo Actual", format_percent(riesgo_actual))
        
        # Obtener monto inicial para simulación (usar promedio de pagos si no hay estado_cuenta)
        if not cliente_info_estado.empty:
            monto_inicial = cliente_info_estado['Inicial'].mean()
        elif not cliente_info_pago.empty:
            monto_inicial = cliente_info_pago['Pagado'].mean()
        else:
            monto_inicial = 10000  # Valor por defecto
    
    with col2:
        monto_simular = st.number_input(
            "💰 Monto a Evaluar",
            min_value=0.0,
            value=float(monto_inicial),
            step=1000.0
        )
        
        plazo_simular = st.number_input(
            "📅 Plazo a Evaluar (días)",
            min_value=1,
            value=30,
            step=15
        )
        
        # Factores adicionales para la simulación
        st.markdown("**🔍 Factores Adicionales**")
        
        # Calcular historial de pagos basado en datos disponibles
        if not cliente_info_pago.empty:
            # Si tenemos datos de pagos, calcular puntuación basada en:
            # 1. Número total de pagos
            # 2. Consistencia en montos
            # 3. Frecuencia de pagos
            
            num_pagos = cliente_info_pago.shape[0]
            monto_std = cliente_info_pago['Pagado'].std()
            freq_pagos = (cliente_info_pago['Fecha_fatura'].max() - cliente_info_pago['Fecha_fatura'].min()).days / num_pagos if num_pagos > 1 else 30
            
            # Calcular puntuación (1-5)
            historial_valor = min(5, max(1, 
                int((num_pagos/10) +          # Más pagos = mejor
                (1 - monto_std/monto_inicial if monto_inicial > 0 else 0) +  # Menor variación = mejor
                (30/freq_pagos if freq_pagos > 0 else 1)    # Pagos más frecuentes = mejor
            )))
        else:
            historial_valor = 3  # Valor por defecto si no hay datos
            
        historial_pagos = st.slider(
            "Historial de Pagos (1 = Malo, 5 = Excelente)",
            min_value=1,
            max_value=5,
            value=historial_valor
        )
        
        # Determinar tipo de cliente basado en datos disponibles
        if not cliente_info_pago.empty:
            num_pagos = cliente_info_pago.shape[0]
            tipo_index = min(3, int(num_pagos / 5))  # Más pagos = mejor clasificación
        else:
            tipo_index = 0  # "Nuevo" por defecto
            
        tipo_cliente = st.selectbox(
            "Tipo de Cliente",
            options=["Nuevo", "Ocasional", "Recurrente", "Preferencial"],
            index=tipo_index
        )
        
        # Cálculo de riesgo simulado mejorado
        riesgo_base = riesgo_actual
        
        # Ajustar riesgo base según historial de pagos (si existe)
        if not cliente_info_pago.empty:
            # Calcular riesgo basado en variabilidad de pagos
            pagos_std = cliente_info_pago['Pagado'].std()
            riesgo_base = max(riesgo_base, min(0.7, pagos_std/(monto_inicial + 1e-6)))  # Más variación = mayor riesgo
            
        # Factores de ajuste
        factor_monto = min(monto_simular / (monto_inicial + 1e-6), 3)  # Evitar división por cero
        factor_plazo = plazo_simular / 30
        factor_historial = 1.5 - (historial_pagos * 0.1)  # Mejor historial reduce riesgo
        factor_tipo = {
            "Nuevo": 1.2,
            "Ocasional": 1.1,
            "Recurrente": 0.9,
            "Preferencial": 0.8
        }[tipo_cliente]
        
        # Cálculo final del riesgo
        riesgo_simulado = min(riesgo_base * factor_monto * factor_plazo * factor_historial * factor_tipo, 0.99)
        
        st.subheader("📈 Resultado de la Simulación", divider="gray")
        
        # Mostrar resultado con estilo según el nivel de riesgo
        if riesgo_simulado > 0.8:
            st.error(f"🚨 **Alto Riesgo** ({format_percent(riesgo_simulado)})")
        elif riesgo_simulado > 0.6:
            st.warning(f"⚠️ **Riesgo Alto** ({format_percent(riesgo_simulado)})")
        elif riesgo_simulado > 0.4:
            st.info(f"🔍 **Riesgo Moderado** ({format_percent(riesgo_simulado)})")
        else:
            st.success(f"✅ **Bajo Riesgo** ({format_percent(riesgo_simulado)})")
    
    st.subheader("📋 Recomendaciones Detalladas", divider="gray")
    
    # Sección de recomendaciones dinámicas
    if riesgo_simulado > 0.8:
        st.error("""
        **🚨 Acciones Recomendadas:**
        1. No aprobar crédito adicional sin garantías sólidas
        2. Exigir pagos adelantados (50% mínimo)
        3. Plazo máximo: 7 días
        4. Revisar histórico completo de pagos
        5. Considerar acciones legales preventivas
        6. Reducción de límite de crédito en 75%
        7. Supervisión diaria del caso
        """)
        
        # Mostrar advertencia adicional para clientes sin historial en estado_cuenta
        if cliente_info_estado.empty:
            st.warning("""
            **⚠️ Atención:** Este cliente no tiene facturas pendientes registradas, 
            pero el análisis de riesgo indica alta probabilidad de mora. 
            Verificar cuidadosamente su historial crediticio externo.
            """)
            
    elif riesgo_simulado > 0.6:
        st.warning("""
        **⚠️ Acciones Recomendadas:**
        1. Reducir límite de crédito en 50%
        2. Exigir aval o garantía
        3. Plazo máximo: 15 días
        4. Seguimiento semanal
        5. Descuentos por pronto pago (máx. 5%)
        6. Requerir historial bancario reciente
        7. Alertas tempranas automatizadas
        """)
        
    elif riesgo_simulado > 0.4:
        st.info("""
        **🔍 Acciones Recomendadas:**
        1. Mantener límite actual o aumento moderado (10-20%)
        2. Plazo máximo: 30 días
        3. Seguimiento mensual
        4. Planes de pago estructurados
        5. Recordatorios automáticos a 5 y 2 días antes
        6. Revisión trimestral de comportamiento
        7. Considerar garantías para montos altos
        """)
        
    else:
        st.success("""
        **✅ Acciones Recomendadas:**
        1. Considerar aumento de línea (20-30%)
        2. Plazos flexibles (hasta 60 días)
        3. Revisión semestral
        4. Beneficios por fidelidad (2% descuento)
        5. Proceso simplificado de aprobación
        6. Incluir en programas preferentes
        7. Evaluar créditos especiales con tasas preferenciales
        """)
    
    # Mostrar información adicional para clientes sin datos en estado_cuenta
    if cliente_info_estado.empty and not cliente_info_pago.empty:
        st.markdown("---")
        st.subheader("ℹ️ Información Adicional del Historial de Pagos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total de Pagos Registrados", cliente_info_pago.shape[0])
            st.metric("Promedio de Pagos", format_currency(cliente_info_pago['Pagado'].mean()))
            
        with col2:
            ultimo_pago = cliente_info_pago['Fecha_fatura'].max()
            st.metric("Último Pago", ultimo_pago.strftime("%d/%m/%Y") if not pd.isnull(ultimo_pago) else "N/A")
            
            monto_total = cliente_info_pago['Pagado'].sum()
            st.metric("Monto Total Pagado", format_currency(monto_total))

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
    **Dashboard cxc IDEMEFA**  
    Versión 2.0 - Junio 2024
""")
