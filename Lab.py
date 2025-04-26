#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
import re

# Configuración global de la página
st.set_page_config(
    page_title="Sistema Integrado de Análisis Vectorial",
    page_icon="🦟",
    layout="wide"
)

# Función global para cargar el archivo de Excel
@st.cache_data
def cargar_excel(sheet_name):
    """Carga una hoja específica del archivo Excel compilado."""
    try:
        df = pd.read_excel("compilado.xlsx", sheet_name=sheet_name)
        # Limpiar nombres de columnas (eliminar espacios en blanco)
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        return df
    except Exception as e:
        #st.error(f"Error al cargar datos de la hoja {sheet_name}: {str(e)}")
        return pd.DataFrame()

# Título principal y descripción de la aplicación
st.title("🦟 Sistema Integrado de Análisis Vectorial")

# Selector de dashboard en la parte superior izquierda
dashboard_options = {
    "Lugares de Visita (Hoja1)": "hoja1", 
    "Control Vectorial - Viviendas (Hoja2)": "hoja2",
    "Control Vectorial - Equipos (Hoja3)": "hoja3",
    "Análisis Temporal (Hoja4)": "hoja4",
    "Análisis Integrado (Hoja1 + Hoja4)": "hoja41",
    "Hospitales (Hoja5)": "hoja5",
    "Protección en IPS (Hoja6)": "hoja6"
}

# Crear contenedor para el selector
selector_container = st.container()

with selector_container:
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_dashboard = st.selectbox(
            "Seleccionar Dashboard:",
            options=list(dashboard_options.keys())
        )
    with col2:
        st.markdown("""
        Esta plataforma integra diferentes herramientas de análisis de datos para el monitoreo 
        y control vectorial. Seleccione el dashboard deseado del menú desplegable.
        """)

# Obtener la clave del dashboard seleccionado
dashboard_key = dashboard_options[selected_dashboard]

# Separador visual
st.markdown("---")

# Contenedor principal para el dashboard seleccionado
dashboard_container = st.container()

with dashboard_container:
    # =====================================================================
    # Dashboard 1: LUGARES DE VISITA (HOJA1)
    # =====================================================================
    if dashboard_key == "hoja1":
        st.header("📍 Explorador de Lugares de Visita")
        st.markdown("""
        Este dashboard permite analizar los diferentes lugares de visita registrados en los municipios,
        incluyendo centros de salud, plazas de mercado, e instituciones educativas rurales.
        """)

        # Cargar datos
        df_places = cargar_excel("Hoja1")

        if not df_places.empty:
            # Calcular totales
            if ('Centro salud' in df_places.columns and 
                'Plazas de mercado' in df_places.columns and 
                'Terminal o terminalito' in df_places.columns and 
                'IPS (entorno)' in df_places.columns):

                df_places['Total Lugares'] = (
                    df_places['Centro salud'] + 
                    df_places['Plazas de mercado'] + 
                    df_places['Terminal o terminalito'] + 
                    df_places['IPS (entorno)']
                )

            # Obtener municipios
            municipios = sorted(df_places['Municipio'].unique())
            municipios_opciones = ['Todos'] + list(municipios)

            # Widget de selección múltiple para municipios
            municipios_seleccionados = st.multiselect(
                'Selecciona Municipio(s):', 
                municipios_opciones, 
                default=['Todos']
            )

            # Filtrar DataFrame
            if 'Todos' in municipios_seleccionados or not municipios_seleccionados:
                df_filtrado = df_places
                titulo = 'Cantidad Total de Lugares de Visita por Municipio'
            else:
                df_filtrado = df_places[df_places['Municipio'].isin(municipios_seleccionados)]
                titulo = f'Cantidad Total de Lugares de Visita para: {", ".join(municipios_seleccionados)}'

            # Gráfico principal
            fig = px.bar(
                df_filtrado, 
                x='Municipio', 
                y='Total Lugares',
                title=titulo, 
                labels={'Total Lugares': 'Total'},
                color='Municipio'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Información adicional por municipio
            if municipios_seleccionados and 'Todos' not in municipios_seleccionados:
                st.subheader('Información Adicional por Municipio Seleccionado')

                df_seleccionado = df_places[df_places['Municipio'].isin(municipios_seleccionados)].copy()

                # Crear columnas para información detallada
                col1, col2 = st.columns(2)

                with col1:
                    # Detalle de tipos de lugar
                    st.subheader('Distribución por Tipo de Lugar')
                    tipos_lugar = ['Centro salud', 'Plazas de mercado', 'Terminal o terminalito', 'IPS (entorno)']
                    df_tipos = pd.melt(
                        df_seleccionado, 
                        id_vars=['Municipio'],
                        value_vars=tipos_lugar,
                        var_name='Tipo de Lugar',
                        value_name='Cantidad'
                    )
                    fig_tipos = px.bar(
                        df_tipos, 
                        x='Tipo de Lugar', 
                        y='Cantidad', 
                        color='Municipio',
                        title=f'Tipos de Lugar en {", ".join(municipios_seleccionados)}',
                        barmode='group'
                    )
                    st.plotly_chart(fig_tipos, use_container_width=True)

                with col2:
                    # Frecuencia de Sedes
                    if 'Sedes' in df_seleccionado.columns:
                        st.subheader('Frecuencia de Sedes')
                        # Procesamiento de sedes (solo si hay datos de sede)
                        sedes_validas = df_seleccionado['Sedes'].dropna()
                        if not sedes_validas.empty:
                            # Extraer y contar sedes
                            todas_sedes = []
                            for sede_str in sedes_validas:
                                if isinstance(sede_str, str):
                                    sedes = [s.strip() for s in sede_str.split(',') if s.strip()]
                                    todas_sedes.extend(sedes)

                            if todas_sedes:
                                conteo_sedes = pd.Series(todas_sedes).value_counts().nlargest(10).reset_index()
                                conteo_sedes.columns = ['Sede', 'Frecuencia']
                                fig_sedes = px.bar(
                                    conteo_sedes, 
                                    x='Sede', 
                                    y='Frecuencia', 
                                    title=f'Top 10 Sedes en: {", ".join(municipios_seleccionados)}'
                                )
                                st.plotly_chart(fig_sedes, use_container_width=True)
                            else:
                                st.info("No hay datos de sedes disponibles para mostrar.")
                        else:
                            st.info("No hay datos de sedes disponibles para mostrar.")

                # Información sobre I.E. (zona rural)
                if 'I.E. (zona rural)' in df_seleccionado.columns:
                    st.subheader('Instituciones Educativas Rurales')

                    # Procesamiento de I.E. rurales
                    df_seleccionado['Nombre_IE'] = df_seleccionado['I.E. (zona rural)'].apply(
                        lambda x: x.split('(')[0].strip() if isinstance(x, str) and '(' in x else x if isinstance(x, str) else None
                    )

                    # Frecuencia de Nombres de I.E.
                    conteo_ie = df_seleccionado['Nombre_IE'].value_counts().nlargest(10).reset_index()
                    conteo_ie.columns = ['Nombre_IE', 'Frecuencia']
                    fig_ie = px.bar(conteo_ie, x='Nombre_IE', y='Frecuencia', 
                                    title=f'Top 10 I.E. (Zona Rural) en: {", ".join(municipios_seleccionados)}')
                    st.plotly_chart(fig_ie, use_container_width=True)

                    # Frecuencia de Actividades de I.E.
                    df_seleccionado['Actividades_IE'] = df_seleccionado['I.E. (zona rural)'].apply(
                        lambda x: [item.strip() for item in x.split('(')[1].replace(')', '').split(';') 
                                if isinstance(x, str) and '(' in x and ')' in x] 
                        if isinstance(x, str) and '(' in x and ')' in x else []
                    )
                    lista_actividades = df_seleccionado['Actividades_IE'].explode().dropna()
                    if not lista_actividades.empty:
                        conteo_actividades = lista_actividades.value_counts().nlargest(10).reset_index()
                        conteo_actividades.columns = ['Actividad', 'Frecuencia']
                        st.subheader('Frecuencia de Actividades en I.E. (Zona Rural)')
                        fig_actividades = px.bar(conteo_actividades, x='Actividad', y='Frecuencia', 
                                                title=f'Top 10 Actividades en I.E. en: {", ".join(municipios_seleccionados)}')
                        st.plotly_chart(fig_actividades, use_container_width=True)
        else:
            st.warning("No se pudieron cargar los datos de lugares de visita. Verifique que el archivo Excel esté disponible y contenga la Hoja1.")

        # Información de pie
        st.markdown("---")
        st.caption("Dashboard de Lugares de Visita (Hoja1) - Desarrollado en Ibagué, Tolima")

    # =====================================================================
    # Dashboard 2: CONTROL VECTORIAL - VIVIENDAS (HOJA2)
    # =====================================================================
    elif dashboard_key == "hoja2":
        st.header("🏘️ Análisis de Control Vectorial en Viviendas")
        st.markdown("""
        Este dashboard permite analizar las intervenciones de control vectorial 
        en viviendas, incluyendo aplicación de larvicidas y fumigación.
        """)

        # Cargar datos
        df_vector = cargar_excel("Hoja2")

        if not df_vector.empty:
            # Convertir columnas con información dentro de paréntesis a numérico
            def extract_numeric(text):
                if isinstance(text, str):
                    match = re.search(r'(\d+)', text)
                    return int(match.group(1)) if match else None
                return text

            # Procesar columnas numéricas si existen
            if 'Viviendas con aplicación de Larvicida' in df_vector.columns:
                df_vector['Viviendas Larvicida Num'] = df_vector['Viviendas con aplicación de Larvicida'].apply(extract_numeric)

            if 'Control Químico Fumigación' in df_vector.columns:
                df_vector['Viviendas Fumigadas Num'] = df_vector['Control Químico Fumigación'].apply(extract_numeric)

            # Filtro por Municipio
            municipios = sorted(df_vector['Municipio'].unique())
            municipios_seleccionados = st.multiselect(
                'Selecciona Municipio(s):',
                ['Todos'] + list(municipios),
                default=['Todos']
            )

            # Filtro por Fecha
            if 'Fecha' in df_vector.columns:
                df_vector['Fecha'] = pd.to_datetime(df_vector['Fecha'], errors='coerce')
                min_date = df_vector['Fecha'].min().date()
                max_date = df_vector['Fecha'].max().date()

                fecha_seleccionada = st.date_input(
                    'Selecciona el rango de fechas:',
                    (min_date, max_date)
                )
            else:
                st.warning("La columna 'Fecha' no está disponible en los datos.")
                min_date = datetime.now().date() - timedelta(days=30)
                max_date = datetime.now().date()
                fecha_seleccionada = (min_date, max_date)

            # Filtrar el DataFrame basado en la selección del usuario
            df_filtrado = df_vector.copy()

            if 'Todos' not in municipios_seleccionados and municipios_seleccionados:
                df_filtrado = df_filtrado[df_filtrado['Municipio'].isin(municipios_seleccionados)]

            if 'Fecha' in df_filtrado.columns and len(fecha_seleccionada) == 2:
                start_date, end_date = fecha_seleccionada
                df_filtrado = df_filtrado[(df_filtrado['Fecha'].dt.date >= start_date) & 
                                          (df_filtrado['Fecha'].dt.date <= end_date)]

            if df_filtrado.empty:
                st.warning("No hay datos disponibles para la selección realizada.")
            else:
                if 'Todos' in municipios_seleccionados or not municipios_seleccionados:
                    groupby_col = 'Municipio'
                    title_suffix = 'por Municipio'
                else:
                    groupby_col = 'Vereda y/o Centro poblado'
                    title_suffix = f'en {", ".join(municipios_seleccionados)} por Vereda/Centro Poblado'

                # Verificar columnas necesarias
                col1, col2 = st.columns(2)
                with col1:
                    if all(col in df_filtrado.columns for col in ['Viviendas existentes en vereda', 'Viviendas Larvicida Num']):
                        # Gráfico de Viviendas Existentes vs. Viviendas con Larvicida
                        st.subheader(f'Viviendas Existentes vs. Viviendas con Aplicación de Larvicida {title_suffix}')
                        df_larvicida = df_filtrado.groupby(groupby_col)[['Viviendas existentes en vereda', 'Viviendas Larvicida Num']].sum().reset_index()
                        df_larvicida = df_larvicida.rename(columns={'Viviendas existentes en vereda': 'Existentes', 'Viviendas Larvicida Num': 'Con Larvicida'})

                        # Convertir a formato largo para crear el gráfico
                        df_larvicida_long = pd.melt(
                            df_larvicida,
                            id_vars=[groupby_col],
                            value_vars=['Existentes', 'Con Larvicida'],
                            var_name='Tipo',
                            value_name='Cantidad'
                        )

                        fig_larvicida = px.bar(
                            df_larvicida_long,
                            x=groupby_col,
                            y='Cantidad',
                            color='Tipo',
                            title=f'Comparación de Viviendas con Larvicida {title_suffix}',
                            labels={'Cantidad': 'Número de Viviendas', groupby_col: groupby_col},
                            barmode='group'
                        )
                        st.plotly_chart(fig_larvicida, use_container_width=True)
                    else:
                        st.warning("No se encontraron algunas columnas necesarias para la visualización de larvicidas.")

                with col2:
                    # Verificar columna de fumigación
                    if 'Viviendas Fumigadas Num' in df_filtrado.columns:
                        # Gráfico de Viviendas con Fumigación
                        st.subheader(f'Viviendas con Control Químico (Fumigación) {title_suffix}')
                        df_fumigacion = df_filtrado.groupby(groupby_col)['Viviendas Fumigadas Num'].sum().reset_index()
                        df_fumigacion = df_fumigacion.rename(columns={'Viviendas Fumigadas Num': 'Fumigadas'})

                        fig_fumigacion = px.bar(
                            df_fumigacion,
                            x=groupby_col,
                            y='Fumigadas',
                            color=groupby_col,
                            title=f'Número de Viviendas Fumigadas {title_suffix}',
                            labels={'Fumigadas': 'Número de Viviendas', groupby_col: groupby_col}
                        )
                        st.plotly_chart(fig_fumigacion, use_container_width=True)
                    else:
                        st.warning("No se encontró la columna necesaria para la visualización de fumigación.")

                # Mostrar tabla de datos
                st.subheader("Datos Detallados")
                st.dataframe(df_filtrado, use_container_width=True)
        else:
            st.warning("No se pudieron cargar los datos de control vectorial. Verifique que el archivo Excel esté disponible y contenga la Hoja2.")

        # Información de pie
        st.markdown("---")
        st.caption("Dashboard de Control Vectorial en Viviendas (Hoja2) - Análisis de Control Vectorial en Municipios del Tolima")

    # =====================================================================
    # Dashboard 3: CONTROL VECTORIAL - EQUIPOS (HOJA3)
    # =====================================================================
    elif dashboard_key == "hoja3":
        st.header("🔬 Dashboard de Control Vectorial por Equipos")
        st.markdown("Visualización de actividades de control vectorial por municipios y tipos de equipos")

        # Función para cargar los datos específicos de este dashboard
        @st.cache_data
        def load_data():
            try:
                df = cargar_excel("Hoja3")

                # Asegurar que la columna de fecha sea tipo datetime
                if 'Fecha' in df.columns:
                    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

                    # Crear columnas adicionales para análisis
                    df['Mes'] = df['Fecha'].dt.month
                    df['Año'] = df['Fecha'].dt.year
                    df['Mes-Año'] = df['Fecha'].dt.strftime('%B-%Y')

                # Normalizar los tipos de equipos para el análisis
                # Eliminar guiones al principio si existen
                if 'Observaciones' in df.columns:
                    df['Observaciones'] = df['Observaciones'].str.replace(r'^-\s*', '', regex=True)

                return df
            except Exception as e:
                st.error(f"Error al cargar datos: {e}")
                # Retornar un DataFrame vacío en caso de error
                return pd.DataFrame(columns=['Fecha', 'Municipio', 'Barrios', 'Observaciones'])

        # Cargar los datos
        df = load_data()

        # Sidebar con filtros
        st.sidebar.header("Filtros")

        # Filtro de municipios
        all_municipios = sorted(df['Municipio'].unique())
        selected_municipios = st.sidebar.multiselect(
            "Seleccionar Municipios:",
            options=all_municipios,
            default=all_municipios
        )

        # Filtro de tipos de equipo
        if 'Observaciones' in df.columns:
            all_equipos = sorted(df['Observaciones'].unique())
            selected_equipos = st.sidebar.multiselect(
                "Seleccionar Tipos de Equipo:",
                options=all_equipos,
                default=all_equipos
            )
        else:
            selected_equipos = []
            st.sidebar.warning("No se encontró la columna 'Observaciones' para filtrar equipos.")

        # Filtro de rango de fechas
        if 'Fecha' in df.columns:
            min_date = df['Fecha'].min().date() if not df.empty else datetime(2020, 1, 1).date()
            max_date = df['Fecha'].max().date() if not df.empty else datetime(2025, 12, 31).date()

            date_range = st.sidebar.date_input(
                "Rango de Fechas:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            # Asegurar que tenemos dos fechas
            if len(date_range) == 2:
                start_date, end_date = date_range
                # Convertir las fechas a datetime para filtrar el DataFrame
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date) + timedelta(days=1)  # Incluir el día final completo
            else:
                start_date = pd.to_datetime(min_date)
                end_date = pd.to_datetime(max_date) + timedelta(days=1)
        else:
            start_date = pd.to_datetime("2020-01-01")
            end_date = pd.to_datetime("2025-12-31")
            st.sidebar.warning("No se encontró la columna 'Fecha' para filtrar por fechas.")

        # Aplicar filtros
        filtered_df = df.copy()

        if selected_municipios:
            filtered_df = filtered_df[filtered_df['Municipio'].isin(selected_municipios)]

        if selected_equipos and 'Observaciones' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Observaciones'].isin(selected_equipos)]

        if 'Fecha' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['Fecha'] >= start_date) & (filtered_df['Fecha'] <= end_date)]

        # KPIs en la parte superior
        st.subheader("Resumen General")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Actividades", len(filtered_df))

        with col2:
            st.metric("Municipios Atendidos", filtered_df['Municipio'].nunique())

        with col3:
            if 'Observaciones' in filtered_df.columns:
                st.metric("Tipos de Equipos Utilizados", filtered_df['Observaciones'].nunique())
            else:
                st.metric("Tipos de Equipos Utilizados", "N/A")

        with col4:
            # Calcular el último mes con actividades
            if not filtered_df.empty and 'Fecha' in filtered_df.columns:
                last_date = filtered_df['Fecha'].max()
                st.metric("Última Actividad", last_date.strftime('%d-%m-%Y'))
            else:
                st.metric("Última Actividad", "N/A")

        # Primera fila de gráficos
        st.subheader("Análisis por Municipio y Tipo de Equipo")
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de barras: Actividades por Municipio
            if not filtered_df.empty:
                municipio_counts = filtered_df['Municipio'].value_counts().reset_index()
                municipio_counts.columns = ['Municipio', 'Cantidad']

                fig_bar = px.bar(
                    municipio_counts,
                    x='Municipio',
                    y='Cantidad',
                    color='Municipio',
                    title="Actividades por Municipio"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No hay datos disponibles para generar el gráfico de municipios.")

        with col2:
            # Gráfico circular: Distribución por Tipo de Equipo
            if not filtered_df.empty and 'Observaciones' in filtered_df.columns:
                equipo_counts = filtered_df['Observaciones'].value_counts().reset_index()
                equipo_counts.columns = ['Tipo de Equipo', 'Cantidad']

                fig_pie = px.pie(
                    equipo_counts,
                    names='Tipo de Equipo',
                    values='Cantidad',
                    title="Distribución por Tipo de Equipo",
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("No hay datos disponibles para generar el gráfico de tipos de equipo.")

        # Segunda fila de gráficos
        st.subheader("Análisis Temporal y Relaciones")
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de línea: Actividades a lo largo del tiempo
            if not filtered_df.empty and 'Fecha' in filtered_df.columns:
                # Preparar datos para la serie temporal
                filtered_df['Mes-Año'] = filtered_df['Fecha'].dt.strftime('%Y-%m')
                time_series_data = filtered_df.groupby('Mes-Año').size().reset_index()
                time_series_data.columns = ['Mes-Año', 'Cantidad']

                # Ordenar cronológicamente
                time_series_data['Fecha-Orden'] = pd.to_datetime(time_series_data['Mes-Año'] + '-01')
                time_series_data = time_series_data.sort_values('Fecha-Orden')
                time_series_data['Mes-Año-Format'] = time_series_data['Fecha-Orden'].dt.strftime('%b %Y')

                fig_line = px.line(
                    time_series_data,
                    x='Mes-Año-Format',
                    y='Cantidad',
                    markers=True,
                    title="Actividades por Mes"
                )
                fig_line.update_layout(xaxis_title="Mes-Año", yaxis_title="Cantidad de Actividades")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para generar el gráfico temporal.")

        with col2:
            # Mapa de calor: Municipio vs Tipo de Equipo
            if (not filtered_df.empty and 'Observaciones' in filtered_df.columns and 
                filtered_df['Municipio'].nunique() > 1 and filtered_df['Observaciones'].nunique() > 1):
                # Crear tabla cruzada para el mapa de calor
                heatmap_data = pd.crosstab(
                    index=filtered_df['Municipio'],
                    columns=filtered_df['Observaciones'],
                    normalize=False  # Usar conteos absolutos
                )

                # Crear figura con Plotly Express
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Tipo de Equipo", y="Municipio", color="Cantidad"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    color_continuous_scale="YlOrRd",
                    title="Relación entre Municipio y Tipo de Equipo"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("No hay datos suficientes para generar el mapa de calor. Se necesitan múltiples municipios y tipos de equipo.")

        # Mostrar datos en tabla
        st.subheader("Datos Detallados")
        # Formatear fechas para mejor visualización
        if not filtered_df.empty:
            display_df = filtered_df.copy()
            if 'Fecha' in display_df.columns:
                display_df['Fecha'] = display_df['Fecha'].dt.strftime('%Y-%m-%d')

            # Opciones de ordenación
            columns_to_sort = [col for col in ['Fecha', 'Municipio', 'Observaciones'] if col in display_df.columns]
            sort_column = st.selectbox("Ordenar por:", columns_to_sort)
            sort_order = st.radio("Orden:", ["Ascendente", "Descendente"], horizontal=True)

            # Aplicar ordenamiento
            if sort_order == "Ascendente":
                display_df = display_df.sort_values(by=sort_column, ascending=True)
            else:
                display_df = display_df.sort_values(by=sort_column, ascending=False)

            # Mostrar tabla con datos
            columns_to_display = [col for col in ['Fecha', 'Municipio', 'Barrios', 'Observaciones'] if col in display_df.columns]
            st.dataframe(
                display_df[columns_to_display],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("No hay datos disponibles para mostrar en la tabla.")

        # Análisis adicional - Distribución geográfica
        st.subheader("Análisis Adicional")
        col1, col2 = st.columns(2)

        with col1:
            # Barrios más atendidos
            if not filtered_df.empty and 'Barrios' in filtered_df.columns:
                st.write("Barrios más frecuentemente atendidos:")

                # Crear un análisis por barrio
                # Nota: Esta es una visualización hipotética ya que la columna 'Barrios' 
                # contiene texto con múltiples barrios en cada registro

                # Mostrar la distribución de barrios como texto
                barrios_text = filtered_df['Barrios'].fillna('No especificado').tolist()
                barrios_display = ", ".join(barrios_text)

                st.text_area("Barrios mencionados:", barrios_display, height=200)
            else:
                st.warning("No hay datos disponibles para analizar barrios.")

        with col2:
            # Observaciones específicas
            if not filtered_df.empty and 'Observaciones' in filtered_df.columns:
                st.write("Detalles de las Observaciones:")

                # Mostrar una tabla con las observaciones específicas
                obs_df = filtered_df[['Municipio', 'Observaciones']].value_counts().reset_index()
                obs_df.columns = ['Municipio', 'Tipo de Equipo', 'Cantidad']

                st.dataframe(obs_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No hay datos disponibles para analizar observaciones.")

        # Notas y conclusiones
        st.subheader("Conclusiones")
        st.markdown("""
            - Este dashboard permite visualizar y analizar las actividades de control vectorial realizadas en diferentes municipios.
            - Se pueden identificar tendencias temporales en las actividades realizadas.
            - Es posible analizar qué tipos de equipos se utilizan más frecuentemente en cada municipio.
            - Los filtros permiten hacer un análisis detallado por municipio, tipo de equipo y período de tiempo.
        """)

        # Información sobre la actualización de datos
        st.sidebar.markdown("---")
        if 'Fecha' in df.columns:
            max_date = df['Fecha'].max().date() if not df.empty else datetime.now().date()
            st.sidebar.info(f"Datos actualizados hasta: {max_date.strftime('%d-%m-%Y')}")
        st.sidebar.markdown("Dashboard creado con Streamlit")

        # Instrucciones de uso
        with st.sidebar.expander("Instrucciones de Uso"):
            st.markdown("""
            1. Use los filtros de la barra lateral para seleccionar municipios, tipos de equipo y rango de fechas
            2. Los gráficos se actualizarán automáticamente según los filtros seleccionados
            3. La tabla de datos muestra información detallada que puede ordenarse según diferentes criterios
            """)

        # Opción para descargar los datos
        if not filtered_df.empty:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Exportar Datos")

            # Convertir el dataframe a csv para descarga
            csv = filtered_df.to_csv(index=False).encode('utf-8')

            st.sidebar.download_button(
                label="Descargar datos como CSV",
                data=csv,
                file_name="datos_control_vectorial.csv",
                mime="text/csv",
            )

    # =====================================================================
    # Dashboard 4: ANÁLISIS TEMPORAL (HOJA4)
    # =====================================================================
    elif dashboard_key == "hoja4":
        st.header("⏱️ Análisis Temporal de Control Vectorial")
        st.markdown("""
        Esta aplicación permite analizar la distribución temporal de las intervenciones de control vectorial.
        """)

        # Función para cargar los datos de esta hoja específica
        @st.cache_data
        def cargar_datos_hoja4():
            # Cargar los datos del Excel
            df = cargar_excel("Hoja4")

            if not df.empty:
                # Asegurar que la columna Fecha sea de tipo datetime
                if 'Fecha' in df.columns:
                    df['Fecha'] = pd.to_datetime(df['Fecha'])

                    # Extraer componentes de fecha para análisis temporal
                    df['Año'] = df['Fecha'].dt.year
                    df['Mes'] = df['Fecha'].dt.month
                    df['Dia'] = df['Fecha'].dt.day
                    df['DiaSemana'] = df['Fecha'].dt.dayofweek
                    df['NombreMes'] = df['Fecha'].dt.month_name()
                    df['NombreDiaSemana'] = df['Fecha'].dt.day_name()

                # Procesamiento de los entornos (separados por punto y coma)
                if 'Entrono' in df.columns:
                    df['Entornos_Lista'] = df['Entrono'].apply(
                        lambda x: [entorno.strip() for entorno in str(x).split(';')] if isinstance(x, str) else [x]
                    )

            return df

        # Función para generar un análisis de tendencia
        def analisis_tendencia(df):
            # Agrupar datos por fecha
            df_tendencia = df.groupby('Fecha').size().reset_index()
            df_tendencia.columns = ['Fecha', 'Intervenciones']

            # Calcular promedio móvil para suavizar la tendencia
            if len(df_tendencia) > 2:
                df_tendencia['Promedio_Movil'] = df_tendencia['Intervenciones'].rolling(window=min(3, len(df_tendencia)), center=True).mean()
            else:
                df_tendencia['Promedio_Movil'] = df_tendencia['Intervenciones']

            return df_tendencia

        # Función para analizar la distribución por día de la semana
        def analisis_dia_semana(df):
            # Orden de los días de la semana comenzando por lunes (0) hasta domingo (6)
            orden_dias = list(range(7))
            nombres_dias = [calendar.day_name[i] for i in orden_dias]

            # Contar intervenciones por día de la semana
            conteo_dias = df['DiaSemana'].value_counts().reindex(orden_dias, fill_value=0).reset_index()
            conteo_dias.columns = ['DiaSemana', 'Intervenciones']
            conteo_dias['NombreDia'] = conteo_dias['DiaSemana'].apply(lambda x: calendar.day_name[x])

            return conteo_dias

        # Función para analizar la distribución por mes
        def analisis_mensual(df):
            # Orden de los meses desde enero (1) hasta diciembre (12)
            orden_meses = list(range(1, 13))
            nombres_meses = [calendar.month_name[i] for i in orden_meses]

            # Contar intervenciones por mes
            conteo_meses = df['Mes'].value_counts().reindex(orden_meses, fill_value=0).reset_index()
            conteo_meses.columns = ['Mes', 'Intervenciones']
            conteo_meses['NombreMes'] = conteo_meses['Mes'].apply(lambda x: calendar.month_name[x])

            return conteo_meses

        # Cargar los datos
        df = cargar_datos_hoja4()

        if not df.empty:
            # Crear contenedor con pestañas para diferentes análisis
            tab1, tab2, tab3 = st.tabs(["Análisis por Fecha", "Análisis por Municipio", "Patrones Temporales"])

            # Pestaña 1: Análisis por Fecha
            with tab1:
                st.header("Análisis por Fecha")

                # Filtros de fecha
                col1, col2 = st.columns(2)

                with col1:
                    # Obtener fecha mínima y máxima
                    fecha_min = df['Fecha'].min().date()
                    fecha_max = df['Fecha'].max().date()

                    # Crear selector de rango de fechas
                    rango_fechas = st.date_input(
                        "Seleccionar Rango de Fechas",
                        [fecha_min, fecha_max],
                        min_value=fecha_min,
                        max_value=fecha_max
                    )

                    # Asegurar que tenemos dos fechas seleccionadas
                    if len(rango_fechas) == 2:
                        fecha_inicio, fecha_fin = rango_fechas
                    else:
                        fecha_inicio, fecha_fin = fecha_min, fecha_max

                with col2:
                    # Filtro de municipios
                    municipios = sorted(df['Municipio'].unique())
                    municipio_seleccionado = st.multiselect(
                        "Seleccionar Municipio(s)",
                        options=municipios,
                        default=municipios
                    )

                # Aplicar filtros
                df_filtrado = df.copy()

                # Filtro de fechas
                df_filtrado = df_filtrado[(df_filtrado['Fecha'].dt.date >= fecha_inicio) & 
                                        (df_filtrado['Fecha'].dt.date <= fecha_fin)]

                # Filtro de municipios
                if municipio_seleccionado:
                    df_filtrado = df_filtrado[df_filtrado['Municipio'].isin(municipio_seleccionado)]

                # Mostrar estadísticas básicas
                st.subheader("Estadísticas del Período Seleccionado")

                if not df_filtrado.empty:
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                    with col_stat1:
                        total_intervenciones = len(df_filtrado)
                        st.metric("Total Intervenciones", total_intervenciones)

                    with col_stat2:
                        dias_periodo = (fecha_fin - fecha_inicio).days + 1
                        promedio_diario = round(total_intervenciones / dias_periodo, 2) if dias_periodo > 0 else 0
                        st.metric("Promedio Diario", promedio_diario)

                    with col_stat3:
                        dias_con_actividad = df_filtrado['Fecha'].dt.date.nunique()
                        porcentaje_cobertura = round((dias_con_actividad / dias_periodo) * 100, 1) if dias_periodo > 0 else 0
                        st.metric("Días con Actividad", f"{dias_con_actividad} ({porcentaje_cobertura}%)")

                    with col_stat4:
                        municipios_atendidos = df_filtrado['Municipio'].nunique()
                        st.metric("Municipios Atendidos", municipios_atendidos)

                    # Gráfico de tendencia temporal
                    st.subheader("Tendencia de Intervenciones")

                    # Generar datos de tendencia
                    df_tendencia = analisis_tendencia(df_filtrado)

                    # Usar Plotly para el gráfico de tendencia
                    fig_tendencia = go.Figure()

                    # Añadir línea de intervenciones diarias
                    fig_tendencia.add_trace(go.Scatter(
                        x=df_tendencia['Fecha'],
                        y=df_tendencia['Intervenciones'],
                        mode='lines+markers',
                        name='Intervenciones Diarias',
                        line=dict(color='blue')
                    ))

                    # Añadir línea de promedio móvil
                    if len(df_tendencia) > 2:
                        fig_tendencia.add_trace(go.Scatter(
                            x=df_tendencia['Fecha'],
                            y=df_tendencia['Promedio_Movil'],
                            mode='lines',
                            name='Tendencia (Promedio Móvil)',
                            line=dict(color='red', dash='dash')
                        ))

                    # Personalizar el gráfico
                    fig_tendencia.update_layout(
                        title='Tendencia de Intervenciones de Control Vectorial',
                        xaxis_title='Fecha',
                        yaxis_title='Número de Intervenciones',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=400
                    )

                    st.plotly_chart(fig_tendencia, use_container_width=True)

                    # Tabla de intervenciones por fecha
                    st.subheader("Detalle de Intervenciones por Fecha")

                    # Agrupar datos por fecha
                    df_por_fecha = df_filtrado.groupby(['Fecha']).size().reset_index()
                    df_por_fecha.columns = ['Fecha', 'Intervenciones']
                    df_por_fecha['Fecha'] = df_por_fecha['Fecha'].dt.date
                    df_por_fecha = df_por_fecha.sort_values('Fecha', ascending=False)

                    st.dataframe(df_por_fecha, use_container_width=True)
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Pestaña 2: Análisis por Municipio
            with tab2:
                st.header("Análisis por Municipio")

                # Filtros
                col1, col2 = st.columns(2)

                with col1:
                    # Obtener fecha mínima y máxima
                    fecha_min = df['Fecha'].min().date()
                    fecha_max = df['Fecha'].max().date()

                    # Crear selector de rango de fechas
                    rango_fechas = st.date_input(
                        "Seleccionar Rango de Fechas",
                        [fecha_min, fecha_max],
                        min_value=fecha_min,
                        max_value=fecha_max,
                        key="fechas_tab2"
                    )

                    # Asegurar que tenemos dos fechas seleccionadas
                    if len(rango_fechas) == 2:
                        fecha_inicio, fecha_fin = rango_fechas
                    else:
                        fecha_inicio, fecha_fin = fecha_min, fecha_max

                with col2:
                    # Agrupar por semana o mes
                    opciones_agrupacion = ["Por Día", "Por Semana", "Por Mes"]
                    agrupacion = st.selectbox("Agrupar Datos", opciones_agrupacion)

                # Aplicar filtros
                df_filtrado = df.copy()

                # Filtro de fechas
                df_filtrado = df_filtrado[(df_filtrado['Fecha'].dt.date >= fecha_inicio) & 
                                          (df_filtrado['Fecha'].dt.date <= fecha_fin)]

                if not df_filtrado.empty:
                    # Preparar datos según la agrupación seleccionada
                    if agrupacion == "Por Día":
                        # Usar los datos diarios directamente
                        df_tiempo = df_filtrado.copy()
                        df_tiempo['Periodo'] = df_tiempo['Fecha']
                    elif agrupacion == "Por Semana":
                        # Agregar columna de semana
                        df_filtrado['Semana'] = df_filtrado['Fecha'].dt.isocalendar().week
                        df_filtrado['Año_Semana'] = df_filtrado['Fecha'].dt.strftime('%Y-%U')
                        df_tiempo = df_filtrado.copy()
                        df_tiempo['Periodo'] = df_tiempo['Año_Semana'].apply(lambda x: f"Semana {x.split('-')[1]} de {x.split('-')[0]}")
                    else:  # Por Mes
                        # Agregar columna de mes
                        df_filtrado['Año_Mes'] = df_filtrado['Fecha'].dt.strftime('%Y-%m')
                        df_tiempo = df_filtrado.copy()
                        df_tiempo['Periodo'] = df_tiempo['Año_Mes'].apply(lambda x: f"{calendar.month_name[int(x.split('-')[1])]} {x.split('-')[0]}")

                    # Gráfico de municipios a lo largo del tiempo
                    st.subheader(f"Intervenciones por Municipio {agrupacion}")

                    # Agrupar datos
                    if agrupacion == "Por Día":
                        df_agrupado = df_tiempo.groupby(['Periodo', 'Municipio']).size().reset_index()
                    else:
                        df_agrupado = df_tiempo.groupby(['Periodo', 'Municipio']).size().reset_index()

                    df_agrupado.columns = ['Periodo', 'Municipio', 'Intervenciones']

                    # Crear gráfico de líneas por municipio
                    municipios = sorted(df_agrupado['Municipio'].unique())

                    # Usar Plotly para el gráfico de líneas
                    fig_municipios = go.Figure()

                    for municipio in municipios:
                        datos_municipio = df_agrupado[df_agrupado['Municipio'] == municipio]
                        fig_municipios.add_trace(go.Scatter(
                            x=datos_municipio['Periodo'],
                            y=datos_municipio['Intervenciones'],
                            mode='lines+markers',
                            name=municipio
                        ))

                    # Personalizar el gráfico
                    fig_municipios.update_layout(
                        title=f'Intervenciones por Municipio {agrupacion}',
                        xaxis_title='Periodo',
                        yaxis_title='Número de Intervenciones',
                        legend_title='Municipio',
                        height=500
                    )

                    st.plotly_chart(fig_municipios, use_container_width=True)

                    # Tabla de resumen por municipio
                    st.subheader("Resumen de Intervenciones por Municipio")

                    # Agrupar datos por municipio
                    df_por_municipio = df_filtrado.groupby(['Municipio']).size().reset_index()
                    df_por_municipio.columns = ['Municipio', 'Intervenciones']
                    df_por_municipio = df_por_municipio.sort_values('Intervenciones', ascending=False)

                    # Calcular porcentaje
                    total_intervenciones = df_por_municipio['Intervenciones'].sum()
                    df_por_municipio['Porcentaje'] = round((df_por_municipio['Intervenciones'] / total_intervenciones) * 100, 1)

                    st.dataframe(df_por_municipio, use_container_width=True)

                    # Gráfico de barras para el resumen por municipio
                    fig_resumen = px.bar(
                        df_por_municipio,
                        x='Municipio',
                        y='Intervenciones',
                        color='Municipio',
                        title='Total de Intervenciones por Municipio',
                        labels={'Intervenciones': 'Número de Intervenciones', 'Municipio': 'Municipio'},
                        height=400
                    )

                    fig_resumen.update_layout(
                        xaxis_title='Municipio',
                        yaxis_title='Número de Intervenciones',
                        showlegend=False
                    )

                    st.plotly_chart(fig_resumen, use_container_width=True)
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Pestaña 3: Patrones Temporales
            with tab3:
                st.header("Patrones Temporales")

                # Filtros
                col1, col2 = st.columns(2)

                with col1:
                    # Filtro de municipios
                    municipios = sorted(df['Municipio'].unique())
                    municipio_seleccionado = st.multiselect(
                        "Seleccionar Municipio(s)",
                        options=municipios,
                        default=municipios,
                        key="municipios_tab3"
                    )

                with col2:
                    # Filtro de entornos
                    if 'Entornos_Lista' in df.columns:
                        todos_entornos = []
                        for lista_entornos in df['Entornos_Lista']:
                            todos_entornos.extend(lista_entornos)

                        entornos_unicos = sorted(set(todos_entornos))

                        entorno_seleccionado = st.multiselect(
                            "Seleccionar Entorno(s)",
                            options=entornos_unicos,
                            default=entornos_unicos
                        )
                    else:
                        entorno_seleccionado = []
                        st.warning("No se encontró la columna de entornos para filtrar.")

                # Aplicar filtros
                df_filtrado = df.copy()

                # Filtro de municipios
                if municipio_seleccionado:
                    df_filtrado = df_filtrado[df_filtrado['Municipio'].isin(municipio_seleccionado)]

                # Filtro de entornos
                if entorno_seleccionado and 'Entornos_Lista' in df_filtrado.columns:
                    df_filtrado = df_filtrado[df_filtrado['Entornos_Lista'].apply(
                        lambda lista: any(entorno in lista for entorno in entorno_seleccionado)
                    )]

                if not df_filtrado.empty:
                    # Crear dos columnas para los gráficos
                    col_grafico1, col_grafico2 = st.columns(2)

                    with col_grafico1:
                        st.subheader("Distribución por Día de la Semana")

                        # Análisis por día de la semana
                        conteo_dias = analisis_dia_semana(df_filtrado)

                        # Crear gráfico con Plotly
                        fig_dias = px.bar(
                            conteo_dias,
                            x='NombreDia',
                            y='Intervenciones',
                            color='NombreDia',
                            title='Intervenciones por Día de la Semana',
                            labels={'Intervenciones': 'Número de Intervenciones', 'NombreDia': 'Día de la Semana'},
                            height=400
                        )

                        fig_dias.update_layout(
                            xaxis_title='Día de la Semana',
                            yaxis_title='Número de Intervenciones',
                            showlegend=False
                        )

                        st.plotly_chart(fig_dias, use_container_width=True)

                    with col_grafico2:
                        st.subheader("Distribución por Mes")

                        # Análisis por mes
                        conteo_meses = analisis_mensual(df_filtrado)

                        # Crear gráfico con Plotly
                        fig_meses = px.bar(
                            conteo_meses,
                            x='NombreMes',
                            y='Intervenciones',
                            color='NombreMes',
                            title='Intervenciones por Mes',
                            labels={'Intervenciones': 'Número de Intervenciones', 'NombreMes': 'Mes'},
                            height=400
                        )

                        fig_meses.update_layout(
                            xaxis_title='Mes',
                            yaxis_title='Número de Intervenciones',
                            showlegend=False
                        )

                        st.plotly_chart(fig_meses, use_container_width=True)

                    # Mapa de calor de actividad
                    st.subheader("Mapa de Calor de Actividad")

                    # Preparar datos para el mapa de calor
                    # Agrupar por día de la semana y mes
                    df_filtrado['DiaSemana'] = df_filtrado['Fecha'].dt.dayofweek
                    df_filtrado['Mes'] = df_filtrado['Fecha'].dt.month

                    df_heatmap = df_filtrado.groupby(['DiaSemana', 'Mes']).size().reset_index()
                    df_heatmap.columns = ['DiaSemana', 'Mes', 'Intervenciones']

                    # Crear un pivot para el mapa de calor
                    pivot_heatmap = df_heatmap.pivot(index='DiaSemana', columns='Mes', values='Intervenciones')
                    pivot_heatmap.fillna(0, inplace=True)

                    # Renombrar índices y columnas
                    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

                    # Mapear índices y columnas a nombres de días y meses
                    dias_mapeados = [dias_semana[i] if i < len(dias_semana) else str(i) for i in pivot_heatmap.index]
                    meses_mapeados = [meses[i-1] if 0 < i <= len(meses) else str(i) for i in pivot_heatmap.columns]

                    # Crear mapa de calor con Plotly
                    fig_heatmap = px.imshow(
                        pivot_heatmap,
                        labels=dict(x='Mes', y='Día de la Semana', color='Intervenciones'),
                        x=meses_mapeados,
                        y=dias_mapeados,
                        color_continuous_scale='YlGnBu',
                        title='Mapa de Calor: Intervenciones por Día de la Semana y Mes'
                    )

                    fig_heatmap.update_layout(
                        height=500,
                        xaxis_title='Mes',
                        yaxis_title='Día de la Semana',
                        coloraxis_colorbar=dict(title='Intervenciones')
                    )

                    # Añadir anotaciones con los valores
                    anotaciones = []
                    for i, dia in enumerate(pivot_heatmap.index):
                        for j, mes in enumerate(pivot_heatmap.columns):
                            valor = pivot_heatmap.iloc[i, j]
                            if valor > 0:  # Solo mostrar valores mayores que cero
                                anotaciones.append(dict(
                                    x=meses_mapeados[j],
                                    y=dias_mapeados[i],
                                    text=str(int(valor)),
                                    showarrow=False,
                                    font=dict(color='white' if valor > 3 else 'black')
                                ))

                    fig_heatmap.update_layout(annotations=anotaciones)

                    st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Análisis de Entornos a lo largo del tiempo
                    if 'Entornos_Lista' in df_filtrado.columns:
                        st.subheader("Entornos Intervenidos a lo Largo del Tiempo")

                        # Procesar datos para este análisis
                        registros_entornos = []

                        for idx, row in df_filtrado.iterrows():
                            fecha = row['Fecha']
                            for entorno in row['Entornos_Lista']:
                                registros_entornos.append({
                                    'Fecha': fecha,
                                    'Entorno': entorno
                                })

                        df_entornos_tiempo = pd.DataFrame(registros_entornos)

                        # Agrupar por fecha y entorno
                        df_entornos_agrupados = df_entornos_tiempo.groupby(['Fecha', 'Entorno']).size().reset_index()
                        df_entornos_agrupados.columns = ['Fecha', 'Entorno', 'Intervenciones']

                        # Usar Plotly para el gráfico de líneas
                        fig_entornos = go.Figure()

                        entornos_unicos = df_entornos_agrupados['Entorno'].unique()

                        for entorno in entornos_unicos:
                            datos_entorno = df_entornos_agrupados[df_entornos_agrupados['Entorno'] == entorno]
                            fig_entornos.add_trace(go.Scatter(
                                x=datos_entorno['Fecha'],
                                y=datos_entorno['Intervenciones'],
                                mode='lines+markers',
                                name=entorno
                            ))

                        # Personalizar el gráfico
                        fig_entornos.update_layout(
                            title='Intervenciones por Entorno a lo Largo del Tiempo',
                            xaxis_title='Fecha',
                            yaxis_title='Número de Intervenciones',
                            legend_title='Entorno',
                            height=500
                        )

                        st.plotly_chart(fig_entornos, use_container_width=True)

                    # Conclusiones y patrones identificados
                    st.subheader("Conclusiones sobre Patrones Temporales")

                    st.markdown("""
                    A partir del análisis de los patrones temporales, se pueden extraer las siguientes conclusiones:

                    1. **Distribución semanal**: Los datos muestran cómo se distribuyen las actividades de control vectorial a lo largo de la semana, lo que puede indicar patrones en la planificación de las intervenciones.

                    2. **Estacionalidad mensual**: El análisis por mes permite identificar si hay temporadas de mayor actividad, posiblemente relacionadas con factores climáticos o epidemiológicos.

                    3. **Evolución de las prioridades**: El gráfico de entornos a lo largo del tiempo muestra cómo ha evolucionado el enfoque de las intervenciones, priorizando diferentes tipos de entornos según el período.

                    **Recomendaciones para optimización temporal:**

                    1. Planificar intervenciones considerando los patrones identificados para maximizar la eficiencia.

                    2. Aumentar la frecuencia de intervenciones en los períodos que históricamente han mostrado mayor necesidad.

                    3. Balancear las actividades a lo largo de la semana para mantener una cobertura constante.

                    4. Anticipar necesidades adicionales en meses que históricamente han requerido mayor atención.
                    """)
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Añadir una sección de recomendaciones generales
            st.header("Recomendaciones para Mejora Continua")

            st.markdown("""
            Basado en el análisis temporal de las intervenciones de control vectorial, se proponen las siguientes recomendaciones para mejorar el sistema de seguimiento y la efectividad de las intervenciones:

            1. **Sistematización de datos**: 
               - Implementar un sistema más detallado de registro que incluya resultados específicos de las intervenciones (ej. índices entomológicos pre y post intervención).
               - Recopilar datos georreferenciados para cada intervención, permitiendo análisis espaciales más precisos.

            2. **Planificación estratégica**:
               - Utilizar los patrones temporales identificados para optimizar la programación de intervenciones futuras.
               - Establecer un calendario proactivo basado en las tendencias históricas y factores climáticos.

            3. **Evaluación de impacto**:
               - Cruzar los datos de intervenciones con información epidemiológica para evaluar la efectividad de las acciones.
               - Implementar indicadores clave de desempeño (KPIs) para medir el éxito de las intervenciones.

            4. **Mejora de la cobertura**:
               - Identificar y abordar las brechas en la cobertura temporal (días o semanas sin intervenciones).
               - Asegurar una distribución equilibrada de recursos entre municipios según sus necesidades específicas.

            5. **Participación comunitaria**:
               - Integrar actividades de educación y participación comunitaria en el cronograma de intervenciones.
               - Implementar sistemas de alertas tempranas con participación de la comunidad.
            """)
        else:
            st.error("No se pudieron cargar los datos temporales. Por favor verifica que el archivo Excel esté disponible y contenga la Hoja4.")

    # =====================================================================
    # Dashboard 5: ANÁLISIS INTEGRADO (HOJA1 + HOJA4)
    # =====================================================================
    elif dashboard_key == "hoja41":
        st.header("🦟 Sistema de Análisis Interactivo de Control Vectorial")
        st.markdown("""
        Esta aplicación permite analizar datos sobre lugares de visita y actividades de control vectorial 
        en diferentes municipios del departamento de Tolima. **Haz clic o pasa el cursor sobre los elementos 
        de los gráficos para ver más detalles.**
        """)

        # Funciones para cargar los datos
        @st.cache_data
        def cargar_datos_lugares_visita():
            """Carga los datos de lugares de visita desde la Hoja1"""
            try:
                df = cargar_excel("Hoja1")
                return df
            except Exception as e:
                st.error(f"Error al cargar datos de lugares de visita: {e}")
                return pd.DataFrame()

        @st.cache_data
        def cargar_datos_control_vectorial():
            """Carga los datos de control vectorial desde la Hoja4"""
            try:
                df = cargar_excel("Hoja4")

                # Asegurar que la columna Fecha sea de tipo datetime
                if 'Fecha' in df.columns:
                    df['Fecha'] = pd.to_datetime(df['Fecha'])

                # Buscar la columna de entorno (puede tener nombres ligeramente diferentes)
                entorno_col = None
                for col in df.columns:
                    if col.lower() in ['entorno', 'entrono', 'ambiente']:
                        entorno_col = col
                        break

                if entorno_col:
                    # Procesar la columna de entornos
                    df['Entornos_Lista'] = df[entorno_col].apply(
                        lambda x: [entorno.strip() for entorno in str(x).split(';')] if isinstance(x, str) else [str(x)]
                    )

                return df
            except Exception as e:
                st.error(f"Error detallado al cargar datos de control vectorial: {str(e)}")
                return pd.DataFrame()

        # Cargar los datos con manejo de errores
        with st.spinner("Cargando datos..."):
            # Datos de lugares de visita (Hoja1)
            df_lugares = cargar_datos_lugares_visita()

            # Datos de control vectorial (Hoja4)
            df_control = cargar_datos_control_vectorial()

            # Verificar si se cargaron los datos correctamente
            if df_lugares.empty:
                st.error("No se pudieron cargar los datos de lugares de visita.")

            if df_control.empty:
                st.error("No se pudieron cargar los datos de control vectorial.")

        # Crear pestañas para las diferentes secciones de análisis
        tab1, tab2, tab3 = st.tabs([
            "📍 Lugares de Visita", 
            "🧪 Control Vectorial", 
            "📊 Análisis Integrado"
        ])

        # --------------------------
        # Pestaña 1: Lugares de Visita
        # --------------------------
        with tab1:
            st.header("Análisis de Lugares de Visita")

            if not df_lugares.empty:
                # Filtro de municipio
                municipios_lugares = sorted(df_lugares['Municipio'].unique())
                municipio_seleccionado = st.selectbox(
                    "Seleccione un municipio:",
                    options=["Todos los municipios"] + municipios_lugares
                )

                # Aplicar filtro de municipio
                if municipio_seleccionado == "Todos los municipios":
                    df_lugares_filtrado = df_lugares.copy()
                else:
                    df_lugares_filtrado = df_lugares[df_lugares['Municipio'] == municipio_seleccionado]

                # Mostrar datos filtrados
                st.subheader("Datos de Lugares de Visita")
                st.dataframe(df_lugares_filtrado, use_container_width=True)

                # Comprobar si hay columnas esperadas
                tipos_lugar = []
                for col in ['IPS (entorno)', 'Centro salud', 'Plazas de mercado', 'Terminal o terminalito']:
                    if col in df_lugares.columns:
                        tipos_lugar.append(col)

                if tipos_lugar:
                    # Crear columnas para los gráficos
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Distribución de Tipos de Lugar")

                        # Preparar datos para la visualización
                        if municipio_seleccionado == "Todos los municipios":
                            # Sumar por tipo de lugar para todos los municipios
                            conteo_tipos = df_lugares[tipos_lugar].sum()
                        else:
                            # Obtener los datos del municipio seleccionado
                            conteo_tipos = df_lugares_filtrado[tipos_lugar].iloc[0]

                        # Crear DataFrame para el gráfico
                        df_tipos = pd.DataFrame({
                            'Tipo de Lugar': tipos_lugar,
                            'Cantidad': conteo_tipos.values
                        })

                        # Crear gráfico de barras interactivo con Plotly
                        fig = px.bar(
                            df_tipos, 
                            x='Tipo de Lugar', 
                            y='Cantidad',
                            color='Tipo de Lugar',
                            title=f'Cantidad de Lugares por Tipo {"en " + municipio_seleccionado if municipio_seleccionado != "Todos los municipios" else ""}',
                            labels={'Cantidad': 'Número de lugares', 'Tipo de Lugar': 'Tipo'},
                            height=400
                        )

                        # Personalizar el gráfico
                        fig.update_layout(
                            xaxis_title='Tipo de Lugar',
                            yaxis_title='Cantidad',
                            xaxis={'categoryorder':'total descending'},
                            hovermode='closest',
                            showlegend=False
                        )

                        # Mostrar el gráfico
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("Instituciones Educativas Rurales")

                        if 'I.E. (zona rural)' in df_lugares_filtrado.columns:
                            # Obtener datos de IE rurales
                            if municipio_seleccionado != "Todos los municipios":
                                ie_rurales = df_lugares_filtrado['I.E. (zona rural)'].iloc[0]

                                if isinstance(ie_rurales, str):
                                    # Dividir por punto y coma si hay múltiples IE
                                    lista_ie = [ie.strip() for ie in ie_rurales.split(';')]

                                    # Mostrar cada IE con sus actividades
                                    for ie in lista_ie:
                                        st.markdown(f"**{ie}**")
                                else:
                                    st.write("No hay información disponible sobre IE rurales.")
                            else:
                                st.write("Seleccione un municipio específico para ver detalles de las IE rurales.")
                        else:
                            st.write("No hay datos disponibles sobre IE rurales.")

                        # Sedes o establecimientos
                        st.subheader("Sedes o Establecimientos")

                        if 'Sedes' in df_lugares_filtrado.columns:
                            if municipio_seleccionado != "Todos los municipios":
                                sedes = df_lugares_filtrado['Sedes'].iloc[0]

                                if isinstance(sedes, str):
                                    # Dividir por punto y coma si hay múltiples sedes
                                    lista_sedes = [sede.strip() for sede in sedes.split(',')]

                                    # Mostrar cada sede
                                    for sede in lista_sedes:
                                        st.markdown(f"• {sede}")
                                else:
                                    st.write("No hay información disponible sobre sedes.")
                            else:
                                st.write("Seleccione un municipio específico para ver detalles de las sedes.")
                        else:
                            st.write("No hay datos disponibles sobre sedes.")

                    # Análisis comparativo entre municipios
                    st.subheader("Comparación entre Municipios")

                    # Preparar datos para la comparación
                    df_comparacion = df_lugares[['Municipio'] + tipos_lugar].copy()

                    # Crear un DataFrame en formato "largo" para Plotly
                    datos_comparativos = []

                    for idx, row in df_comparacion.iterrows():
                        municipio = row['Municipio']
                        for tipo in tipos_lugar:
                            datos_comparativos.append({
                                'Municipio': municipio,
                                'Tipo de Lugar': tipo,
                                'Cantidad': row[tipo]
                            })

                    df_comp_largo = pd.DataFrame(datos_comparativos)

                    # Crear gráfico interactivo con Plotly
                    fig = px.bar(
                        df_comp_largo,
                        x='Municipio',
                        y='Cantidad',
                        color='Tipo de Lugar',
                        barmode='group',
                        title='Comparación de Lugares por Municipio',
                        labels={'Cantidad': 'Número de lugares', 'Municipio': 'Municipio'},
                        height=500
                    )

                    # Personalizar el gráfico
                    fig.update_layout(
                        xaxis_title='Municipio',
                        yaxis_title='Cantidad',
                        legend_title='Tipo de Lugar',
                        hovermode='closest'
                    )

                    # Mostrar gráfico
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No se encontraron columnas esperadas para los tipos de lugar en los datos.")
            else:
                st.warning("No se pudieron cargar los datos de lugares de visita.")

        # --------------------------
        # Pestaña 2: Control Vectorial
        # --------------------------
        with tab2:
            st.header("Análisis de Control Vectorial")

            if not df_control.empty:
                # Crear filtros
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Filtro de municipios
                    municipios_control = sorted(df_control['Municipio'].unique())
                    municipio_seleccionado_control = st.multiselect(
                        "Seleccionar Municipio(s):",
                        options=municipios_control,
                        default=municipios_control[0] if municipios_control else None
                    )

                with col2:
                    # Filtro de fechas
                    fecha_min = df_control['Fecha'].min().date()
                    fecha_max = df_control['Fecha'].max().date()

                    rango_fechas = st.date_input(
                        "Seleccionar Rango de Fechas:",
                        [fecha_min, fecha_max],
                        min_value=fecha_min,
                        max_value=fecha_max
                    )

                    if len(rango_fechas) == 2:
                        fecha_inicio, fecha_fin = rango_fechas
                    else:
                        fecha_inicio, fecha_fin = fecha_min, fecha_max

                with col3:
                    # Filtro de entornos
                    if 'Entornos_Lista' in df_control.columns:
                        todos_entornos = []
                        for lista_entornos in df_control['Entornos_Lista']:
                            todos_entornos.extend(lista_entornos)

                        entornos_unicos = sorted(set(todos_entornos))

                        entorno_seleccionado = st.multiselect(
                            "Seleccionar Entorno(s):",
                            options=entornos_unicos,
                            default=entornos_unicos[0] if entornos_unicos else None
                        )
                    else:
                        entorno_seleccionado = []
                        st.warning("No se encontró la columna de entornos para filtrar.")

                # Aplicar filtros
                df_control_filtrado = df_control.copy()

                # Filtro de municipio
                if municipio_seleccionado_control:
                    df_control_filtrado = df_control_filtrado[df_control_filtrado['Municipio'].isin(municipio_seleccionado_control)]

                # Filtro de fechas
                df_control_filtrado = df_control_filtrado[
                    (df_control_filtrado['Fecha'].dt.date >= fecha_inicio) & 
                    (df_control_filtrado['Fecha'].dt.date <= fecha_fin)
                ]

                # Filtro de entornos
                if entorno_seleccionado and 'Entornos_Lista' in df_control_filtrado.columns:
                    df_control_filtrado = df_control_filtrado[df_control_filtrado['Entornos_Lista'].apply(
                        lambda lista: any(entorno in lista for entorno in entorno_seleccionado)
                    )]

                # Mostrar datos filtrados
                st.subheader("Datos de Control Vectorial Filtrados")
                st.dataframe(df_control_filtrado, use_container_width=True)

                # Análisis y visualizaciones
                if not df_control_filtrado.empty:
                    # Crear columnas para los gráficos
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Intervenciones por Municipio")

                        # Contar intervenciones por municipio
                        conteo_municipios = df_control_filtrado['Municipio'].value_counts().reset_index()
                        conteo_municipios.columns = ['Municipio', 'Intervenciones']

                        # Crear gráfico interactivo con Plotly
                        fig = px.bar(
                            conteo_municipios,
                            x='Municipio',
                            y='Intervenciones',
                            color='Municipio',
                            title='Número de Intervenciones por Municipio',
                            labels={'Intervenciones': 'Número de intervenciones', 'Municipio': 'Municipio'},
                            height=400
                        )

                        # Personalizar gráfico
                        fig.update_layout(
                            xaxis_title='Municipio',
                            yaxis_title='Número de Intervenciones',
                            showlegend=False,
                            hovermode='closest'
                        )

                        # Mostrar gráfico
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("Intervenciones por Entorno")

                        if 'Entornos_Lista' in df_control_filtrado.columns:
                            # Procesar datos de entornos
                            entornos_planos = []
                            for lista_entornos in df_control_filtrado['Entornos_Lista']:
                                entornos_planos.extend(lista_entornos)

                            # Contar entornos
                            conteo_entornos = pd.Series(entornos_planos).value_counts().reset_index()
                            conteo_entornos.columns = ['Entorno', 'Intervenciones']

                            # Crear gráfico de pie interactivo con Plotly
                            fig = px.pie(
                                conteo_entornos,
                                values='Intervenciones',
                                names='Entorno',
                                title='Distribución de Intervenciones por Entorno',
                                height=400
                            )

                            # Personalizar gráfico
                            fig.update_traces(
                                textposition='inside',
                                textinfo='percent+label',
                                hoverinfo='label+percent+value'
                            )

                            fig.update_layout(
                                uniformtext_minsize=12,
                                uniformtext_mode='hide'
                            )

                            # Mostrar gráfico
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No se encontró la columna de entornos para el análisis.")

                    # Gráfico temporal
                    st.subheader("Evolución Temporal de las Intervenciones")

                    # Agrupar por fecha
                    df_temporal = df_control_filtrado.copy()
                    df_temporal['Fecha_Solo'] = df_temporal['Fecha'].dt.date

                    intervenciones_por_dia = df_temporal.groupby('Fecha_Solo').size().reset_index()
                    intervenciones_por_dia.columns = ['Fecha', 'Intervenciones']

                    # Crear gráfico de línea interactivo con Plotly
                    fig = px.line(
                        intervenciones_por_dia,
                        x='Fecha',
                        y='Intervenciones',
                        markers=True,
                        title='Evolución de Intervenciones a lo Largo del Tiempo',
                        labels={'Intervenciones': 'Número de intervenciones', 'Fecha': 'Fecha'},
                        height=400
                    )

                    # Personalizar gráfico
                    fig.update_traces(
                        hovertemplate='<b>Fecha</b>: %{x|%Y-%m-%d}<br><b>Intervenciones</b>: %{y}<extra></extra>'
                    )

                    fig.update_layout(
                        xaxis_title='Fecha',
                        yaxis_title='Número de Intervenciones',
                        hovermode='x unified'
                    )

                    # Mostrar gráfico
                    st.plotly_chart(fig, use_container_width=True)

                    # Análisis por tipo de intervención si existe la columna
                    if 'Intervencion' in df_control_filtrado.columns:
                        st.subheader("Tipos de Intervención Realizados")

                        # Contar tipos de intervención
                        conteo_intervenciones = df_control_filtrado['Intervencion'].value_counts().reset_index()
                        conteo_intervenciones.columns = ['Tipo de Intervención', 'Cantidad']

                        # Crear gráfico de barras horizontales interactivo con Plotly
                        fig = px.bar(
                            conteo_intervenciones,
                            y='Tipo de Intervención',
                            x='Cantidad',
                            color='Tipo de Intervención',
                            orientation='h',
                            title='Tipos de Intervención Realizados',
                            labels={'Cantidad': 'Número de intervenciones', 'Tipo de Intervención': 'Tipo'},
                            height=400
                        )

                        # Personalizar gráfico
                        fig.update_layout(
                            xaxis_title='Cantidad',
                            yaxis_title='Tipo de Intervención',
                            showlegend=False,
                            hovermode='closest'
                        )

                        # Mostrar gráfico
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")
            else:
                st.warning("No se pudieron cargar los datos de control vectorial.")

        # --------------------------
        # Pestaña 3: Análisis Integrado
        # --------------------------
        with tab3:
            st.header("Análisis Integrado de Datos")

            if not df_lugares.empty and not df_control.empty:
                # Verificar tipos de lugar
                tipos_lugar = []
                for col in ['IPS (entorno)', 'Centro salud', 'Plazas de mercado', 'Terminal o terminalito']:
                    if col in df_lugares.columns:
                        tipos_lugar.append(col)

                if not tipos_lugar:
                    st.warning("No se encontraron columnas esperadas para los tipos de lugar.")
                else:
                    # Filtro de municipio para el análisis integrado
                    municipios_comunes = sorted(set(df_lugares['Municipio']).intersection(set(df_control['Municipio'])))

                    municipio_integrado = st.selectbox(
                        "Seleccione un municipio para el análisis integrado:",
                        options=["Todos los municipios"] + municipios_comunes
                    )

                    # Aplicar filtros
                    if municipio_integrado == "Todos los municipios":
                        df_lugares_integrado = df_lugares.copy()
                        df_control_integrado = df_control.copy()
                    else:
                        df_lugares_integrado = df_lugares[df_lugares['Municipio'] == municipio_integrado]
                        df_control_integrado = df_control[df_control['Municipio'] == municipio_integrado]

                    # Panel de métricas clave
                    st.subheader("Métricas Clave")

                    # Crear fila de métricas
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        # Total de lugares de visita
                        if municipio_integrado == "Todos los municipios":
                            total_lugares = df_lugares[tipos_lugar].sum().sum()
                        else:
                            total_lugares = df_lugares_integrado[tipos_lugar].iloc[0].sum() if not df_lugares_integrado.empty else 0

                        st.metric("Total Lugares de Visita", int(total_lugares))

                    with col2:
                        # Total de intervenciones
                        total_intervenciones = len(df_control_integrado)
                        st.metric("Total Intervenciones", total_intervenciones)

                    with col3:
                        # Entornos únicos intervenidos
                        if not df_control_integrado.empty and 'Entornos_Lista' in df_control_integrado.columns:
                            entornos_planos = []
                            for lista_entornos in df_control_integrado['Entornos_Lista']:
                                entornos_planos.extend(lista_entornos)

                            entornos_unicos = len(set(entornos_planos))
                        else:
                            entornos_unicos = 0

                        st.metric("Entornos Intervenidos", entornos_unicos)

                    with col4:
                                                    # Período de tiempo
                        if not df_control_integrado.empty:
                            fecha_min = df_control_integrado['Fecha'].min().date()
                            fecha_max = df_control_integrado['Fecha'].max().date()

                            dias_periodo = (fecha_max - fecha_min).days + 1
                        else:
                            dias_periodo = 0

                        st.metric("Período de Análisis (días)", dias_periodo)

                    # Visualización integrada: Relación entre lugares de visita e intervenciones
                    st.subheader("Relación entre Lugares de Visita e Intervenciones")

                    # Preparar datos para la visualización
                    if municipio_integrado == "Todos los municipios" and municipios_comunes:
                        # Crear DataFrame para la comparación
                        datos_comparativos = []

                        for municipio in municipios_comunes:
                            # Datos de lugares
                            df_lugar_mun = df_lugares[df_lugares['Municipio'] == municipio]
                            total_lugares_mun = df_lugar_mun[tipos_lugar].iloc[0].sum() if not df_lugar_mun.empty else 0

                            # Datos de intervenciones
                            total_intervenciones_mun = len(df_control[df_control['Municipio'] == municipio])

                            datos_comparativos.append({
                                'Municipio': municipio,
                                'Total Lugares': total_lugares_mun,
                                'Total Intervenciones': total_intervenciones_mun
                            })

                        if datos_comparativos:  # Verificar que hay datos para graficar
                            df_comparativo = pd.DataFrame(datos_comparativos)

                            # Verificar que el DataFrame tenga las columnas necesarias
                            columnas_necesarias = ['Municipio', 'Total Lugares', 'Total Intervenciones']
                            columnas_faltantes = [col for col in columnas_necesarias if col not in df_comparativo.columns]

                            if not columnas_faltantes:
                                # Verificar que hay datos para al menos un municipio
                                if len(df_comparativo) > 0:
                                    # Crear un gráfico de barras directamente sin usar melt
                                    # (lo cual puede causar errores con DataFrames vacíos)
                                    fig = go.Figure()

                                    # Añadir barras para lugares
                                    fig.add_trace(go.Bar(
                                        x=df_comparativo['Municipio'],
                                        y=df_comparativo['Total Lugares'],
                                        name='Lugares de Visita',
                                        marker_color='rgb(55, 83, 109)'
                                    ))

                                    # Añadir barras para intervenciones
                                    fig.add_trace(go.Bar(
                                        x=df_comparativo['Municipio'],
                                        y=df_comparativo['Total Intervenciones'],
                                        name='Intervenciones',
                                        marker_color='rgb(26, 118, 255)'
                                    ))

                                    # Personalizar gráfico
                                    fig.update_layout(
                                        title='Comparación entre Lugares de Visita e Intervenciones por Municipio',
                                        xaxis_title='Municipio',
                                        yaxis_title='Cantidad',
                                        barmode='group',
                                        height=500,
                                        hovermode='closest'
                                    )

                                    # Mostrar gráfico
                                    st.plotly_chart(fig, use_container_width=True)

                                    # Análisis de correlación
                                    st.subheader("Correlación entre Variables")

                                    if len(df_comparativo) > 1:  # Necesitamos al menos 2 puntos para correlación
                                        # Calcular correlación
                                        correlacion = df_comparativo[['Total Lugares', 'Total Intervenciones']].corr()

                                        # Crear matriz de correlación interactiva con Plotly
                                        fig = px.imshow(
                                            correlacion,
                                            text_auto=True,
                                            color_continuous_scale='RdBu_r',
                                            title='Matriz de Correlación',
                                            labels={'color': 'Correlación'},
                                            height=400
                                        )

                                        # Personalizar gráfico
                                        fig.update_layout(
                                            coloraxis_colorbar=dict(
                                                title='Correlación',
                                                tickvals=[-1, 0, 1],
                                                ticktext=['-1 (Corr. Negativa)', '0 (Sin Corr.)', '1 (Corr. Positiva)']
                                            )
                                        )

                                        # Mostrar gráfico
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Se necesitan datos de al menos dos municipios para calcular la correlación.")
                                else:
                                    st.warning("No hay suficientes datos para generar la comparación entre municipios.")
                            else:
                                st.warning(f"Faltan columnas necesarias en los datos: {', '.join(columnas_faltantes)}")
                        else:
                            st.warning("No se pudieron generar datos comparativos entre municipios.")
                    elif municipio_integrado == "Todos los municipios" and not municipios_comunes:
                        st.warning("No hay municipios comunes entre los datos de lugares de visita y control vectorial.")
                    else:
                        # Para un municipio específico, mostrar detalle por tipo de entorno
                        st.write(f"Análisis detallado para {municipio_integrado}")

                        # Obtener datos de lugares
                        if not df_lugares_integrado.empty:
                            datos_lugares = df_lugares_integrado[tipos_lugar].iloc[0].to_dict()
                        else:
                            datos_lugares = {tipo: 0 for tipo in tipos_lugar}

                        # Obtener datos de intervenciones por entorno
                        if not df_control_integrado.empty and 'Entornos_Lista' in df_control_integrado.columns:
                            entornos_planos = []
                            for lista_entornos in df_control_integrado['Entornos_Lista']:
                                entornos_planos.extend(lista_entornos)

                            datos_intervenciones = pd.Series(entornos_planos).value_counts().to_dict()
                        else:
                            datos_intervenciones = {}

                        # Crear DataFrame para la visualización
                        df_detalle = pd.DataFrame({
                            'Tipo': list(datos_lugares.keys()) + list(datos_intervenciones.keys()),
                            'Categoría': ['Lugar de Visita'] * len(datos_lugares) + ['Intervención'] * len(datos_intervenciones),
                            'Cantidad': list(datos_lugares.values()) + list(datos_intervenciones.values())
                        })

                        # Crear gráfico de barras interactivo con Plotly
                        fig = px.bar(
                            df_detalle,
                            x='Tipo',
                            y='Cantidad',
                            color='Categoría',
                            barmode='group',
                            title=f'Lugares de Visita e Intervenciones en {municipio_integrado}',
                            labels={'Cantidad': 'Cantidad', 'Tipo': 'Tipo', 'Categoría': 'Categoría'},
                            height=500
                        )

                        # Personalizar gráfico
                        fig.update_layout(
                            xaxis_title='Tipo',
                            yaxis_title='Cantidad',
                            legend_title='Categoría',
                            hovermode='closest',
                            xaxis={'categoryorder':'total descending'}
                        )

                        # Mostrar gráfico
                        st.plotly_chart(fig, use_container_width=True)

                    # Análisis temporal de cobertura
                    st.subheader("Análisis Temporal de Cobertura")

                    if not df_control_integrado.empty:
                        # Preparar datos para el análisis temporal
                        df_temporal = df_control_integrado.copy()
                        df_temporal['Fecha_Solo'] = df_temporal['Fecha'].dt.date
                        df_temporal['Mes'] = df_temporal['Fecha'].dt.month
                        df_temporal['Año'] = df_temporal['Fecha'].dt.year

                        # Agrupar por mes
                        df_mensual = df_temporal.groupby(['Año', 'Mes']).size().reset_index()
                        df_mensual.columns = ['Año', 'Mes', 'Intervenciones']

                        # Crear etiquetas de mes-año para el eje x de manera segura
                        def crear_etiqueta_periodo(row):
                            mes = int(row['Mes'])
                            anio = int(row['Año']) 
                            return calendar.month_abbr[mes] + ". " + str(anio)

                        df_mensual['Periodo'] = df_mensual.apply(crear_etiqueta_periodo, axis=1)

                        # Ordenar por fecha
                        df_mensual = df_mensual.sort_values(['Año', 'Mes'])

                        # Crear gráfico de barras interactivo con Plotly
                        fig = px.bar(
                            df_mensual,
                            x='Periodo',
                            y='Intervenciones',
                            text='Intervenciones',
                            title=f'Intervenciones por Mes {"en " + municipio_integrado if municipio_integrado != "Todos los municipios" else ""}',
                            labels={'Intervenciones': 'Número de intervenciones', 'Periodo': 'Mes'},
                            height=400
                        )

                        # Personalizar gráfico
                        fig.update_traces(
                            textposition='outside',
                            hovertemplate='<b>Periodo</b>: %{x}<br><b>Intervenciones</b>: %{y}<extra></extra>'
                        )

                        fig.update_layout(
                            xaxis_title='Periodo',
                            yaxis_title='Número de Intervenciones',
                            hovermode='closest'
                        )

                        # Mostrar gráfico
                        st.plotly_chart(fig, use_container_width=True)

                        # Distribución por día de la semana
                        st.subheader("Distribución de Intervenciones por Día de la Semana")

                        # Agregar día de la semana
                        df_temporal['DiaSemana'] = df_temporal['Fecha'].dt.dayofweek

                        # Contar intervenciones por día de la semana
                        conteo_dias = df_temporal['DiaSemana'].value_counts().reindex(range(7), fill_value=0).reset_index()
                        conteo_dias.columns = ['DiaSemana', 'Intervenciones']

                        # Agregar nombres de días
                        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                        conteo_dias['NombreDia'] = conteo_dias['DiaSemana'].apply(lambda x: dias_semana[x])

                        # Ordenar por día de la semana
                        conteo_dias = conteo_dias.sort_values('DiaSemana')

                        # Crear gráfico de barras interactivo con Plotly
                        fig = px.bar(
                            conteo_dias,
                            x='NombreDia',
                            y='Intervenciones',
                            text='Intervenciones',
                            color='NombreDia',
                            title=f'Intervenciones por Día de la Semana {"en " + municipio_integrado if municipio_integrado != "Todos los municipios" else ""}',
                            labels={'Intervenciones': 'Número de intervenciones', 'NombreDia': 'Día de la semana'},
                            height=400
                        )

                        # Personalizar gráfico
                        fig.update_traces(
                            textposition='outside',
                            hovertemplate='<b>Día</b>: %{x}<br><b>Intervenciones</b>: %{y}<extra></extra>'
                        )

                        fig.update_layout(
                            xaxis_title='Día de la Semana',
                            yaxis_title='Número de Intervenciones',
                            showlegend=False,
                            hovermode='closest'
                        )

                        # Mostrar gráfico
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No hay datos de intervenciones disponibles para el análisis temporal.")
            else:
                st.warning("No se pudieron cargar los datos necesarios para el análisis integrado.")

    # =====================================================================
    # Dashboard 6: HOSPITALES (HOJA5)
    # =====================================================================
    elif dashboard_key == "hoja5":
        st.header("🏥 Análisis de Hospitales y Presencia de Vectores")
        st.markdown("""
        Esta aplicación permite analizar datos sobre hospitales y la presencia de vectores en diferentes municipios.
        Puedes explorar los datos por municipio, hospital o estado de vectores, y ver la distribución temporal.
        """)

        # Cargar los datos
        with st.spinner("Cargando datos..."):
            df_hospitales = cargar_excel("Hoja5")

        # Verificar si se cargaron los datos correctamente
        if df_hospitales.empty:
            st.error("No se pudieron cargar los datos de hospitales. Por favor verifica el archivo Excel.")
        else:
            # Convertir fechas si es necesario
            if 'Fecha' in df_hospitales.columns:
                df_hospitales['Fecha'] = pd.to_datetime(df_hospitales['Fecha'], errors='coerce')

            # Mostrar un resumen de los datos
            st.subheader("Resumen de Datos")
            st.write(f"Total de registros: {len(df_hospitales)}")

            # Mostrar las primeras filas de los datos
            with st.expander("Ver datos cargados"):
                st.dataframe(df_hospitales)

            # Crear filtros
            col1, col2, col3 = st.columns(3)

            with col1:
                # Filtro de municipios
                municipios = sorted(df_hospitales['Municipio'].unique())
                municipio_seleccionado = st.multiselect(
                    "Seleccionar Municipio(s):",
                    options=municipios,
                    default=municipios
                )

            with col2:
                # Filtro de hospitales
                hospitales = sorted(df_hospitales['Hospital'].unique())
                hospital_seleccionado = st.multiselect(
                    "Seleccionar Hospital(es):",
                    options=hospitales,
                    default=[]
                )

            with col3:
                # Filtro de estado de vectores
                if 'Vectores' in df_hospitales.columns:
                    estados_vector = sorted(df_hospitales['Vectores'].unique())
                    vector_seleccionado = st.multiselect(
                        "Estado de Vectores:",
                        options=estados_vector,
                        default=estados_vector
                    )
                else:
                    vector_seleccionado = []
                    st.warning("No se encontró la columna 'Vectores' en los datos.")

            # Aplicar filtros
            df_filtrado = df_hospitales.copy()

            # Filtro de municipio
            if municipio_seleccionado:
                df_filtrado = df_filtrado[df_filtrado['Municipio'].isin(municipio_seleccionado)]

            # Filtro de hospital
            if hospital_seleccionado:
                df_filtrado = df_filtrado[df_filtrado['Hospital'].isin(hospital_seleccionado)]

            # Filtro de vectores
            if vector_seleccionado and 'Vectores' in df_filtrado.columns:
                df_filtrado = df_filtrado[df_filtrado['Vectores'].isin(vector_seleccionado)]

            # Mostrar datos filtrados
            st.subheader("Datos Filtrados")
            st.dataframe(df_filtrado, use_container_width=True)

            # Panel de métricas
            st.subheader("Métricas Clave")
            col_metrica1, col_metrica2, col_metrica3 = st.columns(3)

            with col_metrica1:
                total_municipios = df_filtrado['Municipio'].nunique()
                st.metric("Municipios", total_municipios)

            with col_metrica2:
                total_hospitales = df_filtrado['Hospital'].nunique()
                st.metric("Hospitales", total_hospitales)

            with col_metrica3:
                if 'Vectores' in df_filtrado.columns:
                    # Calcular % de hospitales con vectores
                    vectores_si = df_filtrado[df_filtrado['Vectores'] == 'Si'].shape[0]
                    total_registros = df_filtrado.shape[0]
                    porcentaje_con_vectores = (vectores_si / total_registros * 100) if total_registros > 0 else 0
                    st.metric("% con Vectores", f"{porcentaje_con_vectores:.1f}%")
                else:
                    st.metric("% con Vectores", "N/A")

            # Visualizaciones
            st.header("Visualizaciones")

            # Crear pestañas para diferentes visualizaciones
            tab1, tab2, tab3 = st.tabs(["Por Municipio", "Por Hospital", "Temporal"])

            # Pestaña 1: Análisis por Municipio
            with tab1:
                st.subheader("Análisis por Municipio")

                if not df_filtrado.empty:
                    # Gráfico 1: Distribución de hospitales por municipio
                    conteo_municipios = df_filtrado['Municipio'].value_counts().reset_index()
                    conteo_municipios.columns = ['Municipio', 'Cantidad']

                    fig1 = px.bar(
                        conteo_municipios,
                        x='Municipio',
                        y='Cantidad',
                        color='Municipio',
                        title='Distribución de Hospitales por Municipio',
                        labels={'Cantidad': 'Número de Hospitales', 'Municipio': 'Municipio'},
                        height=400
                    )

                    fig1.update_layout(
                        xaxis_title='Municipio',
                        yaxis_title='Número de Hospitales',
                        showlegend=False,
                        hovermode='closest',
                        xaxis={'categoryorder':'total descending'}
                    )

                    st.plotly_chart(fig1, use_container_width=True)

                    # Gráfico 2: Estado de vectores por municipio (si aplica)
                    if 'Vectores' in df_filtrado.columns:
                        # Crear una tabla pivote para contar vectores por municipio
                        pivot = pd.pivot_table(
                            df_filtrado, 
                            values='Hospital',
                            index=['Municipio'],
                            columns=['Vectores'],
                            aggfunc='count',
                            fill_value=0
                        ).reset_index()

                        # Preparar datos para gráfico apilado
                        categorias_vector = [col for col in pivot.columns if col != 'Municipio']
                        fig2 = go.Figure()

                        # Añadir una barra para cada estado de vector
                        for categoria in categorias_vector:
                            fig2.add_trace(go.Bar(
                                name=categoria,
                                x=pivot['Municipio'],
                                y=pivot[categoria],
                                text=pivot[categoria],
                                textposition='auto'
                            ))

                        # Personalizar gráfico
                        fig2.update_layout(
                            title='Estado de Vectores por Municipio',
                            xaxis_title='Municipio',
                            yaxis_title='Número de Hospitales',
                            barmode='stack',
                            hovermode='closest',
                            xaxis={'categoryorder':'total descending'},
                            height=500
                        )

                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Pestaña 2: Análisis por Hospital
            with tab2:
                st.subheader("Análisis por Hospital")

                if not df_filtrado.empty:
                    # Gráfico 1: Estado de vectores por hospital (si aplica)
                    if 'Vectores' in df_filtrado.columns:
                        # Crear un DataFrame con la información de cada hospital y su estado de vector
                        df_hospital_vector = df_filtrado[['Hospital', 'Vectores']].drop_duplicates()

                        # Contar estados por hospital
                        conteo_vectores = df_hospital_vector['Vectores'].value_counts().reset_index()
                        conteo_vectores.columns = ['Estado', 'Cantidad']

                        # Gráfico circular
                        fig3 = px.pie(
                            conteo_vectores,
                            values='Cantidad',
                            names='Estado',
                            title='Distribución de Estados de Vectores en Hospitales',
                            hole=0.4,
                            height=400
                        )

                        fig3.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hoverinfo='label+percent+value'
                        )

                        fig3.update_layout(
                            uniformtext_minsize=12,
                            uniformtext_mode='hide'
                        )

                        st.plotly_chart(fig3, use_container_width=True)

                    # Gráfico 2: Tabla de hospitales
                    st.subheader("Detalle de Hospitales")

                    # Crear una tabla más detallada con información de cada hospital
                    detalle_hospitales = df_filtrado.groupby('Hospital').agg({
                        'Municipio': 'first',
                        'Vectores': lambda x: x.iloc[0] if 'Vectores' in df_filtrado.columns else "N/A",
                        'Fecha': lambda x: x.min().strftime('%Y-%m-%d') if 'Fecha' in df_filtrado.columns and pd.notnull(x.min()) else "N/A"
                    }).reset_index()

                    # Mostrar tabla
                    st.dataframe(detalle_hospitales, use_container_width=True)
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Pestaña 3: Análisis Temporal
            with tab3:
                st.subheader("Análisis Temporal")

                if not df_filtrado.empty and 'Fecha' in df_filtrado.columns:
                    # Verificar si hay fechas válidas
                    fechas_validas = df_filtrado['Fecha'].notna()

                    if fechas_validas.any():
                        # Gráfico 1: Distribución temporal de visitas/registros
                        df_temp = df_filtrado.copy()
                        df_temp['Fecha_Solo'] = df_temp['Fecha'].dt.date

                        # Contar registros por fecha
                        registros_por_fecha = df_temp.groupby('Fecha_Solo').size().reset_index()
                        registros_por_fecha.columns = ['Fecha', 'Cantidad']

                        # Crear gráfico de línea
                        fig4 = px.line(
                            registros_por_fecha,
                            x='Fecha',
                            y='Cantidad',
                            markers=True,
                            title='Evolución Temporal de Registros',
                            labels={'Cantidad': 'Número de Registros', 'Fecha': 'Fecha'},
                            height=400
                        )

                        fig4.update_traces(
                            line=dict(width=3),
                            marker=dict(size=10),
                            hovertemplate='<b>Fecha</b>: %{x|%Y-%m-%d}<br><b>Registros</b>: %{y}<extra></extra>'
                        )

                        fig4.update_layout(
                            xaxis_title='Fecha',
                            yaxis_title='Número de Registros',
                            hovermode='x unified'
                        )

                        st.plotly_chart(fig4, use_container_width=True)

                        # Gráfico 2: Distribución temporal de vectores (si aplica)
                        if 'Vectores' in df_filtrado.columns:
                            # Agrupar por fecha y estado de vector
                            vectores_tiempo = df_temp.groupby(['Fecha_Solo', 'Vectores']).size().reset_index()
                            vectores_tiempo.columns = ['Fecha', 'Estado', 'Cantidad']

                            # Crear gráfico de área apilada
                            fig5 = px.area(
                                vectores_tiempo,
                                x='Fecha',
                                y='Cantidad',
                                color='Estado',
                                title='Evolución Temporal por Estado de Vectores',
                                labels={'Cantidad': 'Número de Registros', 'Fecha': 'Fecha', 'Estado': 'Estado de Vectores'},
                                height=400
                            )

                            fig5.update_layout(
                                xaxis_title='Fecha',
                                yaxis_title='Número de Registros',
                                hovermode='x unified',
                                legend_title='Estado de Vectores'
                            )

                            st.plotly_chart(fig5, use_container_width=True)
                    else:
                        st.warning("No hay fechas válidas en los datos para realizar el análisis temporal.")
                else:
                    if 'Fecha' not in df_filtrado.columns:
                        st.warning("No se encontró la columna de fecha en los datos.")
                    else:
                        st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Conclusiones y recomendaciones
            st.header("Conclusiones")

            # Mostrar algunas conclusiones basadas en los datos
            if not df_filtrado.empty:
                # Lista de conclusiones
                conclusiones = []

                # Conclusión sobre municipios
                municipio_max = df_filtrado['Municipio'].value_counts().idxmax()
                cant_municipio_max = df_filtrado['Municipio'].value_counts().max()
                conclusiones.append(f"El municipio con más hospitales registrados es **{municipio_max}** con {cant_municipio_max} hospitales.")

                # Conclusión sobre vectores (si aplica)
                if 'Vectores' in df_filtrado.columns:
                    # Calcular porcentajes
                    total_registros = len(df_filtrado)
                    vectores_si = df_filtrado[df_filtrado['Vectores'] == 'Si'].shape[0]
                    vectores_no = df_filtrado[df_filtrado['Vectores'] == 'No'].shape[0]
                    vectores_sd = total_registros - vectores_si - vectores_no

                    porc_si = (vectores_si / total_registros * 100) if total_registros > 0 else 0
                    porc_no = (vectores_no / total_registros * 100) if total_registros > 0 else 0
                    porc_sd = (vectores_sd / total_registros * 100) if total_registros > 0 else 0

                    conclusiones.append(f"De los hospitales analizados, el **{porc_si:.1f}%** tiene presencia confirmada de vectores, el **{porc_no:.1f}%** no tiene vectores, y el **{porc_sd:.1f}%** no tiene datos específicos.")

                    # Municipios con mayor presencia de vectores
                    if vectores_si > 0:
                        vectores_por_municipio = df_filtrado[df_filtrado['Vectores'] == 'Si'].groupby('Municipio').size().sort_values(ascending=False)
                        if len(vectores_por_municipio) > 0:
                            mun_max_vectores = vectores_por_municipio.index[0]
                            cant_max_vectores = vectores_por_municipio.iloc[0]
                            conclusiones.append(f"El municipio con mayor presencia de vectores es **{mun_max_vectores}** con {cant_max_vectores} hospitales afectados.")

                # Conclusión temporal (si aplica)
                if 'Fecha' in df_filtrado.columns and df_filtrado['Fecha'].notna().any():
                    fecha_min = df_filtrado['Fecha'].min().strftime('%Y-%m-%d')
                    fecha_max = df_filtrado['Fecha'].max().strftime('%Y-%m-%d')
                    dias_periodo = (df_filtrado['Fecha'].max() - df_filtrado['Fecha'].min()).days
                    conclusiones.append(f"El período de análisis abarca desde **{fecha_min}** hasta **{fecha_max}**, un total de {dias_periodo} días.")

                # Mostrar conclusiones
                for conclusion in conclusiones:
                    st.markdown(f"- {conclusion}")

                # Recomendaciones
                st.subheader("Recomendaciones")

                recomendaciones = [
                    "Realizar un seguimiento periódico en los hospitales con presencia confirmada de vectores.",
                    "Implementar medidas preventivas en municipios con alta incidencia de vectores.",
                    "Completar la información de hospitales donde el estado de vectores no está determinado."
                ]

                for recomendacion in recomendaciones:
                    st.markdown(f"- {recomendacion}")
            else:
                st.warning("No hay datos suficientes para generar conclusiones.")

    # =====================================================================
    # Dashboard 7: PROTECCIÓN VECTORIAL EN IPS (HOJA6)
    # =====================================================================
    elif dashboard_key == "hoja6":
        st.header("🦟 Análisis de Protección Vectorial en IPS")
        st.markdown("""
        Esta aplicación permite analizar datos sobre la protección contra vectores en diferentes IPS (Instituciones Prestadoras de Servicios de Salud).
        Puedes explorar indicadores como cobertura de toldillos, presencia de vectores, medidas de protección y más.
        """)

        # Función específica para este dashboard
        @st.cache_data
        def cargar_datos_ips():
            """Carga los datos de IPS y protección vectorial desde la Hoja6"""
            try:
                # Cargar datos del Excel
                df = cargar_excel("Hoja6")

                # Asegurar que las columnas de texto sean de tipo string para evitar problemas de ordenamiento
                columnas_texto = [
                    'Nombre de la IPS', 
                    'Municipio', 
                    'Complejidad',
                    'Ventanas abiertas o ventilaciones expuestas',
                    'Angeos en las ventanas o ventilaciones',
                    'Aedes aegypti',
                    'Aedes albopictus',
                    'Presencia de criaderos de mosquitos?',
                    'Los toldillos se encontraban correctamente instalados?',
                    'Capacitación en instalación y manejo de toldillos',
                    'Identificación de prioridades para instalación de métodos de barrera contra insectos',
                    'Observaciones'
                ]

                # Convertir todas las columnas de texto a string
                for col in columnas_texto:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
                        # Reemplazar 'nan' (que resulta de convertir NaN a string) con un valor más apropiado
                        df[col] = df[col].replace('nan', 'No especificado')

                # Convertir columnas numéricas explícitamente
                columnas_numericas = [
                    'Camas urgencias/observación', 
                    'Camas hospitalización', 
                    'Camas totales', 
                    'Toldillos totales', 
                    'Cobertura camas con toldillo'
                ]

                for col in columnas_numericas:
                    if col in df.columns:
                        # Convertir a tipo numérico, los errores se convierten en NaN
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Convertir fechas si es necesario
                if 'Fecha' in df.columns:
                    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

                # Calcular indicadores adicionales si es posible
                if 'Camas totales' in df.columns and 'Toldillos totales' in df.columns:
                    # Calcular déficit de toldillos (si hay menos toldillos que camas)
                    df['Déficit de toldillos'] = df.apply(
                        lambda row: max(0, row['Camas totales'] - row['Toldillos totales']) 
                        if pd.notna(row['Camas totales']) and pd.notna(row['Toldillos totales']) 
                        else np.nan, 
                        axis=1
                    )

                    # Calcular indicador de cobertura (si no existe ya)
                    if 'Cobertura camas con toldillo' not in df.columns:
                        df['Cobertura camas con toldillo'] = df.apply(
                            lambda row: row['Toldillos totales'] / row['Camas totales'] 
                            if pd.notna(row['Camas totales']) and row['Camas totales'] > 0 and pd.notna(row['Toldillos totales']) 
                            else 0,
                            axis=1
                        )

                return df
            except Exception as e:
                st.error(f"Error al cargar datos de IPS: {str(e)}")
                return pd.DataFrame()

        # Función para crear mapa de calor de protección
        def crear_mapa_proteccion(df):
            """Crea un mapa de calor que resume el nivel de protección contra vectores por IPS"""
            try:
                # Seleccionar columnas relevantes para la evaluación
                columnas_proteccion = [
                    'Cobertura camas con toldillo',
                    'Angeos en las ventanas o ventilaciones',
                    'Presencia de criaderos de mosquitos?',
                    'Los toldillos se encontraban correctamente instalados?',
                    'Capacitación en instalación y manejo de toldillos'
                ]

                # Crear un DataFrame para el mapa de calor
                datos_mapa = []

                # Procesar cada IPS
                for _, row in df.iterrows():
                    ips = row['Nombre de la IPS']
                    municipio = row['Municipio']

                    # Evaluar cada factor de protección
                    evaluaciones = {}

                    # Cobertura de toldillos (convertir a valor numérico)
                    if 'Cobertura camas con toldillo' in row:
                        cobertura = row['Cobertura camas con toldillo']
                        if pd.notna(cobertura):
                            # Escala: 0 = sin cobertura, 1 = cobertura completa
                            evaluaciones['Cobertura de toldillos'] = float(cobertura)

                    # Angeos en ventanas
                    if 'Angeos en las ventanas o ventilaciones' in row:
                        angeos = row['Angeos en las ventanas o ventilaciones']
                        if pd.notna(angeos):
                            if isinstance(angeos, str):
                                if 'si' in angeos.lower():
                                    evaluaciones['Angeos en ventanas'] = 1.0
                                elif 'no' in angeos.lower():
                                    evaluaciones['Angeos en ventanas'] = 0.0
                                else:
                                    evaluaciones['Angeos en ventanas'] = 0.5  # Parcial

                    # Ausencia de criaderos (inversa de presencia)
                    if 'Presencia de criaderos de mosquitos?' in row:
                        criaderos = row['Presencia de criaderos de mosquitos?']
                        if pd.notna(criaderos):
                            if isinstance(criaderos, str):
                                if criaderos.lower().startswith('no'):
                                    evaluaciones['Sin criaderos'] = 1.0
                                elif criaderos.lower().startswith('si'):
                                    evaluaciones['Sin criaderos'] = 0.0
                                else:
                                    evaluaciones['Sin criaderos'] = 0.5

                    # Instalación correcta de toldillos
                    if 'Los toldillos se encontraban correctamente instalados?' in row:
                        instalacion = row['Los toldillos se encontraban correctamente instalados?']
                        if pd.notna(instalacion):
                            if isinstance(instalacion, str):
                                if instalacion.lower().startswith('si'):
                                    evaluaciones['Instalación correcta'] = 1.0
                                elif instalacion.lower().startswith('no'):
                                    evaluaciones['Instalación correcta'] = 0.0
                                else:
                                    evaluaciones['Instalación correcta'] = 0.5

                    # Capacitación
                    if 'Capacitación en instalación y manejo de toldillos' in row:
                        capacitacion = row['Capacitación en instalación y manejo de toldillos']
                        if pd.notna(capacitacion):
                            if isinstance(capacitacion, str):
                                if capacitacion.lower().startswith('si'):
                                    evaluaciones['Capacitación'] = 1.0
                                elif capacitacion.lower().startswith('no'):
                                    evaluaciones['Capacitación'] = 0.0
                                else:
                                    evaluaciones['Capacitación'] = 0.5

                    # Añadir cada evaluación al dataset para el mapa de calor
                    for factor, valor in evaluaciones.items():
                        datos_mapa.append({
                            'IPS': ips,
                            'Municipio': municipio,
                            'Factor de protección': factor,
                            'Nivel': valor
                        })

                # Crear DataFrame
                return pd.DataFrame(datos_mapa)
            except Exception as e:
                st.warning(f"No se pudo crear el mapa de protección: {str(e)}")
                return pd.DataFrame()

        # Cargar los datos
        with st.spinner("Cargando datos..."):
            df_ips = cargar_datos_ips()

        # Verificar si se cargaron los datos correctamente
        if df_ips.empty:
            st.error("No se pudieron cargar los datos de IPS. Por favor verifica el archivo Excel.")
        else:
            # Mostrar un resumen de los datos
            st.subheader("Resumen de Datos")
            st.write(f"Total de registros: {len(df_ips)}")

            # Mostrar las primeras filas de los datos
            with st.expander("Ver datos cargados"):
                st.dataframe(df_ips)

            # Crear filtros
            col1, col2, col3 = st.columns(3)

            with col1:
                # Filtro de municipios
                municipios = sorted(df_ips['Municipio'].unique())
                municipio_seleccionado = st.multiselect(
                    "Seleccionar Municipio(s):",
                    options=municipios,
                    default=municipios
                )

            with col2:
                # Filtro de IPS
                # Convertir todos los valores a string antes de ordenar
                df_ips['Nombre de la IPS'] = df_ips['Nombre de la IPS'].astype(str)
                ips_list = sorted(df_ips['Nombre de la IPS'].unique())
                ips_seleccionada = st.multiselect(
                    "Seleccionar IPS:",
                    options=ips_list,
                    default=[]
                )

            with col3:
                # Filtro de complejidad si existe
                if 'Complejidad' in df_ips.columns:
                    # Convertir todos los valores a string antes de ordenar
                    df_ips['Complejidad'] = df_ips['Complejidad'].astype(str)
                    complejidades = sorted(df_ips['Complejidad'].unique())
                    complejidad_seleccionada = st.multiselect(
                        "Nivel de Complejidad:",
                        options=complejidades,
                        default=complejidades
                    )
                else:
                    complejidad_seleccionada = []

            # Aplicar filtros
            df_filtrado = df_ips.copy()

            # Filtro de municipio
            if municipio_seleccionado:
                df_filtrado = df_filtrado[df_filtrado['Municipio'].isin(municipio_seleccionado)]

            # Filtro de IPS
            if ips_seleccionada:
                df_filtrado = df_filtrado[df_filtrado['Nombre de la IPS'].isin(ips_seleccionada)]

            # Filtro de complejidad
            if complejidad_seleccionada and 'Complejidad' in df_filtrado.columns:
                df_filtrado = df_filtrado[df_filtrado['Complejidad'].isin(complejidad_seleccionada)]

            # Mostrar datos filtrados
            st.subheader("Datos Filtrados")
            st.dataframe(df_filtrado, use_container_width=True)

            # Panel de métricas clave
            st.subheader("Métricas Clave")
            col_metrica1, col_metrica2, col_metrica3, col_metrica4 = st.columns(4)

            with col_metrica1:
                # Total de IPS - Asegurarse que los valores son string antes de contar
                df_filtrado['Nombre de la IPS'] = df_filtrado['Nombre de la IPS'].astype(str)
                total_ips = df_filtrado['Nombre de la IPS'].nunique()
                st.metric("IPS Analizadas", total_ips)

            with col_metrica2:
                # Camas totales
                if 'Camas totales' in df_filtrado.columns:
                    total_camas = df_filtrado['Camas totales'].sum()
                    st.metric("Total de Camas", int(total_camas) if not pd.isna(total_camas) else "N/A")
                else:
                    st.metric("Total de Camas", "N/A")

            with col_metrica3:
                # Cobertura promedio de toldillos
                if 'Cobertura camas con toldillo' in df_filtrado.columns:
                    cobertura_promedio = df_filtrado['Cobertura camas con toldillo'].mean() * 100
                    st.metric("Cobertura Promedio de Toldillos", f"{cobertura_promedio:.1f}%" if not pd.isna(cobertura_promedio) else "N/A")
                else:
                    st.metric("Cobertura Promedio de Toldillos", "N/A")

            with col_metrica4:
                # Porcentaje de IPS con Aedes aegypti
                if 'Aedes aegypti' in df_filtrado.columns:
                    ips_con_aedes = df_filtrado[df_filtrado['Aedes aegypti'] == 'Si'].shape[0]
                    porcentaje_aedes = (ips_con_aedes / len(df_filtrado) * 100) if len(df_filtrado) > 0 else 0
                    st.metric("IPS con Aedes aegypti", f"{porcentaje_aedes:.1f}%")
                else:
                    st.metric("IPS con Aedes aegypti", "N/A")

            # Visualizaciones
            st.header("Visualizaciones")

            # Crear pestañas para diferentes visualizaciones
            tab1, tab2, tab3, tab4 = st.tabs([
                "Cobertura de Toldillos", 
                "Presencia de Vectores", 
                "Evaluación de Protección",
                "Detalles por IPS"
            ])

            # Pestaña 1: Cobertura de Toldillos
            with tab1:
                st.subheader("Análisis de Cobertura de Toldillos")

                if not df_filtrado.empty:
                    # Columnas para los gráficos
                    col1, col2 = st.columns(2)

                    with col1:
                        # Gráfico 1: Cobertura de toldillos por IPS
                        if 'Cobertura camas con toldillo' in df_filtrado.columns:
                            # Verificar si hay datos válidos
                            datos_cobertura = df_filtrado[['Nombre de la IPS', 'Municipio', 'Cobertura camas con toldillo']].copy()
                            datos_cobertura = datos_cobertura.dropna(subset=['Cobertura camas con toldillo'])

                            if not datos_cobertura.empty:
                                # Calcular porcentaje
                                datos_cobertura['Porcentaje de Cobertura'] = datos_cobertura['Cobertura camas con toldillo'] * 100

                                # Ordenar por cobertura
                                datos_cobertura = datos_cobertura.sort_values('Porcentaje de Cobertura', ascending=False)

                                # Crear gráfico de barras
                                fig1 = px.bar(
                                    datos_cobertura,
                                    x='Nombre de la IPS',
                                    y='Porcentaje de Cobertura',
                                    color='Municipio',
                                    title='Porcentaje de Cobertura de Toldillos por IPS',
                                    labels={
                                        'Nombre de la IPS': 'IPS', 
                                        'Porcentaje de Cobertura': 'Cobertura (%)'
                                    },
                                    height=500
                                )

                                fig1.update_layout(
                                    xaxis_title='IPS',
                                    yaxis_title='Cobertura (%)',
                                    xaxis={'categoryorder':'total descending'},
                                    hovermode='closest'
                                )

                                # Añadir línea de cobertura objetivo (100%)
                                fig1.add_shape(
                                    type='line',
                                    x0=-0.5,
                                    x1=len(datos_cobertura)-0.5,
                                    y0=100,
                                    y1=100,
                                    line=dict(color='red', width=2, dash='dash'),
                                )

                                # Mostrar gráfico
                                st.plotly_chart(fig1, use_container_width=True)
                            else:
                                st.warning("No hay datos válidos de cobertura para generar el gráfico.")
                        else:
                            st.warning("No se encontraron datos de cobertura de toldillos en el dataset.")

                    with col2:
                        # Gráfico 2: Comparación entre camas y toldillos
                        if 'Camas totales' in df_filtrado.columns and 'Toldillos totales' in df_filtrado.columns:
                            # Filtrar filas con datos válidos
                            datos_validos = df_filtrado.dropna(subset=['Camas totales', 'Toldillos totales'])

                            if not datos_validos.empty:
                                # Crear DataFrame para el gráfico
                                datos_comparacion = datos_validos[['Nombre de la IPS', 'Municipio', 'Camas totales', 'Toldillos totales']].copy()

                                # Asegurar que los datos son numéricos
                                datos_comparacion['Camas totales'] = pd.to_numeric(datos_comparacion['Camas totales'], errors='coerce')
                                datos_comparacion['Toldillos totales'] = pd.to_numeric(datos_comparacion['Toldillos totales'], errors='coerce')

                                # Eliminar filas con NaN después de la conversión
                                datos_comparacion = datos_comparacion.dropna(subset=['Camas totales', 'Toldillos totales'])

                                if not datos_comparacion.empty:
                                    # Convertir a formato largo para el gráfico agrupado
                                    datos_comp_largo = pd.melt(
                                        datos_comparacion,
                                        id_vars=['Nombre de la IPS', 'Municipio'],
                                        value_vars=['Camas totales', 'Toldillos totales'],
                                        var_name='Tipo',
                                        value_name='Cantidad'
                                    )

                                    # Crear gráfico de barras agrupadas
                                    fig2 = px.bar(
                                        datos_comp_largo,
                                        x='Nombre de la IPS',
                                        y='Cantidad',
                                        color='Tipo',
                                        barmode='group',
                                        title='Comparación entre Camas y Toldillos por IPS',
                                        labels={
                                            'Nombre de la IPS': 'IPS', 
                                            'Cantidad': 'Cantidad', 
                                            'Tipo': 'Tipo'
                                        },
                                        height=500
                                    )

                                    fig2.update_layout(
                                        xaxis_title='IPS',
                                        yaxis_title='Cantidad',
                                        xaxis={'categoryorder':'total descending'},
                                        hovermode='closest',
                                        legend_title='Tipo'
                                    )

                                    # Mostrar gráfico
                                    st.plotly_chart(fig2, use_container_width=True)
                                else:
                                    st.warning("No hay datos numéricos válidos para generar la comparación.")
                            else:
                                st.warning("No hay datos válidos para generar la comparación entre camas y toldillos.")
                        else:
                            st.warning("No se encontraron datos completos de camas y toldillos en el dataset.")

                    # Déficit de toldillos por municipio
                    if 'Déficit de toldillos' in df_filtrado.columns:
                        st.subheader("Déficit de Toldillos por Municipio")

                        # Filtrar valores válidos
                        df_deficit = df_filtrado.dropna(subset=['Déficit de toldillos'])

                        if not df_deficit.empty:
                            # Agrupar por municipio
                            deficit_por_municipio = df_deficit.groupby('Municipio')['Déficit de toldillos'].sum().reset_index()
                            deficit_por_municipio = deficit_por_municipio.sort_values('Déficit de toldillos', ascending=False)

                            # Crear gráfico de barras
                            fig3 = px.bar(
                                deficit_por_municipio,
                                x='Municipio',
                                y='Déficit de toldillos',
                                title='Déficit Total de Toldillos por Municipio',
                                labels={
                                    'Municipio': 'Municipio', 
                                    'Déficit de toldillos': 'Déficit (unidades)'
                                },
                                height=400
                            )

                            fig3.update_layout(
                                xaxis_title='Municipio',
                                yaxis_title='Déficit (unidades)',
                                xaxis={'categoryorder':'total descending'},
                                hovermode='closest'
                            )

                            # Mostrar gráfico
                            st.plotly_chart(fig3, use_container_width=True)
                        else:
                            st.warning("No hay datos válidos de déficit de toldillos para mostrar.")
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Pestaña 2: Presencia de Vectores
            with tab2:
                st.subheader("Análisis de Presencia de Vectores")

                if not df_filtrado.empty:
                    # Columnas para los gráficos
                    col1, col2 = st.columns(2)

                    with col1:
                        # Gráfico 1: Presencia de Aedes aegypti
                        if 'Aedes aegypti' in df_filtrado.columns:
                            # Contar presencia por municipio
                            presencia_aegypti = df_filtrado.groupby(['Municipio', 'Aedes aegypti']).size().reset_index(name='Conteo')

                            # Crear gráfico de barras apiladas
                            fig4 = px.bar(
                                presencia_aegypti,
                                x='Municipio',
                                y='Conteo',
                                color='Aedes aegypti',
                                title='Presencia de Aedes aegypti por Municipio',
                                labels={
                                    'Municipio': 'Municipio', 
                                    'Conteo': 'Número de IPS', 
                                    'Aedes aegypti': 'Presencia'
                                },
                                height=400
                            )

                            fig4.update_layout(
                                xaxis_title='Municipio',
                                yaxis_title='Número de IPS',
                                xaxis={'categoryorder':'total descending'},
                                hovermode='closest',
                                legend_title='Presencia'
                            )

                            # Mostrar gráfico
                            st.plotly_chart(fig4, use_container_width=True)
                        else:
                            st.warning("No se encontraron datos sobre Aedes aegypti en el dataset.")

                    with col2:
                        # Gráfico 2: Presencia de Aedes albopictus (si está disponible)
                        if 'Aedes albopictus' in df_filtrado.columns:
                            # Contar presencia por municipio
                            presencia_albopictus = df_filtrado.groupby(['Municipio', 'Aedes albopictus']).size().reset_index(name='Conteo')

                            # Crear gráfico de barras apiladas
                            fig5 = px.bar(
                                presencia_albopictus,
                                x='Municipio',
                                y='Conteo',
                                color='Aedes albopictus',
                                title='Presencia de Aedes albopictus por Municipio',
                                labels={
                                    'Municipio': 'Municipio', 
                                    'Conteo': 'Número de IPS', 
                                    'Aedes albopictus': 'Presencia'
                                },
                                height=400
                            )

                            fig5.update_layout(
                                xaxis_title='Municipio',
                                yaxis_title='Número de IPS',
                                xaxis={'categoryorder':'total descending'},
                                hovermode='closest',
                                legend_title='Presencia'
                            )

                            # Mostrar gráfico
                            st.plotly_chart(fig5, use_container_width=True)
                        else:
                            st.warning("No se encontraron datos sobre Aedes albopictus en el dataset.")

                    # Gráfico 3: Presencia de criaderos
                    if 'Presencia de criaderos de mosquitos?' in df_filtrado.columns:
                        st.subheader("Presencia de Criaderos de Mosquitos")

                        # Procesar respuestas para categorizarlas mejor
                        df_criaderos = df_filtrado.copy()
                        df_criaderos['Criaderos'] = df_criaderos['Presencia de criaderos de mosquitos?'].apply(
                            lambda x: 'Si' if isinstance(x, str) and x.lower().startswith('si') else 
                            ('No' if isinstance(x, str) and x.lower().startswith('no') else 'No especificado')
                        )

                        # Contar por categoría
                        conteo_criaderos = df_criaderos['Criaderos'].value_counts().reset_index()
                        conteo_criaderos.columns = ['Estado', 'Conteo']

                        # Crear gráfico circular
                        fig6 = px.pie(
                            conteo_criaderos,
                            values='Conteo',
                            names='Estado',
                            title='Presencia de Criaderos de Mosquitos',
                            hole=0.4,
                            height=400
                        )

                        fig6.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            marker=dict(
                                colors=['#ff6b6b', '#51cf66', '#868e96'],
                                line=dict(color='#fff', width=2)
                            )
                        )

                        # Mostrar gráfico
                        st.plotly_chart(fig6, use_container_width=True)
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Pestaña 3: Evaluación de Protección
            with tab3:
                st.subheader("Evaluación de Medidas de Protección")

                if not df_filtrado.empty:
                    # Crear mapa de calor de factores de protección
                    df_mapa = crear_mapa_proteccion(df_filtrado)

                    if not df_mapa.empty:
                        # Crear una tabla pivote para el mapa de calor
                        pivot_mapa = df_mapa.pivot_table(
                            values='Nivel',
                            index='IPS',
                            columns='Factor de protección',
                            aggfunc='mean'
                        ).fillna(0)

                        # Calcular puntuación general
                        if len(pivot_mapa.columns) > 0:
                            pivot_mapa['Puntuación Total'] = pivot_mapa.mean(axis=1)
                            pivot_mapa = pivot_mapa.sort_values('Puntuación Total', ascending=False)

                        # Crear mapa de calor
                        fig7 = px.imshow(
                            pivot_mapa,
                            labels=dict(x='Factor de Protección', y='IPS', color='Nivel'),
                            x=pivot_mapa.columns,
                            y=pivot_mapa.index,
                            color_continuous_scale='RdYlGn',  # Rojo a verde
                            title='Evaluación de Factores de Protección por IPS',
                            height=600
                        )

                        fig7.update_layout(
                            xaxis_title='Factor de Protección',
                            yaxis_title='IPS',
                            coloraxis_colorbar=dict(
                                title='Nivel',
                                tickvals=[0, 0.5, 1],
                                ticktext=['Bajo', 'Medio', 'Alto']
                            )
                        )

                        st.plotly_chart(fig7, use_container_width=True)

                        # Mostrar tabla de puntuaciones
                        st.subheader("Puntuaciones de Protección Vectorial")
                        st.dataframe(pivot_mapa.round(2), use_container_width=True)
                    else:
                        st.warning("No se pudieron generar suficientes datos para la evaluación de protección.")

                    # Gráfico de estado de toldillos correctamente instalados
                    if 'Los toldillos se encontraban correctamente instalados?' in df_filtrado.columns:
                        st.subheader("Estado de Instalación de Toldillos")

                        # Procesar respuestas
                        df_instalacion = df_filtrado.copy()
                        df_instalacion['Estado de Instalación'] = df_instalacion['Los toldillos se encontraban correctamente instalados?'].apply(
                            lambda x: 'Correcta' if isinstance(x, str) and x.lower().startswith('si') else 
                            ('Incorrecta' if isinstance(x, str) and x.lower().startswith('no') else 'No especificado')
                        )

                        # Contar por categoría
                        conteo_instalacion = df_instalacion['Estado de Instalación'].value_counts().reset_index()
                        conteo_instalacion.columns = ['Estado', 'Conteo']

                        # Crear gráfico de barras
                        fig8 = px.bar(
                            conteo_instalacion,
                            x='Estado',
                            y='Conteo',
                            color='Estado',
                            title='Estado de Instalación de Toldillos',
                            labels={
                                'Estado': 'Estado de Instalación', 
                                'Conteo': 'Número de IPS'
                            },
                            height=400,
                            color_discrete_map={
                                'Correcta': '#51cf66',
                                'Incorrecta': '#ff6b6b',
                                'No especificado': '#868e96'
                            }
                        )

                        fig8.update_layout(
                            xaxis_title='Estado de Instalación',
                            yaxis_title='Número de IPS',
                            showlegend=False,
                            hovermode='closest'
                        )

                        # Mostrar gráfico
                        st.plotly_chart(fig8, use_container_width=True)
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Pestaña 4: Detalles por IPS
            with tab4:
                st.subheader("Detalles por IPS")

                if not df_filtrado.empty:
                    # Asegurar que todos los valores de IPS son strings
                    df_filtrado['Nombre de la IPS'] = df_filtrado['Nombre de la IPS'].astype(str)
                    # Selector de IPS
                    ips_detalle = st.selectbox(
                        "Seleccionar IPS para ver detalles:",
                        options=sorted(df_filtrado['Nombre de la IPS'].unique())
                    )

                    # Filtrar datos para la IPS seleccionada
                    datos_ips = df_filtrado[df_filtrado['Nombre de la IPS'] == ips_detalle]

                    if not datos_ips.empty:
                        # Crear columnas para mostrar información
                        col1, col2 = st.columns(2)

                        with col1:
                            # Información básica
                            st.subheader(f"{ips_detalle}")
                            st.write(f"**Municipio:** {datos_ips['Municipio'].iloc[0]}")

                            if 'Complejidad' in datos_ips.columns:
                                st.write(f"**Nivel de complejidad:** {datos_ips['Complejidad'].iloc[0]}")

                            # Información de camas y toldillos - manejo seguro
                            if 'Camas totales' in datos_ips.columns:
                                camas_totales = datos_ips['Camas totales'].iloc[0]
                                if pd.notna(camas_totales):
                                    st.write(f"**Camas totales:** {int(camas_totales)}")
                                else:
                                    st.write("**Camas totales:** No disponible")

                            if 'Camas urgencias/observación' in datos_ips.columns:
                                camas_urgencias = datos_ips['Camas urgencias/observación'].iloc[0]
                                if pd.notna(camas_urgencias):
                                    st.write(f"**Camas en urgencias/observación:** {int(camas_urgencias)}")
                                else:
                                    st.write("**Camas en urgencias/observación:** No disponible")

                            if 'Camas hospitalización' in datos_ips.columns:
                                camas_hosp = datos_ips['Camas hospitalización'].iloc[0]
                                if pd.notna(camas_hosp):
                                    st.write(f"**Camas hospitalización:** {int(camas_hosp)}")
                                else:
                                    st.write("**Camas hospitalización:** No disponible")

                            if 'Toldillos totales' in datos_ips.columns:
                                toldillos = datos_ips['Toldillos totales'].iloc[0]
                                if pd.notna(toldillos):
                                    st.write(f"**Toldillos totales:** {int(toldillos)}")
                                else:
                                    st.write("**Toldillos totales:** No disponible")

                            if 'Cobertura camas con toldillo' in datos_ips.columns:
                                cobertura = datos_ips['Cobertura camas con toldillo'].iloc[0]
                                if pd.notna(cobertura):
                                    st.write(f"**Cobertura de toldillos:** {cobertura*100:.1f}%")
                                else:
                                    st.write("**Cobertura de toldillos:** No disponible")

                        with col2:
                            # Estado de vectores y protección
                            st.subheader("Estado de Vectores y Protección")

                            if 'Aedes aegypti' in datos_ips.columns:
                                st.write(f"**Presencia de Aedes aegypti:** {datos_ips['Aedes aegypti'].iloc[0]}")

                            if 'Aedes albopictus' in datos_ips.columns:
                                st.write(f"**Presencia de Aedes albopictus:** {datos_ips['Aedes albopictus'].iloc[0]}")

                            if 'Presencia de criaderos de mosquitos?' in datos_ips.columns:
                                st.write(f"**Presencia de criaderos:** {datos_ips['Presencia de criaderos de mosquitos?'].iloc[0]}")

                            if 'Ventanas abiertas o ventilaciones expuestas' in datos_ips.columns:
                                st.write(f"**Ventanas/ventilaciones expuestas:** {datos_ips['Ventanas abiertas o ventilaciones expuestas'].iloc[0]}")

                            if 'Angeos en las ventanas o ventilaciones' in datos_ips.columns:
                                st.write(f"**Angeos en ventanas:** {datos_ips['Angeos en las ventanas o ventilaciones'].iloc[0]}")

                            if 'Los toldillos se encontraban correctamente instalados?' in datos_ips.columns:
                                st.write(f"**Toldillos correctamente instalados:** {datos_ips['Los toldillos se encontraban correctamente instalados?'].iloc[0]}")

                            if 'Capacitación en instalación y manejo de toldillos' in datos_ips.columns:
                                st.write(f"**Capacitación en toldillos:** {datos_ips['Capacitación en instalación y manejo de toldillos'].iloc[0]}")

                        # Observaciones
                        if 'Observaciones' in datos_ips.columns and datos_ips['Observaciones'].iloc[0] != 'No especificado':
                            st.subheader("Observaciones")
                            observaciones = datos_ips['Observaciones'].iloc[0]

                            # Formatear observaciones como lista si contiene múltiples puntos
                            if isinstance(observaciones, str):
                                # Dividir por saltos de línea o guiones
                                if '\n' in observaciones or '-' in observaciones:
                                    items = [item.strip() for item in observaciones.replace('\r', '').split('\n') if item.strip()]

                                    if not items:  # Si no se dividió correctamente por saltos de línea
                                        items = [item.strip() for item in observaciones.split('-') if item.strip()]

                                    for item in items:
                                        if item:
                                            # Añadir un guión si no comienza con uno
                                            if not item.startswith('-'):
                                                item = '- ' + item
                                            st.markdown(item)
                                else:
                                    st.write(observaciones)

                        # Recomendaciones basadas en datos
                        st.subheader("Recomendaciones")

                        recomendaciones = []

                        # Revisar cobertura de toldillos
                        if 'Cobertura camas con toldillo' in datos_ips.columns:
                            cobertura = datos_ips['Cobertura camas con toldillo'].iloc[0]
                            if pd.notna(cobertura):
                                if cobertura < 0.5:
                                    recomendaciones.append("Incrementar la cantidad de toldillos para mejorar la cobertura (menos del 50% actual).")
                                elif cobertura < 0.9:
                                    recomendaciones.append("Complementar la dotación de toldillos para alcanzar una cobertura óptima (90-100%).")

                        # Revisar instalación de toldillos
                        if 'Los toldillos se encontraban correctamente instalados?' in datos_ips.columns:
                            instalacion = datos_ips['Los toldillos se encontraban correctamente instalados?'].iloc[0]
                            if isinstance(instalacion, str) and instalacion.lower().startswith('no'):
                                recomendaciones.append("Realizar capacitación práctica en la correcta instalación de toldillos.")

                        # Revisar presencia de criaderos
                        if 'Presencia de criaderos de mosquitos?' in datos_ips.columns:
                            criaderos = datos_ips['Presencia de criaderos de mosquitos?'].iloc[0]
                            if isinstance(criaderos, str) and criaderos.lower().startswith('si'):
                                recomendaciones.append("Eliminar inmediatamente los criaderos de mosquitos y establecer un programa de revisión periódica.")

                        # Revisar angeos
                        if 'Angeos en las ventanas o ventilaciones' in datos_ips.columns:
                            angeos = datos_ips['Angeos en las ventanas o ventilaciones'].iloc[0]
                            if isinstance(angeos, str) and 'no' in angeos.lower():
                                recomendaciones.append("Instalar angeos en ventanas y ventilaciones para prevenir la entrada de vectores.")

                        # Revisar capacitación
                        if 'Capacitación en instalación y manejo de toldillos' in datos_ips.columns:
                            capacitacion = datos_ips['Capacitación en instalación y manejo de toldillos'].iloc[0]
                            if isinstance(capacitacion, str) and capacitacion.lower().startswith('no'):
                                recomendaciones.append("Programar capacitación al personal sobre instalación y manejo de toldillos.")

                        # Mostrar recomendaciones
                        if recomendaciones:
                            for recomendacion in recomendaciones:
                                st.markdown(f"- {recomendacion}")
                        else:
                            st.write("No se generaron recomendaciones específicas para esta IPS.")
                    else:
                        st.warning("No se encontraron datos para la IPS seleccionada.")
                else:
                    st.warning("No hay datos disponibles para los filtros seleccionados.")

            # Sección de conclusiones generales
            st.header("Conclusiones Generales")

            if not df_filtrado.empty:
                # Generar conclusiones basadas en el análisis
                conclusiones = []

                # Conclusión sobre cobertura de toldillos
                if 'Cobertura camas con toldillo' in df_filtrado.columns:
                    # Filtrar valores válidos (no NaN) antes de calcular estadísticas
                    df_cob = df_filtrado.dropna(subset=['Cobertura camas con toldillo'])

                    if not df_cob.empty:
                        cobertura_promedio = df_cob['Cobertura camas con toldillo'].mean() * 100
                        cobertura_minima = df_cob['Cobertura camas con toldillo'].min() * 100
                        ips_baja_cobertura = df_cob[df_cob['Cobertura camas con toldillo'] < 0.5]['Nombre de la IPS'].nunique()

                        if ips_baja_cobertura > 0:
                            porcentaje_baja = (ips_baja_cobertura / df_cob['Nombre de la IPS'].nunique()) * 100
                            conclusiones.append(f"El {porcentaje_baja:.1f}% de las IPS analizadas presenta una cobertura de toldillos inferior al 50%.")

                        conclusiones.append(f"La cobertura promedio de toldillos en las IPS analizadas es del {cobertura_promedio:.1f}%, con un mínimo de {cobertura_minima:.1f}%.")

                # Conclusión sobre presencia de vectores
                if 'Aedes aegypti' in df_filtrado.columns:
                    ips_con_aedes = df_filtrado[df_filtrado['Aedes aegypti'] == 'Si']['Nombre de la IPS'].nunique()
                    total_ips = df_filtrado['Nombre de la IPS'].nunique()

                    if total_ips > 0:
                        porcentaje_aedes = (ips_con_aedes / total_ips) * 100
                        conclusiones.append(f"Se ha detectado presencia de Aedes aegypti en el {porcentaje_aedes:.1f}% de las IPS analizadas.")

                        # Identificar municipios más afectados
                        if ips_con_aedes > 0:
                            municipios_afectados = df_filtrado[df_filtrado['Aedes aegypti'] == 'Si']['Municipio'].value_counts()
                            if len(municipios_afectados) > 0:
                                mun_mas_afectado = municipios_afectados.index[0]
                                cant_ips_afectadas = municipios_afectados.iloc[0]
                                conclusiones.append(f"El municipio más afectado por la presencia de Aedes aegypti es {mun_mas_afectado}, con {cant_ips_afectadas} IPS afectadas.")

                # Conclusión sobre criaderos
                if 'Presencia de criaderos de mosquitos?' in df_filtrado.columns:
                    df_con_criaderos = df_filtrado[df_filtrado['Presencia de criaderos de mosquitos?'].astype(str).str.lower().str.startswith('si')]
                    ips_con_criaderos = df_con_criaderos['Nombre de la IPS'].nunique()

                    if ips_con_criaderos > 0:
                        porcentaje_criaderos = (ips_con_criaderos / df_filtrado['Nombre de la IPS'].nunique()) * 100
                        conclusiones.append(f"Se han detectado criaderos de mosquitos en el {porcentaje_criaderos:.1f}% de las IPS analizadas.")

                # Conclusión sobre instalación de toldillos
                if 'Los toldillos se encontraban correctamente instalados?' in df_filtrado.columns:
                    df_instalacion_incorrecta = df_filtrado[df_filtrado['Los toldillos se encontraban correctamente instalados?'].astype(str).str.lower().str.startswith('no')]
                    ips_inst_incorrecta = df_instalacion_incorrecta['Nombre de la IPS'].nunique()

                    if ips_inst_incorrecta > 0:
                        porcentaje_incorrecto = (ips_inst_incorrecta / df_filtrado['Nombre de la IPS'].nunique()) * 100
                        conclusiones.append(f"En el {porcentaje_incorrecto:.1f}% de las IPS, los toldillos no se encuentran correctamente instalados.")

                # Mostrar conclusiones
                for conclusion in conclusiones:
                    st.markdown(f"- {conclusion}")

                # Recomendaciones generales
                st.subheader("Recomendaciones Generales")

                recomendaciones_generales = [
                    "Implementar un programa de seguimiento regular para garantizar la correcta instalación y uso de toldillos en todas las IPS.",
                    "Priorizar la distribución de toldillos adicionales en las IPS con cobertura insuficiente.",
                    "Establecer protocolos de eliminación inmediata de criaderos de mosquitos en las áreas circundantes a las IPS.",
                    "Reforzar la capacitación del personal en medidas de prevención y control vectorial.",
                    "Instalar angeos en ventanas y ventilaciones en todas las áreas críticas de las IPS."
                ]

                for recomendacion in recomendaciones_generales:
                    st.markdown(f"- {recomendacion}")
            else:
                st.warning("No hay datos suficientes para generar conclusiones generales.")

# Pie de página común para todos los dashboards
st.markdown("---")
st.caption(f"Sistema Integrado de Análisis Vectorial - Departamento de Tolima, Colombia - Última actualización: {datetime.now().strftime('%d-%m-%Y')}")


# In[ ]:




