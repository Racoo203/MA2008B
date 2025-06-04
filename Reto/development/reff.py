import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="🧠 Reforestación Inteligente", layout="wide")
st.title("🌱 Planificación Estratégica de Reforestación en el Altiplano Mexicano")
st.markdown("""
Este dashboard presenta una visualización avanzada del proceso de restauración ecológica
impulsado por CONAFOR. Se enfoca en el modelado matemático y la optimización logística
para coordinar la compra, almacenamiento, distribución y siembra de especies vegetales.

El *Polígono 18* actúa como centro logístico previo a la distribución hacia los polígonos de siembra (1 al 32, excluyendo el 18).
""")

# --- SIMULACIÓN DE DATOS (REEMPLAZABLE POR DATOS REALES) ---
np.random.seed(42)
fechas = pd.date_range("2025-06-01", "2025-06-30")
actividades = ["Plantación", "Riego", "Descanso", "Preparación de terreno", "Entrega"]
poligonos = [f"Polígono {i}" for i in range(1, 33) if i != 18]
rutas = ["Ruta A", "Ruta B", "Ruta C"]
especies = ["Agave lechuguilla", "Agave salmiana", "Agave scabra", "Agave striata", "Opuntia cantabrigiensis", 
            "Opuntia engelmani", "Opuntia robusta", "Opuntia streptacanta", "Prosopis laevigata", "Yucca filifera"]

# --- CREACIÓN DEL DATAFRAME ---
data = pd.DataFrame({
    "Fecha": np.random.choice(fechas, 100),
    "Actividad": np.random.choice(actividades, 100),
    "Polígono": np.random.choice(poligonos, 100),
    "Ruta": np.random.choice(rutas, 100),
    "Especie": np.random.choice(especies, 100),
    "Horas trabajadas": np.random.randint(3, 9, 100),
    "Gasto ($)": np.random.randint(500, 3000, 100)
})

# --- KPIs GLOBALES ---
st.header("📊 Indicadores Globales del Proyecto")
col1, col2, col3 = st.columns(3)
col1.metric("🌲 Actividades totales", len(data))
col2.metric("💵 Gasto total", f"${data['Gasto ($)'].sum():,.0f}")
col3.metric("⏱️ Horas totales", f"{data['Horas trabajadas'].sum()} hrs")

# --- FILTROS ---
st.sidebar.header("🎯 Filtros Interactivos")
actividad = st.sidebar.multiselect("🔧 Actividad", opciones := data["Actividad"].unique(), default=opciones)
poli = st.sidebar.multiselect("📍 Polígonos", opciones_p := data["Polígono"].unique(), default=opciones_p)

filtro = (data["Actividad"].isin(actividad)) & (data["Polígono"].isin(poli))
df_filtrado = data[filtro]

# --- TABLA FILTRADA DEL MES COMPLETO ---
st.subheader("📅 Registros del mes completo filtrados")
if not df_filtrado.empty:
    st.dataframe(df_filtrado, use_container_width=True)
else:
    st.warning("No hay registros con los filtros seleccionados.")

# --- GANTT ---
st.subheader("📅 Diagrama de Gantt de Actividades")
df_gantt = df_filtrado.copy()
if not df_gantt.empty:
    df_gantt["Fin"] = df_gantt["Fecha"] + pd.Timedelta(hours=2)
    fig = px.timeline(df_gantt, x_start="Fecha", x_end="Fin", y="Polígono", color="Actividad",
                      title="Actividades Programadas por Polígono")
    fig.update_layout(height=500, xaxis_title="Fecha", yaxis_title="Polígono")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay datos disponibles para mostrar en el diagrama de Gantt con los filtros actuales.")

# --- MAPA DE DÍAS DE DESCANSO ---
st.subheader("🛌 Mapa de Frecuencia de Descansos")
conteo = df_filtrado.groupby(["Fecha", "Actividad"]).size().unstack(fill_value=0)
descanso = conteo.get("Descanso", pd.Series(0, index=fechas))
fig2 = px.bar(descanso.reset_index(), x="Fecha", y="Descanso",
              title="Frecuencia de Descansos por Día (Filtrada)")
st.plotly_chart(fig2, use_container_width=True)

# --- DISTRIBUCIÓN DE ESPECIES ---
st.subheader("🌿 Distribución de Especies por Polígono")
fig3 = px.sunburst(df_filtrado, path=["Especie", "Polígono"], values="Gasto ($)",
                   color="Especie", title="Asignación Presupuestal por Especie y Zona (Filtrada)")
st.plotly_chart(fig3, use_container_width=True)

# --- COMENTARIO FINAL ---
st.markdown("""
Este sistema integra componentes logísticos, agronómicos y computacionales para garantizar que la cadena de suministro
funcione de forma eficiente, considerando restricciones de capacidad, tiempos de aclimatación y rutas óptimas.
Idealmente, este tipo de sistema se vincula con modelos de optimización por metas y algoritmos metaheurísticos como
Algoritmos Genéticos, Recocido Simulado o Colonia de Hormigas para encontrar soluciones viables y robustas en contextos reales.
""")
