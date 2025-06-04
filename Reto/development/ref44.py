import streamlit as st
import pandas as pd
import random
import multiprocessing
from joblib import Parallel, delayed
import json
from datetime import datetime
import os


def asignar_actividades(df):
    actividades_adicionales = ["Plantación", "Riego", "Preparación de terreno", "Descanso"]
    probabilities = [0.4, 0.2, 0.2, 0.2]  # puedes ajustar

    df = df.copy()
    df["Actividad"] = "Entrega"  # por default

    # Agrupar por día y polígono para añadir una actividad adicional si hay entrega
    entregas = df.groupby(["Day", "Polygon"]).size().reset_index(name="conteo")

    extra_actividades = []
    for _, row in entregas.iterrows():
        act = random.choices(actividades_adicionales, probabilities)[0]
        extra_actividades.append({"Day": row["Day"], "Polygon": row["Polygon"], "ActividadExtra": act})

    df_extra = pd.DataFrame(extra_actividades)

    # Unir la actividad extra
    df = pd.merge(df, df_extra, on=["Day", "Polygon"], how="left")

    # Unir ambas actividades en la misma celda si se desea
    df["Actividad"] = df["Actividad"] + " + " + df["ActividadExtra"]
    df.drop(columns=["ActividadExtra"], inplace=True)

    return df

# ---------------- MONTECARLO GENETIC OPTIMIZER ----------------
class MonteCarloSupplyChainOptimizer:
    def __init__(self, demand_df, prices_df, max_load=8000, transport_cost=4500, iterations=1000, n_jobs=-1):
        self.original_demand_df = demand_df.copy()
        self.prices_df = prices_df.copy()
        self.max_load = max_load
        self.transport_cost = transport_cost
        self.iterations = iterations
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()

        self.demands = self.original_demand_df.groupby(['polygon', 'specie'])['demand'].sum().reset_index()

    def generate_random_solution(self):
        remaining = self.demands.sample(frac=1).copy()
        schedule = []
        day = 0

        while remaining['demand'].sum() > 0:
            day += 1
            remaining_capacity = self.max_load
            day_plan = []
            fulfilled_indices = []

            for idx, row in remaining[remaining['demand'] > 0].iterrows():
                if row['demand'] > remaining_capacity:
                    continue
                suppliers = self.prices_df[self.prices_df['specie'] == row['specie']]['supplier'].tolist()
                if not suppliers:
                    continue
                supplier = random.choice(suppliers)
                day_plan.append((day, row['polygon'], row['specie'], supplier, row['demand']))
                remaining_capacity -= row['demand']
                fulfilled_indices.append(idx)

            schedule.extend(day_plan)
            remaining.loc[fulfilled_indices, 'demand'] = 0

        return schedule

    def fitness(self, schedule):
        cost = 0
        for day in range(1, max(s[0] for s in schedule) + 1):
            day_orders = [s for s in schedule if s[0] == day]
            suppliers = set()
            for (_, _, specie, supplier, amount) in day_orders:
                row = self.prices_df[(self.prices_df['specie'] == specie) & (self.prices_df['supplier'] == supplier)]
                if row.empty:
                    return float('inf')
                cost += row.iloc[0]['price'] * amount
                suppliers.add(supplier)
            cost += self.transport_cost * len(suppliers)
        return cost

    def evaluate_one_solution(self, i):
        candidate = self.generate_random_solution()
        candidate_cost = self.fitness(candidate)
        return candidate, candidate_cost

    def run(self):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate_one_solution)(i) for i in range(self.iterations)
        )
        best_solution, best_cost = min(results, key=lambda x: x[1])
        return best_solution, best_cost

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="🌱 Reforestación Inteligente", layout="wide")
st.title("🧠 Dashboard de Calendarización con Algoritmo Genético")

st.markdown("""
Esta aplicación permite:
1. **📦 Calendarizar pedidos optimizados con algoritmo genético**
2. **🌿 Consultar entregas diarias por polígono y especie**
""")

if "df_solution" not in st.session_state:
    st.session_state["df_solution"] = None

if "df_orders" not in st.session_state:
    st.session_state["df_orders"] = None

# Tabs
tab1, tab2 = st.tabs(["📦 Calendarización de Pedidos", "🌿 Entregas Diarias"])

with tab1:
    st.header("📥 Subida de Archivos")
    demand_file = st.file_uploader("Demanda (`demand.csv`)", type="csv")
    price_file = st.file_uploader("Precios (`supplier_prices.csv`)", type="csv")

    if demand_file and price_file:
        df_demand = pd.read_csv(demand_file)
        df_prices = pd.read_csv(price_file)

        st.subheader("🔎 Vista previa de los datos")
        st.dataframe(df_demand)
        st.dataframe(df_prices)

        if st.button("🧬 Ejecutar Algoritmo Genético"):
            optimizer = MonteCarloSupplyChainOptimizer(df_demand, df_prices, iterations=2000)
            best_solution, best_cost = optimizer.run()

            df_solution = pd.DataFrame(best_solution, columns=["Day", "Polygon", "Specie", "Supplier", "Amount"])
            df_orders = df_solution.groupby(["Day", "Supplier", "Specie"])["Amount"].sum().reset_index()

            st.session_state["df_solution"] = df_solution
            st.session_state["df_orders"] = df_orders

            st.success(f"✅ Calendarización completada. Costo total: ${best_cost:,.2f}")
            st.subheader("📦 Tabla General de Pedidos")
            st.dataframe(df_orders, use_container_width=True)
    else:
        st.info("Sube ambos archivos para continuar.")

with tab2:
    st.header("🌿 Detalle de Entregas por Día")
    df_solution = st.session_state["df_solution"]
    if st.session_state["df_solution"] is not None:
        df_solution = asignar_actividades(st.session_state["df_solution"])
        dias = sorted(df_solution["Day"].unique())
        dia_seleccionado = st.selectbox("Selecciona un día", dias)

        df_dia = df_solution[df_solution["Day"] == dia_seleccionado].sort_values("Polygon")

        st.subheader(f"📋 Día {dia_seleccionado} - Actividades de Entrega")
        st.dataframe(df_dia, use_container_width=True)
    else:
        st.warning("Primero realiza la calendarización en la pestaña anterior.")