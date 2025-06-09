import streamlit as st
import pandas as pd
import random
import json
import os
import logging

# Import the loader function to create datasets (we will use it for polygon_df)
from data import loader

# Import the new optimizer classes and constants from the file you just created/modified
from optimizers_classes import SASCOpt, TRUCK_CAPACITY, HQ_POLYGON, VRP

# Set up basic logging for Streamlit to see messages in console
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    for handler in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def asignar_actividades(df):
    actividades_adicionales = ["Plantación", "Riego", "Preparación de terreno", "Descanso"]
    probabilities = [0.4, 0.2, 0.2, 0.2]

    df = df.copy()
    df["Actividad"] = "Entrega"

    entregas = df.groupby(["Day", "Polygon"]).size().reset_index(name="conteo")

    extra_actividades = []
    for _, row in entregas.iterrows():
        if 'Polygon' in row and 'Day' in row:
            act = random.choices(actividades_adicionales, probabilities)[0]
            extra_actividades.append({"Day": row["Day"], "Polygon": row["Polygon"], "ActividadExtra": act})
        else:
            logging.warning(f"Missing 'Day' or 'Polygon' in row for assigning activities: {row}. Skipping this row.")

    if extra_actividades:
        df_extra = pd.DataFrame(extra_actividades)
        df = pd.merge(df, df_extra, on=["Day", "Polygon"], how="left")
        
        df["Actividad"] = df.apply(
            lambda row: row["Actividad"] + " + " + row["ActividadExtra"] if pd.notna(row["ActividadExtra"]) else row["Actividad"],
            axis=1
        )
        df.drop(columns=["ActividadExtra"], inplace=True)
    return df

# --- LAYOUT Y LÓGICA DE LA APLICACIÓN STREAMLIT ---
st.set_page_config(page_title="🌱 Reforestación Inteligente", layout="wide")
st.title("🧠 Dashboard de Calendarización y Rutas con Simulated Annealing")

st.markdown("""
Esta aplicación permite:
1. **📥 Cargar tus propios archivos de demanda y precios.**
2. **📦 Calendarizar pedidos optimizados y generar rutas de entrega con Simulated Annealing.**
3. **🌿 Consultar entregas diarias por polígono y especie.**
4. **🚚 Visualizar las acciones de los camiones (cargas, entregas, retornos).**
""")

if "optimization_results" not in st.session_state:
    st.session_state["optimization_results"] = {
        "df_solution": None,        
        "df_orders": None,          
        "vrp_actions_data": None,   
        "best_cost": None,          
        "polygon_df": None          
    }

tab1, tab2, tab3 = st.tabs(["📦 Ejecutar Optimización", "🌿 Detalle de Entregas", "🚚 Rutas de Camiones"])

with tab1:
    st.header("⚙️ Configuración y Ejecución del Modelo")

    st.subheader("📥 Carga de Archivos de Datos")
    st.info("Sube tu `demand.csv` y `supplier_prices.csv`. El archivo de polígonos (`polygon_df`) con las coordenadas se cargará automáticamente del sistema (`data/loader.py`).")
    
    demand_file = st.file_uploader("Demanda (`demand.csv`)", type="csv", key="demand_uploader")
    price_file = st.file_uploader("Precios (`supplier_prices.csv`)", type="csv", key="prices_uploader")

    df_demand = None
    df_prices = None
    
    if demand_file and price_file:
        try:
            df_demand = pd.read_csv(demand_file)
            df_prices = pd.read_csv(price_file)
            
            st.subheader("🔎 Vista previa de los Datos")
            st.write("--- Demanda ---")
            st.dataframe(df_demand)
            st.write("--- Precios de Proveedores ---")
            st.dataframe(df_prices)
            
            _, _, polygon_df_loaded = loader.create_datasets()
            st.session_state["optimization_results"]["polygon_df"] = polygon_df_loaded 
            st.success("Archivos de demanda y precios cargados. Coordenadas de polígonos obtenidas. ¡Listo para optimizar!")

        except Exception as e:
            st.error(f"Error al leer los archivos CSV: {e}. Asegúrate de que el formato sea correcto.")
            logging.exception("Error durante la lectura de archivos CSV:") 
            df_demand = None 
            df_prices = None
            st.stop() 
    else:
        st.warning("Por favor, sube ambos archivos (`demand.csv` y `supplier_prices.csv`) para habilitar la optimización.")

    st.subheader("Parámetros del Algoritmo de Simulated Annealing:")
    col1, col2 = st.columns(2)
    with col1:
        iterations = st.number_input("Número de Iteraciones (SA)", min_value=100, max_value=10000, value=2000, step=100)
        initial_temp = st.number_input("Temperatura Inicial", min_value=100.0, max_value=50000.0, value=10000.0, step=100.0)
    with col2:
        cooling_rate = st.number_input("Tasa de Enfriamiento", min_value=0.9, max_value=0.9999, value=0.995, step=0.001, format="%.4f")
        transport_cost = st.number_input("Costo de Transporte por Camión/Día", min_value=1000.0, max_value=10000.0, value=4500.0, step=100.0)
    
    st.info(f"La capacidad del camión está fijada en {TRUCK_CAPACITY} unidades (definido en `optimizer_classes.py`).")

    if st.button("🚀 Ejecutar Optimización (SA + VRP)"):
        if df_demand is not None and df_prices is not None and st.session_state["optimization_results"]["polygon_df"] is not None:
            os.makedirs("./output", exist_ok=True) 
            
            with st.spinner("Ejecutando el algoritmo de optimización... Esto puede tardar varios minutos."):
                optimizer = SASCOpt(
                    demand_df=df_demand,
                    prices_df=df_prices,
                    polygon_df=st.session_state["optimization_results"]["polygon_df"], 
                    max_load=TRUCK_CAPACITY,
                    transport_cost=transport_cost,
                    iterations=iterations,
                    initial_temp=initial_temp,
                    cooling_rate=cooling_rate,
                    log_file='./output/SA_Supply_Chain.log' 
                )
                
                try:
                    best_solution_raw, best_cost, supply_chain_json, vrp_actions_json = optimizer.run()
                    
                    st.session_state["optimization_results"]["best_cost"] = best_cost
                    
                    st.session_state["optimization_results"]["df_solution"] = pd.DataFrame(
                        best_solution_raw,
                        columns=["Day", "Polygon", "Specie", "Supplier", "Amount"]
                    )
                    
                    st.session_state["optimization_results"]["df_orders"] = st.session_state["optimization_results"]["df_solution"].groupby(
                        ["Day", "Supplier", "Specie"]
                    )["Amount"].sum().reset_index()

                    st.session_state["optimization_results"]["vrp_actions_data"] = json.loads(vrp_actions_json)

                    st.success(f"✅ Optimización completada. Costo Total: ${best_cost:,.2f}")
                    st.subheader("📦 Tabla Resumen de Pedidos Optimizados")
                    st.dataframe(st.session_state["optimization_results"]["df_orders"], use_container_width=True)

                except Exception as e:
                    st.error(f"Error durante la ejecución del algoritmo: {e}")
                    logging.exception("Error durante la ejecución de SASCOpt:") 
                    st.stop() 
        else:
            st.error("Por favor, asegúrate de haber subido ambos archivos de demanda y precios y que los datos se hayan cargado correctamente antes de ejecutar la optimización.")

with tab2:
    st.header("🌿 Detalle Diario de Entregas y Actividades")
    if st.session_state["optimization_results"]["df_solution"] is not None:
        df_solution_activities = asignar_actividades(st.session_state["optimization_results"]["df_solution"])
        
        # OBTENER TODOS LOS DÍAS POSIBLES HASTA EL ÚLTIMO DÍA DE CUALQUIER ACTIVIDAD
        max_day_orders = df_solution_activities["Day"].max() if not df_solution_activities.empty else 0
        
        max_day_vrp = 0
        if st.session_state["optimization_results"]["vrp_actions_data"]:
            max_day_vrp = max(action_set['day'] for action_set in st.session_state["optimization_results"]["vrp_actions_data"])
        
        # El rango de días a mostrar debe incluir todos los días desde el 1 hasta el día máximo
        # en que hubo pedidos O actividad VRP.
        max_overall_day = max(max_day_orders, max_day_vrp)
        dias = list(range(1, max_overall_day + 1)) # Genera una secuencia de días completa
        
        if dias: 
            dia_seleccionado = st.selectbox("Selecciona un día para ver las actividades:", dias)
            
            # FILTRADO CORREGIDO Y ROBUSTO:
            df_dia = df_solution_activities[df_solution_activities["Day"] == dia_seleccionado].sort_values("Polygon")
            
            # Si para un día seleccionado no hay pedidos, mostrar un mensaje
            if df_dia.empty:
                st.info(f"No se programaron pedidos para el Día {dia_seleccionado}. La tabla de actividades estará vacía.")
            
            st.subheader(f"📋 Día {dia_seleccionado} - Actividades de Entrega y Otros")
            st.dataframe(df_dia, use_container_width=True)
        else:
            st.warning("No hay actividades de entrega programadas.")
    else:
        st.warning("Por favor, ejecuta la optimización primero en la pestaña 'Ejecutar Optimización' para ver los detalles.")

with tab3:
    st.header("🚚 Rutas de Camiones y Acciones VRP")
    if st.session_state["optimization_results"]["vrp_actions_data"] is not None:
        vrp_data = st.session_state["optimization_results"]["vrp_actions_data"]
        
        # OBTENER TODOS LOS DÍAS POSIBLES HASTA EL ÚLTIMO DÍA DE ACTIVIDAD VRP
        max_day_vrp = max(action_set['day'] for action_set in vrp_data) if vrp_data else 0
        vrp_days = list(range(1, max_day_vrp + 1)) # Genera una secuencia de días VRP completa
        
        if vrp_days: 
            selected_vrp_day = st.selectbox("Selecciona un día para ver las acciones del camión:", vrp_days)
            
            st.subheader(f"📋 Día {selected_vrp_day} - Detalles de la Ruta y Acciones del Camión")
            
            # Filtra las acciones VRP para el día seleccionado
            day_vrp_actions_for_display = [action_set for action_set in vrp_data if action_set['day'] == selected_vrp_day]
            
            if day_vrp_actions_for_display:
                for action_set in day_vrp_actions_for_display: 
                    st.markdown(f"#### Acciones del Camión para el Día {action_set['day']}")
                    
                    truck_actions_raw = action_set.get('actions', []) 
                    
                    actual_delivered_items = []
                    for act_item in truck_actions_raw: 
                        if act_item.get('type') == 'deliver': 
                            actual_delivered_items.append(act_item.get('order')) 
                    
                    actual_delivered_items = [item for item in actual_delivered_items if item is not None]

                    delivered_df = pd.DataFrame(actual_delivered_items)
                    if not delivered_df.empty:
                        st.write("Órdenes Entregadas:")
                        st.dataframe(delivered_df, use_container_width=True)
                    else:
                        st.info("No se realizaron entregas en este día.")

                    st.write("Secuencia de Acciones del Camión:")
                    
                    truck_actions_display = []
                    for action_dict in truck_actions_raw: 
                        description = ""
                        details_dict = action_dict.copy() 
                        
                        action_type = action_dict.get('type') 

                        if action_type == 'load':
                            description = f"Carga de {action_dict.get('amount_loaded', 'N/A')} unidades en HQ ({HQ_POLYGON})."
                        elif action_type == 'deliver':
                            order_info = action_dict.get('order', {}) 
                            description = (
                                f"Entrega de {order_info.get('amount', 'N/A')} unidades en Polígono {order_info.get('polygon', 'N/A')}. "
                                f"Origen: {action_dict.get('from_loc', 'N/A')}, Destino: {action_dict.get('to_loc', 'N/A')}, Tiempo de Viaje: {action_dict.get('travel_time', 'N/A'):.1f} min."
                            )
                        elif action_type == 'return':
                            description = (
                                f"Regreso a HQ ({HQ_POLYGON}). "
                                f"Origen: {action_dict.get('from_loc', 'N/A')}, Destino: {action_dict.get('to_loc', 'N/A')}, Tiempo de Viaje: {action_dict.get('travel_time', 'N/A'):.1f} min."
                            )
                        
                        details_dict['Descripción'] = description 
                        truck_actions_display.append(details_dict)

                    truck_actions_df = pd.DataFrame(truck_actions_display)
                    
                    display_cols = ['type', 'Descripción']
                    available_cols = [col for col in display_cols if col in truck_actions_df.columns]
                    st.dataframe(truck_actions_df[available_cols], use_container_width=True)
                    st.markdown("---") 
            else:
                st.info("No se registraron acciones de camiones para este día.")
        else:
            st.warning("No hay datos de acciones de camiones disponibles de la optimización.")
    else:
        st.warning("Por favor, ejecuta la optimización primero en la pestaña 'Ejecutar Optimización' para ver los detalles de las rutas.")