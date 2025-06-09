import numpy as np
import pandas as pd
import random
import os
import logging
from typing import List, Tuple, Any, Dict
from tqdm import tqdm
import json
from math import sqrt

# --- Constants ---
HQ_POLYGON = 18
MAX_TIME = 360  # minutes
LOADING_TIME = 30
UNLOADING_TIME = 30
TRUCK_CAPACITY = 8000 

# --- Utility Functions ---

def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_time_matrix(polygon_df: pd.DataFrame) -> pd.DataFrame:
    polygons = polygon_df['Poligono'].tolist()
    coords = polygon_df.set_index('Poligono')[['X', 'Y']].to_dict('index')
    time_data = []
    for i in polygons:
        for j in polygons:
            if i in coords and j in coords: # Asegurarse de que las coordenadas existan
                dist = euclidean_distance(list(coords[i].values()), list(coords[j].values())) / 1000  # KM
                time_minutes = np.ceil((dist / 10) * 60) # Asumiendo 10 km/hr de velocidad media
                time_data.append({'origin': i, 'target': j, 'time': time_minutes})
            else:
                logging.warning(f"Coordenadas faltantes para polígono {i} o {j}. No se puede calcular el tiempo.")

    time_df = pd.DataFrame(time_data)
    
    time_pivot = time_df.pivot(index='origin', columns='target', values='time')
    all_polygons_set = set(polygons)

    for p in all_polygons_set:
        if p not in time_pivot.index:
            time_pivot.loc[p] = np.nan
        if p not in time_pivot.columns:
            time_pivot[p] = np.nan
        time_pivot.loc[p, p] = 0 # El tiempo de un polígono a sí mismo es 0

    return time_pivot

# FUNCIÓN PARA CONVERTIR TIPOS NUMPY (Solución al error de serialización)
def convert_numpy_types(obj):
    """
    Recursively converts numpy.int64, numpy.float64, etc. to native Python int/float.
    This is used by json.dumps to handle NumPy types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        # Si es un array NumPy o una lista/tupla, convertir sus elementos
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, dict):
        # Si es un diccionario, convertir sus valores
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    return obj


# --- VRP Class ---

class VRP:
    def __init__(self, inventory_df: pd.DataFrame, polygon_df: pd.DataFrame):
        self.inventory_df = inventory_df.copy()
        self.polygon_df = polygon_df
        self.time_df = compute_time_matrix(polygon_df)
        self.reset()

    def reset(self):
        self.current_state = {
            'location': HQ_POLYGON,
            'time': 0,
            'load': 0,
            'inventory': [],
            'delivered': []
        }
        self.action_history = []
        self.remaining = self._get_initial_remaining()

    def _get_initial_remaining(self) -> pd.DataFrame:
        if 'polygon' not in self.inventory_df.columns or 'amount' not in self.inventory_df.columns:
            logging.warning("VRP inventory_df must contain 'polygon' and 'amount' columns. Returning empty DataFrame for remaining.")
            return pd.DataFrame(columns=['polygon', 'amount'])
        return self.inventory_df.groupby('polygon').agg({'amount': 'sum'}).reset_index()

    def get_state(self) -> dict:
        return self.current_state.copy()

    def get_actions(self, state: dict) -> List[dict]:
        actions = []
        if state['location'] == HQ_POLYGON and state['time'] + LOADING_TIME <= MAX_TIME and not state['inventory'] and not self.remaining.empty and self.remaining['amount'].sum() > 0:
            actions.append({'type': 'load'})
        
        for order in state['inventory']:
            polygon = order['polygon']
            if polygon not in self.time_df.index or state['location'] not in self.time_df.columns:
                logging.warning(f"Polygon {polygon} or current location {state['location']} not in time matrix for delivery. Skipping action.")
                continue

            travel_time = self.time_df.at[state['location'], polygon]
            
            time_to_reach_deliver = travel_time if state['location'] != polygon else 0
            time_to_unload = UNLOADING_TIME
            time_to_return = self.time_df.at[polygon, HQ_POLYGON] 

            total_trip_time = time_to_reach_deliver + time_to_unload + time_to_return

            if state['time'] + total_trip_time <= MAX_TIME:
                actions.append({'type': 'deliver', 'order': order})
        
        if state['location'] != HQ_POLYGON and not state['inventory']:
            travel_time_to_hq = self.time_df.at[state['location'], HQ_POLYGON]
            if state['time'] + travel_time_to_hq <= MAX_TIME:
                actions.append({'type': 'return'})
        return actions

    def protocol(self, actions: List[dict], state: dict) -> dict:
        for action in actions:
            if action['type'] == 'deliver':
                return action
        for action in actions:
            if action['type'] == 'load':
                return action
        for action in actions:
            if action['type'] == 'return':
                return action
        return None

    def apply_action(self, action: dict):
        if action['type'] == 'load':
            self._load_truck()
        elif action['type'] == 'deliver':
            self._deliver_order(action['order'])
        elif action['type'] == 'return':
            self._return_to_hq()

    def _load_truck(self):
        capacity = TRUCK_CAPACITY
        new_inventory_on_truck = []
        new_remaining_demand_at_hq = [] 
        
        if not self.remaining.empty:
            self.remaining = self.remaining.sort_values(by='polygon').reset_index(drop=True)

        indices_loaded = []
        
        for idx, row in self.remaining.iterrows():
            amount = row['amount']
            if amount <= capacity:
                new_inventory_on_truck.append(row.to_dict())
                capacity -= amount
                indices_loaded.append(idx)
            else:
                if capacity > 0:
                    partial_load_amount = capacity
                    new_inventory_on_truck.append({'polygon': row['polygon'], 'amount': partial_load_amount})
                    remaining_after_partial_load = amount - partial_load_amount
                    new_remaining_demand_at_hq.append({'polygon': row['polygon'], 'amount': remaining_after_partial_load})
                    capacity = 0 
                else:
                    new_remaining_demand_at_hq.append(row.to_dict())
                break 
        
        self.remaining = self.remaining.drop(indices_loaded).reset_index(drop=True)
        self.remaining = pd.concat([self.remaining, pd.DataFrame(new_remaining_demand_at_hq)], ignore_index=True)

        self.current_state['inventory'].extend(new_inventory_on_truck)
        self.current_state['load'] = TRUCK_CAPACITY - capacity 
        self.current_state['time'] += LOADING_TIME
        
        self.action_history.append({
            'type': 'load', 
            'amount_loaded': self.current_state['load'], 
            'loaded_items_details': new_inventory_on_truck 
        })

    def _deliver_order(self, order: dict):
        state = self.current_state
        polygon_to_deliver = order['polygon']
        amount_to_deliver = order['amount']

        if polygon_to_deliver not in self.time_df.index or state['location'] not in self.time_df.columns:
            logging.error(f"Cannot deliver to polygon {polygon_to_deliver}. Not in time matrix or current location {state['location']} invalid.")
            return

        travel_time = self.time_df.at[state['location'], polygon_to_deliver]
        
        if state['location'] != polygon_to_deliver:
            state['time'] += travel_time
            state['location'] = polygon_to_deliver 
        
        state['time'] += UNLOADING_TIME 

        removed_from_inventory_idx = -1
        for i, inv_item in enumerate(state['inventory']):
            if inv_item['polygon'] == polygon_to_deliver and inv_item['amount'] == amount_to_deliver:
                removed_from_inventory_idx = i
                break
        
        if removed_from_inventory_idx != -1:
            state['inventory'].pop(removed_from_inventory_idx)
            state['load'] -= amount_to_deliver
            state['delivered'].append(order) 
            self.action_history.append({
                'type': 'deliver', 
                'order': order, 
                'from_loc': state['location'], 
                'to_loc': polygon_to_deliver, 
                'travel_time': travel_time
            })
        else:
            logging.warning(f"Attempted to deliver {order} but it was not found in current truck inventory.")


    def _return_to_hq(self):
        state = self.current_state
        if state['location'] != HQ_POLYGON:
            travel_time = self.time_df.at[state['location'], HQ_POLYGON]
            state['time'] += travel_time
            state['location'] = HQ_POLYGON
            self.action_history.append({
                'type': 'return', 
                'from_loc': state['location'], 
                'to_loc': HQ_POLYGON, 
                'travel_time': travel_time
            })

    def run(self) -> Tuple[Dict, List[Dict]]:
        while True:
            actions = self.get_actions(self.current_state)
            
            if not actions:
                break 
            
            selected_action = self.protocol(actions, self.current_state)
            if not selected_action: 
                break
            
            self.apply_action(selected_action)

            if self.current_state['time'] >= MAX_TIME:
                break
            
            if not self.current_state['inventory'] and (self.remaining.empty or self.remaining['amount'].sum() == 0):
                if self.current_state['location'] != HQ_POLYGON:
                    travel_to_hq_time = self.time_df.at[self.current_state['location'], HQ_POLYGON]
                    if self.current_state['time'] + travel_to_hq_time <= MAX_TIME:
                        self._return_to_hq()
                break 

        return self.current_state, self.action_history

# --- Warehouse Class ---

class Warehouse:
    """
    Central warehouse that stores inventory by species and day.
    """
    def __init__(self):
        self.daily_inventory: Dict[int, List[Dict[str, Any]]] = {}  # {day: [{'polygon': x, 'specie': y, 'amount': z}]}
        self.max_capacity: int = TRUCK_CAPACITY * 2 
        self.remaining_capacity: int = self.max_capacity

    def receive_deliveries(self, day: int, deliveries: List[Tuple[int, int, str, str, float]]):
        if day not in self.daily_inventory:
            self.daily_inventory[day] = []
        
        for _, polygon, specie, _, amount in deliveries: 
            if self.remaining_capacity >= amount:
                self.daily_inventory[day].append({
                    'polygon': polygon,
                    'specie': specie,
                    'amount': amount
                })
                self.remaining_capacity -= amount
            else:
                logging.warning(f"Warehouse full! Could not receive {amount} units for {specie} at polygon {polygon} on day {day}. Capacity left: {self.remaining_capacity}")

    def get_available_orders(self, day: int) -> pd.DataFrame:
        eligible_days = [d for d in self.daily_inventory if d <= day - 3] 
        if not eligible_days:
            return pd.DataFrame(columns=['polygon', 'specie', 'amount'])
        
        records = []
        for d in sorted(eligible_days): 
            records.extend(self.daily_inventory[d])
        
        if not records:
            return pd.DataFrame(columns=['polygon', 'specie', 'amount'])

        df = pd.DataFrame(records)
        grouped = df.groupby('polygon').agg({'amount': 'sum'}).reset_index()
        return grouped
        
    def update_after_delivery(self, delivered_orders: List[Dict], day: int):
        eligible_days = [d for d in self.daily_inventory if d <= day - 3]
        total_removed = 0
        
        temp_delivered_orders = [d.copy() for d in delivered_orders]

        for d in sorted(eligible_days):
            current_day_inventory = self.daily_inventory[d]
            
            for i in range(len(current_day_inventory) - 1, -1, -1):
                inv_entry = current_day_inventory[i]
                
                for delivered_idx in range(len(temp_delivered_orders) - 1, -1, -1):
                    delivered_item = temp_delivered_orders[delivered_idx]
                    
                    if inv_entry['polygon'] == delivered_item['polygon']:
                        
                        amount_to_process = min(inv_entry['amount'], delivered_item['amount'])
                        
                        inv_entry['amount'] -= amount_to_process
                        delivered_item['amount'] -= amount_to_process
                        total_removed += amount_to_process
                        
                        if inv_entry['amount'] == 0:
                            current_day_inventory.pop(i) 
                        
                        if delivered_item['amount'] == 0:
                            temp_delivered_orders.pop(delivered_idx) 
                        
                        break 
                
                if not temp_delivered_orders: 
                    break
            
            if not temp_delivered_orders: 
                break
        
        self.remaining_capacity = min(self.remaining_capacity + total_removed, self.max_capacity)

    def is_empty(self):
        return all(not self.daily_inventory[d] for d in self.daily_inventory)


# --- SASCOpt Class ---

class SASCOpt:
    """
    Optimizes supply chain scheduling using Simulated Annealing.
    """
    def __init__(
        self,
        demand_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        polygon_df: pd.DataFrame,
        max_load: int = TRUCK_CAPACITY,
        transport_cost: float = 4500, 
        iterations: int = 1000,
        initial_temp: float = 10000,
        cooling_rate: float = 0.995,
        log_file: str = './output/SA_Supply_Chain.log'
    ):
        self.demand_df = demand_df.copy()
        self.prices_df = prices_df.copy()
        self.polygon_df = polygon_df.copy()
        self.max_load = max_load
        self.transport_cost = transport_cost
        self.iterations = iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.log_file = log_file

        self.demands = self._aggregate_demands()
        self.species = self.demands['specie'].unique()
        self.suppliers = self.prices_df['supplier'].unique()

        self._setup_logging()

    def _aggregate_demands(self) -> pd.DataFrame:
        return self.demand_df.groupby(['polygon', 'specie'])['demand'].sum().reset_index()

    def _setup_logging(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
        logging.basicConfig(filename=self.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
    def initial_solution(self) -> Tuple[List[Tuple[int, Any, Any, Any, float]], List[Dict]]:
        """
        Genera una solución inicial de calendarización de pedidos y simula las acciones VRP asociadas.
        Esta versión produce un 'vrp_action_history' para la solución inicial.
        """
        remaining = self.demands.copy()
        order_schedule = []
        vrp_action_history = [] 
        warehouse = Warehouse() 

        day = 0
        max_simulation_days = 365 

        while (remaining['demand'].sum() > 0 or not warehouse.is_empty()) and day < max_simulation_days:
            day += 1
            
            day_plan = [] 
            current_day_demands_to_schedule = remaining[remaining['demand'] > 0].sample(frac=0.5 if day < 5 else 1.0, random_state=random.randint(0,1000))

            for idx, row in current_day_demands_to_schedule.iterrows():
                polygon, specie, demand = row['polygon'], row['specie'], row['demand']
                
                if warehouse.remaining_capacity < demand:
                    continue 
                
                valid_suppliers = self.prices_df[self.prices_df['specie'] == specie]['supplier'].unique()
                if len(valid_suppliers) == 0:
                    continue
                supplier = random.choice(valid_suppliers)
                
                day_plan.append((day, polygon, specie, supplier, demand))
                warehouse.receive_deliveries(day, [(day, polygon, specie, supplier, demand)]) 
                remaining.loc[idx, 'demand'] = 0 

            if day_plan:
                order_schedule.extend(day_plan)

            available_inventory = warehouse.get_available_orders(day)
            
            if not available_inventory.empty:
                vrp_optimizer = VRP(
                    inventory_df=available_inventory,
                    polygon_df=self.polygon_df
                )
                final_state, action_history = vrp_optimizer.run()

                if action_history: 
                    vrp_action_history.append({
                        "day": day,
                        "actions": action_history 
                    })
                
                if final_state['delivered']:
                    warehouse.update_after_delivery(final_state['delivered'], day)
        
            if remaining['demand'].sum() == 0 and warehouse.is_empty():
                break
        
        if day >= max_simulation_days:
            logging.warning(f"Initial solution generation reached max_simulation_days ({max_simulation_days}) without fulfilling all demand or emptying warehouse.")

        order_schedule.sort(key=lambda x: x[0])
        
        return order_schedule, vrp_action_history


    def fitness(self, schedule: List[Tuple[int, Any, Any, Any, float]]) -> float:
        cost = 0
        if not schedule: 
            return float('inf') 

        horizon_days = max(entry[0] for entry in schedule)
        
        for day in range(1, horizon_days + 1):
            day_orders = [entry for entry in schedule if entry[0] == day]
            day_suppliers = set()
            
            if not day_orders: 
                continue

            for (_, polygon, specie, supplier, amount) in day_orders:
                price_row = self.prices_df[
                    (self.prices_df['specie'] == specie) &
                    (self.prices_df['supplier'] == supplier)
                ]
                if not price_row.empty:
                    unit_price = price_row.iloc[0]['price']
                    cost += unit_price * amount
                else:
                    cost += 1e9 
                day_suppliers.add(supplier)
            
            cost += self.transport_cost * len(day_suppliers) 
        
        total_initial_demand = self.demands['demand'].sum()
        total_scheduled_amount = sum(s[4] for s in schedule)
        unfulfilled_demand_penalty = max(0.0, total_initial_demand - total_scheduled_amount) * 1e9 
        cost += unfulfilled_demand_penalty

        return cost

    def neighbor(self, schedule: List[Tuple[int, Any, Any, Any, float]]) -> List[Tuple[int, Any, Any, Any, float]]:
        new_schedule = [list(item) for item in schedule] 
        if not new_schedule:
            return []

        idx = random.randint(0, len(new_schedule) - 1)
        day, polygon, specie, current_supplier, amount = new_schedule[idx]
        
        valid_suppliers = self.prices_df[self.prices_df['specie'] == specie]['supplier'].unique()
        if len(valid_suppliers) > 1: 
            new_supplier = random.choice([s for s in valid_suppliers if s != current_supplier])
            new_schedule[idx] = (day, polygon, specie, new_supplier, amount)
        
        return [tuple(item) for item in new_schedule] 

    def run(self) -> Tuple[List[Tuple[int, Any, Any, Any, float]], float, str, str]:
        logging.info("Starting initial solution generation and VRP simulation for it...")
        order_schedule, vrp_actions = self.initial_solution() 
        
        current_solution = order_schedule
        current_cost = self.fitness(current_solution) 
        
        best_solution = current_solution
        best_cost = current_cost
        best_vrp_actions = vrp_actions 
        
        temp = self.initial_temp

        logging.info(f"SA initialized. Initial cost: {current_cost:.2f}")

        for i in tqdm(range(self.iterations), desc="Simulated Annealing Progress"):
            new_solution_candidate = self.neighbor(current_solution)
            new_cost = self.fitness(new_solution_candidate) 

            accept = new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temp)
            
            if accept:
                current_solution = new_solution_candidate
                current_cost = new_cost
                
                if new_cost < best_cost:
                    best_solution = current_solution
                    best_cost = new_cost
            temp *= self.cooling_rate
            logging.info(f"Iteration {i+1}: Current Cost = {current_cost:.2f}, Best Cost = {best_cost:.2f}, Temp = {temp:.2f}")

        logging.info(f"Optimization finished. Best solution found with cost: {best_cost:.2f}")

        # MODIFICACIÓN APLICANDO convert_numpy_types
        supply_chain_json = json.dumps(
            [{"day": day, "polygon": polygon, "specie": specie, "supplier": supplier, "amount": amount}
             for day, polygon, specie, supplier, amount in best_solution],
            indent=2,
            default=convert_numpy_types 
        )

        # MODIFICACIÓN APLICANDO convert_numpy_types
        vrp_actions_json = json.dumps(
            best_vrp_actions,
            indent=2,
            default=convert_numpy_types 
        )

        return best_solution, best_cost, supply_chain_json, vrp_actions_json
