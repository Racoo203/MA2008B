import numpy as np
import pandas as pd
import random
import os
import logging
from typing import List, Tuple, Any
from tqdm import tqdm

import json
import pandas as pd
import numpy as np
from math import sqrt

# --- Constants ---
HQ_POLYGON = 18
MAX_TIME = 360  # minutes
LOADING_TIME = 30
UNLOADING_TIME = 30
TRUCK_CAPACITY = 1000

# --- Utility Functions ---

def load_polygons(filepath):
    return pd.read_csv(filepath)

def euclidean_distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_time_matrix(polygon_df):
    polygons = polygon_df['Poligono'].tolist()
    coords = polygon_df.set_index('Poligono')[['X', 'Y']].to_dict('index')
    time_data = []
    for i in polygons:
        for j in polygons:
            dist = euclidean_distance(list(coords[i].values()), list(coords[j].values())) / 1000  # KM
            time_minutes = np.ceil((dist / 10) * 60)
            time_data.append({'origin': i, 'target': j, 'time': time_minutes})
    time_df = pd.DataFrame(time_data)
    return time_df.pivot(index='origin', columns='target', values='time')

# --- VRP Class ---

class VRP:
    def __init__(self, inventory_df, polygon_df):
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

    def _get_initial_remaining(self):
        # Aggregate demand for day 1 by polygon
        return self.inventory_df.groupby('polygon').agg({'amount': 'sum'}).reset_index()

    def get_state(self):
        return self.current_state.copy()

    def get_actions(self, state):
        actions = []
        # Load at HQ
        if state['location'] == HQ_POLYGON and state['time'] + LOADING_TIME <= MAX_TIME and not state['inventory']:
            actions.append({'type': 'load'})
        # Deliveries
        for order in state['inventory']:
            polygon = order['polygon']
            travel_time = self.time_df.at[state['location'], polygon]
            total_time = travel_time + UNLOADING_TIME + self.time_df.at[polygon, HQ_POLYGON]
            if state['location'] != polygon and state['time'] + total_time <= MAX_TIME:
                actions.append({'type': 'deliver', 'order': order})
            elif state['location'] == polygon and state['time'] + UNLOADING_TIME + self.time_df.at[polygon, HQ_POLYGON] <= MAX_TIME:
                actions.append({'type': 'deliver', 'order': order})
        # Return to HQ
        if state['location'] != HQ_POLYGON:
            travel_time = self.time_df.at[state['location'], HQ_POLYGON]
            if state['time'] + travel_time <= MAX_TIME:
                actions.append({'type': 'return'})
        return actions

    def protocol(self, actions, state):
        # Prioritize deliver > load > return
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

    def apply_action(self, action):
        state = self.current_state
        if action['type'] == 'load':
            self._load_truck()
        elif action['type'] == 'deliver':
            self._deliver_order(action['order'])
        elif action['type'] == 'return':
            self._return_to_hq()

    def _load_truck(self):
        capacity = TRUCK_CAPACITY
        new_inventory = []
        new_remaining = []
        for _, row in self.remaining.iterrows():
            amount = row['amount']
            if amount <= capacity:
                new_inventory.append(row.to_dict())
                capacity -= amount
            else:
                new_remaining.append(row.to_dict())
        self.current_state['inventory'] = new_inventory
        self.remaining = pd.DataFrame(new_remaining)
        self.current_state['load'] = TRUCK_CAPACITY - capacity
        self.current_state['time'] += LOADING_TIME
        self.action_history.append(('load', len(new_inventory)))

    def _deliver_order(self, order):
        state = self.current_state
        travel_time = self.time_df.at[state['location'], order['polygon']]
        state['time'] += travel_time + UNLOADING_TIME
        state['location'] = order['polygon']
        state['inventory'].remove(order)
        state['load'] -= order['amount']
        state['delivered'].append(order)
        self.action_history.append(('deliver', order))

    def _return_to_hq(self):
        state = self.current_state
        travel_time = self.time_df.at[state['location'], HQ_POLYGON]
        state['time'] += travel_time
        state['location'] = HQ_POLYGON
        self.action_history.append(('return',))

    def run(self):
        while self.current_state['time'] < MAX_TIME:
            actions = self.get_actions(self.current_state)
            if not actions:
                break
            selected_action = self.protocol(actions, self.current_state)
            if not selected_action:
                break
            self.apply_action(selected_action)
        return self.current_state, self.action_history
    
# --- SASCOpt Class ---

class SASCOpt:
    """
    Optimizes supply chain scheduling using Simulated Annealing.
    """

    def __init__(
        self,
        inventory_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        max_load: int = 8000,
        transport_cost: float = 4500,
        iterations: int = 1000,
        initial_temp: float = 10000,
        cooling_rate: float = 0.995,
        log_file: str = './output/SA_Supply_Chain.log'
    ):
        self.inventory_df = inventory_df.copy()
        self.prices_df = prices_df.copy()
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
        """Aggregate demands by polygon and specie."""
        return self.inventory_df.groupby(['polygon', 'specie'])['demand'].sum().reset_index()

    def _setup_logging(self):
        """Set up logging to file."""
        if os.path.exists(self.log_file):
            logging.shutdown()
            os.remove(self.log_file)
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

    def initial_solution(self) -> List[Tuple[int, Any, Any, Any, float]]:
        """
        Generate an initial feasible solution.
        Returns:
            List of tuples: (day, polygon, specie, supplier, amount)
        """
        remaining = self.demands.copy()
        order_schedule = []
        warehouse_storage = []
        day = 0

        while remaining['demand'].sum() > 0:
            day += 1
            remaining_capacity = self.max_load
            fulfilled_indices = []
            day_plan = []

            # Shuffle demands for fairness
            for idx, row in remaining[remaining['demand'] > 0].sample(frac=1).iterrows():
                polygon, specie, demand = row['polygon'], row['specie'], row['demand']
                if demand > remaining_capacity:
                    continue

                valid_suppliers = self.prices_df[self.prices_df['specie'] == specie]['supplier'].unique()
                if len(valid_suppliers) == 0:
                    continue

                supplier = random.choice(valid_suppliers)
                day_plan.append((day, polygon, specie, supplier, demand))
                remaining_capacity -= demand
                fulfilled_indices.append(idx)

            order_schedule.extend(day_plan)
            remaining.loc[fulfilled_indices, 'demand'] = 0

            warehouse_storage.append(day_plan)



        return order_schedule

    def fitness(self, schedule: List[Tuple[int, Any, Any, Any, float]]) -> float:
        """
        Calculate the total cost of a schedule.
        """
        cost = 0
        horizon_days = max(entry[0] for entry in schedule)
        for day in range(1, horizon_days + 1):
            day_orders = [entry for entry in schedule if entry[0] == day]
            day_suppliers = set()
            for (_, polygon, specie, supplier, amount) in day_orders:
                price_row = self.prices_df[
                    (self.prices_df['specie'] == specie) &
                    (self.prices_df['supplier'] == supplier)
                ]
                unit_price = price_row.iloc[0]['price']
                cost += unit_price * amount
                day_suppliers.add(supplier)
            cost += self.transport_cost * len(day_suppliers)
        return cost

    def neighbor(self, schedule: List[Tuple[int, Any, Any, Any, float]]) -> List[Tuple[int, Any, Any, Any, float]]:
        """
        Generate a neighbor solution by changing the supplier for a random order.
        """
        new_schedule = schedule.copy()
        if len(new_schedule) < 2:
            return new_schedule

        idx = random.randint(0, len(new_schedule) - 1)
        day, polygon, specie, _, amount = new_schedule[idx]
        valid_suppliers = self.prices_df[self.prices_df['specie'] == specie]['supplier'].unique()
        new_supplier = random.choice(valid_suppliers)
        new_schedule[idx] = (day, polygon, specie, new_supplier, amount)
        return new_schedule

    def run(self) -> Tuple[List[Tuple[int, Any, Any, Any, float]], float]:
        """
        Run the Simulated Annealing optimization.
        Returns:
            Tuple of (best_solution, best_cost)
        """
        current_solution = self.initial_solution()
        current_cost = self.fitness(current_solution)
        best_solution = current_solution
        best_cost = current_cost
        temp = self.initial_temp

        for i in tqdm(range(self.iterations)):
            new_solution = self.neighbor(current_solution)
            new_cost = self.fitness(new_solution)

            accept = new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temp)
            if accept:
                current_solution = new_solution
                current_cost = new_cost
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost

            temp *= self.cooling_rate
            logging.info(f"Iteration {i+1}: Cost = {current_cost:.2f}, Temp = {temp:.2f}")

        logging.info(f"Best solution found with cost: {best_cost:.2f}")
        return best_solution, best_cost