import numpy as np
import pandas as pd
from typing import List, Tuple, Any
from math import sqrt

# --- Utility Functions ---

def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_time_matrix(polygon_df: pd.DataFrame, speed_kmh: float = 10.0) -> pd.DataFrame:
    polygons = polygon_df['Poligono'].tolist()
    coords = polygon_df.set_index('Poligono')[['X', 'Y']].to_dict('index')
    time_data = [
        {
            'origin': i,
            'target': j,
            'time': np.ceil((euclidean_distance(list(coords[i].values()), list(coords[j].values())) / 1000 / speed_kmh) * 60)
        }
        for i in polygons for j in polygons
    ]
    time_df = pd.DataFrame(time_data)
    return time_df.pivot(index='origin', columns='target', values='time')

# --- VRP Class ---

class VRP:
    def __init__(
        self,
        polygon_df: pd.DataFrame,
        hq_polygon: Any,
        truck_capacity: int,
        loading_time: int,
        unloading_time: int,
        max_time: int,
        speed_kmh: float = 10.0
    ):
        self.inventory_df = None
        self.polygon_df = polygon_df
        self.hq_polygon = hq_polygon
        self.truck_capacity = truck_capacity
        self.loading_time = loading_time
        self.unloading_time = unloading_time
        self.max_time = max_time
        self.time_df = compute_time_matrix(polygon_df, speed_kmh)

    def reset(self):
        self.current_state = {
            'location': self.hq_polygon,
            'time': 0,
            'load': 0,
            'inventory': [],
            'delivered': []
        }
        self.action_history = []
        self.remaining = self._get_initial_remaining()

    def update_inventory(self, new_inventory_df: pd.DataFrame):
        """
        Updates the internal inventory and resets the delivery plan.
        """
        self.inventory_df = new_inventory_df.copy()
        self.reset()

    def _get_initial_remaining(self) -> pd.DataFrame:
        return self.inventory_df.groupby('polygon').agg({'amount': 'sum'}).reset_index()

    def get_state(self) -> dict:
        return self.current_state.copy()

    def get_actions(self, state: dict) -> List[dict]:
        actions = []
        # Load at HQ
        if (
            state['location'] == self.hq_polygon and
            state['time'] + self.loading_time <= self.max_time and
            not state['inventory']
        ):
            actions.append({'type': 'load'})
        # Deliveries
        for order in state['inventory']:
            polygon = order['polygon']
            travel_time = self.time_df.at[state['location'], polygon]
            total_time = travel_time + self.unloading_time + self.time_df.at[polygon, self.hq_polygon]
            if state['location'] != polygon and state['time'] + total_time <= self.max_time:
                actions.append({'type': 'deliver', 'order': order})
            elif (
                state['location'] == polygon and
                state['time'] + self.unloading_time + self.time_df.at[polygon, self.hq_polygon] <= self.max_time
            ):
                actions.append({'type': 'deliver', 'order': order})
        # Return to HQ
        if state['location'] != self.hq_polygon:
            travel_time = self.time_df.at[state['location'], self.hq_polygon]
            if state['time'] + travel_time <= self.max_time:
                actions.append({'type': 'return'})
        return actions

    def protocol(self, actions: List[dict], state: dict) -> dict:
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

    def apply_action(self, action: dict):
        if action['type'] == 'load':
            self._load_truck()
        elif action['type'] == 'deliver':
            self._deliver_order(action['order'])
        elif action['type'] == 'return':
            self._return_to_hq()

    def _load_truck(self):
        capacity = self.truck_capacity
        new_inventory = []
        updated_remaining = []

        for _, row in self.remaining.iterrows():
            polygon = row['polygon']
            demand = row['amount']

            if demand <= capacity:
                # Full load possible
                new_inventory.append({'polygon': polygon, 'amount': demand})
                capacity -= demand
            elif capacity > 0:
                # Partial load
                new_inventory.append({'polygon': polygon, 'amount': capacity})
                updated_remaining.append({'polygon': polygon, 'amount': demand - capacity})
                capacity = 0
                break  # Truck is full after this partial load
            else:
                # No room at all
                updated_remaining.append({'polygon': polygon, 'amount': demand})

        self.current_state['inventory'] = new_inventory
        self.remaining = pd.DataFrame(updated_remaining)
        self.current_state['load'] = self.truck_capacity - capacity
        self.current_state['time'] += self.loading_time
        self.action_history.append(('load', len(new_inventory)))


    def _deliver_order(self, order: dict):
        state = self.current_state
        travel_time = self.time_df.at[state['location'], order['polygon']]
        state['time'] += travel_time + self.unloading_time
        state['location'] = order['polygon']
        state['inventory'].remove(order)
        state['load'] -= order['amount']
        state['delivered'].append(order)
        self.action_history.append(('deliver', order))

    def _return_to_hq(self):
        state = self.current_state
        travel_time = self.time_df.at[state['location'], self.hq_polygon]
        state['time'] += travel_time
        state['location'] = self.hq_polygon
        self.action_history.append(('return',))

    def run(self) -> Tuple[dict, list]:
        while self.current_state['time'] < self.max_time:
            actions = self.get_actions(self.current_state)
            if not actions:
                break
            selected_action = self.protocol(actions, self.current_state)
            if not selected_action:
                break
            self.apply_action(selected_action)
        return self.current_state, self.action_history