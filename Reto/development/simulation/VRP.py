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

def load_demand(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['best_solution'], columns=['day', 'polygon', 'specie', 'supplier', 'amount'])
    return df

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
    def __init__(self, demand_df, polygon_df):
        self.demand_df = demand_df.copy()
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
        demand_day = self.demand_df[self.demand_df['day'] == 1]
        return demand_day.groupby('polygon').agg({'amount': 'sum'}).reset_index()

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