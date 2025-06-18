import numpy as np
import pandas as pd
import random
import os
import logging
from typing import List, Tuple, Any, Dict, Optional
from tqdm import tqdm
import json
import copy

from utils.vrp import VRP
from utils.warehouse import Warehouse

class SASCOpt:
    """
    Optimizes supply chain scheduling using Simulated Annealing.
    Modularized for flexible parameterization.
    """
    def __init__(
        self,
        demand_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        polygon_df: pd.DataFrame,
        max_load: int,
        transport_cost: float,
        iterations: int,
        initial_temp: float,
        cooling_rate: float,
        log_file: str,
        warehouse : Warehouse,
        vrp_optimizer : VRP,
        utilization_threshold: float = 0.75,
        utilization_penalty_scale: float = 1_000_000
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
        self.utilization_threshold = utilization_threshold
        self.utilization_penalty_scale = utilization_penalty_scale
        self.warehouse = warehouse
        self.vrp_optimizer = vrp_optimizer

        self.demands = self._aggregate_demands()
        self.species = self.demands['specie'].unique()
        self.suppliers = self.prices_df['supplier'].unique()

        self._setup_logging()

    def _aggregate_demands(self) -> pd.DataFrame:
        return self.demand_df.groupby(['polygon', 'specie'])['demand'].sum().reset_index()

    def _setup_logging(self):
        if os.path.exists(self.log_file):
            logging.shutdown()
            os.remove(self.log_file)
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

    def generate_schedule(
        self,
        demands_df: pd.DataFrame,
        initial_remaining: Optional[pd.DataFrame] = None
    ) -> Tuple[List[Tuple[int, Any, Any, Any, float]], List[Dict]]:
        """
        Generates a schedule from the given demands starting at start_day.
        If initial_remaining is given, uses that instead of recalculating from self.demands.
        """
        remaining = initial_remaining if initial_remaining is not None else demands_df.copy()
        order_schedule = []
        warehouse_schedule = []
        vrp_action_history = []
        day = 0

        while (remaining['demand'].sum() > 0) or not self.warehouse.is_empty():
            day += 1
            truck_remaining_capacity = self.max_load
            warehouse_dayload_capacity = self.warehouse.remaining_capacity
            fulfilled_indices = []
            day_plan = []

            for idx, row in remaining[remaining['demand'] > 0].sample(frac=1).iterrows():
                polygon, specie, demand = row['polygon'], row['specie'], row['demand']
                max_possible = min(truck_remaining_capacity, warehouse_dayload_capacity, demand)
                if max_possible <= 0:
                    continue

                valid_suppliers = self.prices_df[self.prices_df['specie'] == specie]['supplier'].unique()
                if len(valid_suppliers) == 0:
                    continue

                supplier = random.choice(valid_suppliers)
                day_plan.append((day, polygon, specie, supplier, max_possible))
                truck_remaining_capacity -= max_possible
                warehouse_dayload_capacity -= max_possible

                remaining.at[idx, 'demand'] -= max_possible
                if remaining.at[idx, 'demand'] <= 0:
                    fulfilled_indices.append(idx)

            if day_plan:
                order_schedule.extend(day_plan)
                remaining.loc[fulfilled_indices, 'demand'] = 0
                self.warehouse.receive_deliveries(
                    day = day, 
                    deliveries = day_plan
                )

            warehouse_schedule.append({"day": day, "actions": copy.deepcopy(self.warehouse.daily_inventory)})

            available_inventory = self.warehouse.get_available_orders(day)

            self.vrp_optimizer.update_inventory(available_inventory)
            final_state, action_history = self.vrp_optimizer.run()
            vrp_action_history.append({"day": day, "actions": action_history})

            self.warehouse.update_after_delivery(
                delivered_orders = final_state['delivered'], 
                day = day
            )

            # print(warehouse_schedule)

        return order_schedule, warehouse_schedule, vrp_action_history

    def scheduling_cost(self, order_schedule: List[Tuple[int, Any, Any, Any, float]]) -> float:
        cost = 0
        horizon_days = max(entry[0] for entry in order_schedule)
        for day in range(1, horizon_days + 1):
            day_orders = [entry for entry in order_schedule if entry[0] == day]
            day_suppliers = set()

            for (_, _, specie, supplier, amount) in day_orders:
                price_row = self.prices_df[
                    (self.prices_df['specie'] == specie) &
                    (self.prices_df['supplier'] == supplier)
                ]
                unit_price = price_row.iloc[0]['price']
                cost += unit_price * amount
                day_suppliers.add(supplier)

            cost += self.transport_cost * len(day_suppliers)

        return cost
    
    def fitness(
        self, 
        schedule: List[Tuple[int, Any, Any, Any, float]],
    ) -> float:
        cost = 0
        horizon_days = max(entry[0] for entry in schedule)
        utilization_penalty = 0

        for day in range(1, horizon_days + 1):
            day_orders = [entry for entry in schedule if entry[0] == day]
            truck_load = 0
            day_suppliers = set()

            for (_, _, specie, supplier, amount) in day_orders:
                price_row = self.prices_df[
                    (self.prices_df['specie'] == specie) & 
                    (self.prices_df['supplier'] == supplier)
                ]
                unit_price = price_row.iloc[0]['price']
                cost += unit_price * amount
                truck_load += amount
                day_suppliers.add(supplier)

            cost += self.transport_cost * len(day_suppliers)

            # Utilization penalty
            utilization = truck_load / self.max_load
            if utilization < self.utilization_threshold:
                utilization_penalty += (1 - utilization) * self.utilization_penalty_scale

        return cost + utilization_penalty


    def get_supply_chain_utilization(
        self, 
        schedule: List[Tuple[int, Any, Any, Any, float]],
    ):
        horizon_days = max(entry[0] for entry in schedule)
        utilization_rates = []

        for day in range(1, horizon_days + 1):
            day_orders = [entry for entry in schedule if entry[0] == day]
            truck_load = 0

            for (_, _, specie, supplier, amount) in day_orders:
                truck_load += amount
            
            # Utilization penalty
            utilization = truck_load / self.max_load
            utilization_rates.append(utilization)

        return utilization_rates
    
    def get_warehouse_utilization(
        self,
        schedule
    ) -> float:
        
        utilization = []
        capacity = self.warehouse.max_capacity

        for elem in schedule:
            amt = 0
            actions = elem['actions']
            for _, inventory in actions.items():
                for item in inventory:
                    amt += item['amount']

            utilization.append(amt / capacity)

        return utilization

    def get_vrp_utilizations(
        self,
        schedule
    ) -> float:
        
        utilization = []

        for elem in schedule:
            amt_delivered = 0
            actions = elem['actions']
            for action in actions:
                if action[0] == 'deliver':
                    amt_delivered += action[1]['amount']

            utilization.append(amt_delivered)

        return utilization          

    @staticmethod
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def run(self) -> Dict[str, Any]:
    # Solución inicial
        init_order_schedule, init_warehouse_schedule, init_vrp_schedule = self.generate_schedule(
            demands_df=self.demand_df,
        )
        # print(init_warehouse_schedule)
        current_fitness = self.fitness(init_order_schedule)

        best_order_schedule = init_order_schedule
        best_warehouse_schedule = init_warehouse_schedule
        best_vrp_schedule = init_vrp_schedule
        best_fitness = current_fitness

        temp = self.initial_temp

        for i in tqdm(range(self.iterations)):

            new_order_schedule, new_warehouse_schedule, new_vrp_schedule = self.generate_schedule(
                demands_df=self.demand_df,
            )

            new_fitness = self.fitness(new_order_schedule)

            c1 = (new_fitness < best_fitness)
            c2 = (np.log(random.random()) < (best_fitness - new_fitness) / temp)
            # print((best_fitness - new_fitness) / temp)
            accept_solution = c1 or c2
            
            if accept_solution:
                best_order_schedule = new_order_schedule
                best_warehouse_schedule = new_warehouse_schedule
                best_vrp_schedule = new_vrp_schedule
                best_fitness = new_fitness

            temp *= self.cooling_rate
            logging.info(f"Iteration {i+1}: Fitness = {new_fitness:.2f}, Temp = {temp:.2f}")

        best_cost = self.scheduling_cost(best_order_schedule)
        best_supply_chain_utilization = self.get_supply_chain_utilization(best_order_schedule)
        best_warehouse_utilization = self.get_warehouse_utilization(best_warehouse_schedule)
        best_vrp_utilization = self.get_vrp_utilizations(best_vrp_schedule)
        
        logging.info(f"Best solution found with cost: {best_cost:.2f}")

        with open("./scheduling/order_schedule.json", "w") as f:
            json.dump(best_order_schedule, f, indent=2, default=SASCOpt.convert_to_serializable)

        with open("./scheduling/warehouse_schedule.json", "w") as f:
            json.dump(best_warehouse_schedule, f, indent=2, default=SASCOpt.convert_to_serializable)

        with open("./scheduling/vrp_schedule.json", "w") as f:
            json.dump(best_vrp_schedule, f, indent=2, default=SASCOpt.convert_to_serializable)

        return {
            "order_schedule": best_order_schedule,
            "warehouse_schedule": best_warehouse_schedule,
            "vrp_schedule": best_vrp_schedule,
            "fitness_cost": best_fitness,
            "real_cost": best_cost,
            'supply_chain_utilization': best_supply_chain_utilization,
            'warehouse_utilization': best_warehouse_utilization,
            'vrp_utilization': best_vrp_utilization
        }
