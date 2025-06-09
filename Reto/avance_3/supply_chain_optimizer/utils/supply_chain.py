import numpy as np
import pandas as pd
import random
import os
import logging
from typing import List, Tuple, Any, Dict, Optional
from tqdm import tqdm
import json

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
        start_day: int = 1,
        initial_remaining: Optional[pd.DataFrame] = None
    ) -> Tuple[List[Tuple[int, Any, Any, Any, float]], List[Dict]]:
        """
        Generates a schedule from the given demands starting at `start_day`.
        If `initial_remaining` is given, uses that instead of recalculating from self.demands.
        """
        remaining = initial_remaining if initial_remaining is not None else demands_df.copy()
        order_schedule = []
        vrp_action_history = []
        day = start_day - 1

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
                self.warehouse.receive_deliveries(day, day_plan)

            available_inventory = self.warehouse.get_available_orders(day)
            self.vrp_optimizer.update_inventory(available_inventory)
            final_state, action_history = self.vrp_optimizer.run()
            vrp_action_history.append({"day": day, "actions": action_history})

            self.warehouse.update_after_delivery(final_state['delivered'], day)
            # print(self.warehouse.remaining_capacity)

        return order_schedule, vrp_action_history


    def initial_solution(self) -> Tuple[List[Tuple[int, Any, Any, Any, float]], List[Dict]]:
        return self.generate_schedule(demands_df=self.demands.copy(), start_day=1)


    def scheduling_cost(self, schedule: List[Tuple[int, Any, Any, Any, float]]) -> float:
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

    def fitness(self, schedule: List[Tuple[int, Any, Any, Any, float]]) -> float:
        cost = 0
        horizon_days = max(entry[0] for entry in schedule)
        utilization_penalty = 0
        overstay_penalty = 0
        max_warehouse_days = 7

        self.warehouse.reset()
        self.vrp_optimizer.reset()

        for day in range(1, horizon_days + 1):
            day_orders = [entry for entry in schedule if entry[0] == day]
            truck_load = 0
            day_suppliers = set()

            # Simulate receiving
            self.warehouse.receive_deliveries(day, day_orders)

            for (_, polygon, specie, supplier, amount) in day_orders:
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

            # Overstay penalty
            overstayed = self.warehouse.get_overstayed_items(day, max_warehouse_days)
            if overstayed:
                # total_overstayed = sum(float(item['amount']) for item in overstayed)
                # overstay_penalty += total_overstayed * 1_000_000_000  # or another large penalty scale
                overstay_penalty += 1_000_000_000

            # Simulate delivery
            available_inventory = self.warehouse.get_available_orders(day)
            self.vrp_optimizer.update_inventory(available_inventory)
            final_state, _ = self.vrp_optimizer.run()
            self.warehouse.update_after_delivery(final_state['delivered'], day)

        return cost + utilization_penalty + overstay_penalty


    def neighbor(self, schedule: List[Tuple[int, Any, Any, Any, float]]) -> List[Tuple[int, Any, Any, Any, float]]:
        max_attempts = 5
        for attempt in range(max_attempts):
            if not schedule:
                return schedule

            days = sorted(set(order[0] for order in schedule))
            if len(days) <= 1:
                return schedule

            cutoff_day = random.choice(days[:-1])

            preserved = [order for order in schedule if order[0] < cutoff_day]
            to_reschedule = [order for order in schedule if order[0] >= cutoff_day]

            remaining_demand = pd.DataFrame(to_reschedule, columns=['day', 'polygon', 'specie', 'supplier', 'amount']) \
                .groupby(['polygon', 'specie'])['amount'].sum().reset_index()
            remaining_demand.rename(columns={'amount': 'demand'}, inplace=True)

            self.warehouse.reset()
            self.vrp_optimizer.reset()

            regenerated, _ = self.generate_schedule(
                demands_df=remaining_demand,
                start_day=cutoff_day,
                initial_remaining=remaining_demand.copy()
            )

            trial_solution = preserved + regenerated

            # Check for overstay
            self.warehouse.reset()
            self.vrp_optimizer.reset()
            days_in_schedule = max(entry[0] for entry in trial_solution)
            feasible = True
            for day in range(1, days_in_schedule + 1):
                daily = [entry for entry in trial_solution if entry[0] == day]
                self.warehouse.receive_deliveries(day, daily)
                overstayed = self.warehouse.get_overstayed_items(day, max_days=7)
                if overstayed:
                    feasible = False
                    break

                available_inventory = self.warehouse.get_available_orders(day)
                self.vrp_optimizer.update_inventory(available_inventory)
                final_state, _ = self.vrp_optimizer.run()
                self.warehouse.update_after_delivery(final_state['delivered'], day)

            if feasible:
                return trial_solution

        # fallback: return current schedule (i.e., no change)
        return schedule


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


    def run(self) -> Tuple[List[Tuple[int, Any, Any, Any, float]], float, str, str]:
        order_schedule, vrp_actions = self.initial_solution()
        current_solution = order_schedule
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

        supply_chain_json = json.dumps([
            {"day": day, "polygon": polygon, "specie": specie, "supplier": supplier, "amount": amount}
            for day, polygon, specie, supplier, amount in best_solution
        ], indent=2, default=self.convert_to_serializable)

        vrp_actions_json = json.dumps(vrp_actions, indent=2, default=self.convert_to_serializable)

        return best_solution, best_cost, supply_chain_json, vrp_actions_json
