import numpy as np

class Supply_Chain_Agent:
    def __init__(self, species, providers, planning_days, truck_capacity, warehouse_capacity, species_area, purchase_cost, available_species):
        self.species = species
        self.providers = providers
        self.planning_days = planning_days
        self.truck_capacity = truck_capacity
        self.warehouse_capacity = warehouse_capacity
        self.species_area = species_area
        self.purchase_cost = purchase_cost
        self.available_species = available_species

        # Inventories: inve[species][day][age]
        self.inventory = {e: {t: [0]*7 for t in planning_days} for e in species}
        self.purchase_plan = []

    def _can_store(self, species, quantity, day):
        total_area = sum(self.species_area[e] * sum(self.inventory[e][day]) for e in self.species)
        new_area = self.species_area[species] * quantity
        return total_area + new_area <= self.warehouse_capacity

    def greedy_schedule(self, demand):
        # demand[(species, zone)] = quantity
        for (e, z), q in demand.items():
            delivered = 0
            for t in self.planning_days:
                for j in self.providers:
                    if self.available_species[e][j] == 0:
                        continue
                    max_can_buy = min(q - delivered, self.truck_capacity)
                    if max_can_buy <= 0:
                        break
                    if not self._can_store(e, max_can_buy, t):
                        continue
                    self.purchase_plan.append((e, j, t, max_can_buy))
                    self.inventory[e][t][0] += max_can_buy
                    delivered += max_can_buy
                    break  # Go to next day
                if delivered >= q:
                    break

    def get_inventory(self):
        return self.inventory

    def get_purchase_plan(self):
        return self.purchase_plan
