import pandas as pd
from typing import List, Tuple, Dict
import copy

class Warehouse:
    """
    Central warehouse that stores inventory by species and day.
    """

    def __init__(
        self,
        max_capacity: int,
        min_storage_days: int = 3
    ):
        """
        :param capacity_multiplier: Multiplier for max warehouse capacity.
        :param min_storage_days: Minimum days before inventory is eligible for orders.
        """
        self.daily_inventory = {}  # {day: [{'polygon': x, 'specie': y, 'amount': z}]}
        self.max_capacity = max_capacity
        self.remaining_capacity = self.max_capacity
        self.min_storage_days = min_storage_days

    def receive_deliveries(
        self,
        day: int,
        deliveries: List[Tuple[int, int, str, str, float]]
    ):
        if day not in self.daily_inventory:
            self.daily_inventory[day] = []
        total_received = 0
        for _, polygon, specie, _, amount in deliveries:
            self.daily_inventory[day].append({'polygon': polygon, 'specie': specie, 'amount': amount})
            total_received += amount
        self.remaining_capacity -= total_received

    def get_available_orders(self, day: int) -> pd.DataFrame:
        eligible_days = [d for d in self.daily_inventory if d <= day - self.min_storage_days]
        if not eligible_days:
            return pd.DataFrame(columns=['day', 'polygon', 'specie', 'amount'])
        records = []
        for d in eligible_days:
            for entry in self.daily_inventory[d]:
                record = entry.copy()
                record['day'] = d
                records.append(record)
        if not records:
            return pd.DataFrame(columns=['day', 'polygon', 'specie', 'amount'])
        df = pd.DataFrame(records)
        return df[['day', 'polygon', 'specie', 'amount']]

    def update_after_delivery(self, delivered_orders: List[dict], day: int):
        eligible_days = [d for d in self.daily_inventory if d <= day - self.min_storage_days]
        total_removed = 0
        for delivered in delivered_orders:
            amount_to_remove = delivered['amount']
            for d in eligible_days:
                current = self.daily_inventory[d]
                for i in range(len(current)):
                    entry = current[i]
                    if entry['polygon'] == delivered['polygon'] and entry['specie'] == delivered.get('specie', entry['specie']):
                        stored_amount = entry['amount']
                        if stored_amount <= amount_to_remove:
                            amount_to_remove -= stored_amount
                            total_removed += stored_amount
                            current.pop(i)
                            break
                        else:
                            entry['amount'] -= amount_to_remove
                            total_removed += amount_to_remove
                            amount_to_remove = 0
                            break
                if amount_to_remove == 0:
                    break
        self.remaining_capacity = min(self.remaining_capacity + total_removed, self.max_capacity)

    def get_overstayed_items(self, current_day: int, max_days: int = 7):
        overstayed = []
        for day_key, deliveries in self.daily_inventory.items():
            if current_day - day_key > max_days:
                overstayed.extend(deliveries)
        return overstayed
    
    def is_empty(self):
        return self.max_capacity == self.remaining_capacity
    
    def reset(self):
        self.current_inventory = []
        self.remaining_capacity = self.max_capacity
        self.history = {}