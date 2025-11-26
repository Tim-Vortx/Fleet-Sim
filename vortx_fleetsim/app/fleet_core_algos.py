import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class VehicleSpecs:
    model: str
    category: str
    battery_capacities: List[float]
    ranges: List[float]
    battery_capacity: float = field(default=0.0)
    range: float = field(default=0.0)
    vehicle_class: List[str] = field(default_factory=list)
    vehicle_types: List[str] = field(default_factory=list)
    efficiency: float = field(default=0.0)

    def __post_init__(self):
        if self.battery_capacity not in self.battery_capacities:
            raise ValueError(f"Selected battery capacity {self.battery_capacity} not in available capacities {self.battery_capacities}")
        if not self.ranges:
            raise ValueError("Vehicle must have at least one range value")
        if self.efficiency <= 0:
            self.efficiency = self.battery_capacity / self.range if self.range else 2.0

class FleetManagementSystem:
    def __init__(self):
        self.vehicle_catalog = {}
        self.fleet_composition = {}
        self.schedule = pd.DataFrame()

    def load_hvip_data(self, csv_path):
        """
        Load vehicle data from a CSV file into the vehicle catalog.
        
        Args:
            csv_path (str): Path to the CSV file containing vehicle data
        """
        try:
            logger.info(f"Loading vehicle data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            def extract_numbers(text):
                if pd.isna(text):
                    return []
                numbers = []
                # Split by <br> tag and process each part
                parts = str(text).replace('Up to ', '').replace('<br>', ',').split(',')
                for part in parts:
                    try:
                        # Remove 'kWh' and any other non-numeric characters except decimal points
                        cleaned = ''.join(c for c in part if c.isdigit() or c == '.')
                        if cleaned:
                            numbers.append(float(cleaned))
                    except (ValueError, IndexError):
                        continue
                return numbers

            vehicle_count = 0  # Counter for vehicles added
            for _, row in df.iterrows():
                try:
                    # Extract battery capacities and ranges
                    batteries = extract_numbers(row['Battery'])
                    ranges = extract_numbers(row['Range'])
                    
                    if not batteries or not ranges:
                        continue
                    
                    model_name = str(row['Model']).strip()
                    
                    # Make sure we have matching number of ranges for batteries
                    while len(ranges) < len(batteries):
                        ranges.extend(ranges[:len(batteries)-len(ranges)])
                    while len(batteries) < len(ranges):
                        batteries.extend(batteries[:len(ranges)-len(batteries)])
                    
                    # Process each battery capacity and range combination
                    for idx in range(len(batteries)):
                        battery_value = batteries[idx]
                        range_value = ranges[idx]
                        
                        if battery_value <= 0 or range_value <= 0:
                            continue
                        
                        # Create variant key using model name and battery capacity
                        base_key = model_name.lower().replace(' ', '_')
                        variant_key = f"{base_key}_{int(battery_value)}"
                        
                        # Process vehicle types and GVWR
                        vehicle_types = [t.strip() for t in str(row['Vehicle Types']).split(';') if t.strip()]
                        gvwr_classes = [g.strip() for g in str(row['GVWR']).split(';') if g.strip()]
                        
                        # Calculate efficiency
                        efficiency = battery_value / range_value
                        
                        # Create VehicleSpecs object
                        vehicle_specs = VehicleSpecs(
                            model=model_name,
                            category=str(row['Category']).strip(),
                            battery_capacities=batteries,
                            ranges=ranges,
                            battery_capacity=battery_value,
                            range=range_value,
                            vehicle_class=gvwr_classes,
                            vehicle_types=vehicle_types,
                            efficiency=efficiency
                        )
                        
                        # Add to vehicle catalog
                        self.vehicle_catalog[variant_key] = vehicle_specs
                        vehicle_count += 1  # Increment the counter
                        logger.debug(f"Added vehicle: {variant_key}")
                
                except Exception as row_error:
                    logger.error(f"Error processing row for model {row.get('Model', 'Unknown')}: {str(row_error)}")
                    continue
            
            logger.info(f"Successfully loaded {vehicle_count} vehicles from {csv_path}")
        
        except Exception as e:
            logger.error(f"Error loading vehicle data from {csv_path}: {str(e)}")
            raise

    def export_for_charging_optimization(self) -> pd.DataFrame:
        """Export fleet data for charging optimization."""
        if len(self.schedule) == 0:
            # If no schedule, generate a default fleet
            fleet_data = []
            for i in range(5):  # Generate 5 default vehicles
                vehicle_schedule = self.generate_random_schedule(
                    base_departure="08:00", 
                    base_return="18:00", 
                    route_miles_range=(50, 200)
                )
                vehicle_schedule['vehicle_id'] = f'Vehicle_{i+1}'
                fleet_data.append(vehicle_schedule)
            
            self.schedule = pd.DataFrame(fleet_data)
            self.schedule['battery_capacity'] = 452  # Default battery capacity
            self.schedule['energy_needed_kwh'] = self.schedule['route_miles'] * 1.65
            self.schedule['remaining_soc'] = 80.0

        # Ensure all required columns exist
        required_columns = ['Vehicle_ID', 'return_time', 'route_miles']
        for col in required_columns:
            if col not in self.schedule.columns:
                if col == 'Vehicle_ID':
                    self.schedule['Vehicle_ID'] = [f'Vehicle_{i+1}' for i in range(len(self.schedule))]
                elif col == 'return_time':
                    self.schedule['return_time'] = '18:00'
                elif col == 'route_miles':
                    self.schedule['route_miles'] = 100.0

        optimizer_data = pd.DataFrame({
            'Vehicle_ID': self.schedule['Vehicle_ID'],
            'Return_Time': pd.to_datetime(self.schedule['return_time'].apply(
                lambda x: f"2023-01-01 {x}")),
            'Departure_Time': pd.to_datetime(self.schedule['return_time'].apply(
                lambda x: f"2023-01-01 {x}")),
            'Energy_Needed_kWh': self.schedule.get('energy_needed_kwh', 
                                                   self.schedule['route_miles'] * 1.65),
            'Battery_Capacity_kWh': self.schedule.get('battery_capacity', 452),
            'Initial_SOC': self.schedule.get('remaining_soc', 80.0),
            'Target_SOC': 80.0
        })
        
        # Handle overnight schedules
        mask = optimizer_data['Departure_Time'] <= optimizer_data['Return_Time']
        optimizer_data.loc[mask, 'Departure_Time'] += pd.Timedelta(days=1)
        
        return optimizer_data

    def generate_random_schedule(self, base_departure, base_return, route_miles_range):
        """
        Generate a random vehicle schedule for default fleet generation.
        
        Args:
            base_departure (str): Base departure time
            base_return (str): Base return time
            route_miles_range (tuple): Range of route miles
        
        Returns:
            dict: Generated vehicle schedule
        """
        import random
        from datetime import datetime, timedelta
        
        route_miles = random.uniform(route_miles_range[0], route_miles_range[1])
        
        # Generate times with some variability
        base_date = datetime.now().strftime('%Y-%m-%d')
        departure_time = datetime.strptime(f"{base_date} {base_departure}", '%Y-%m-%d %H:%M')
        return_time = datetime.strptime(f"{base_date} {base_return}", '%Y-%m-%d %H:%M')
        
        # Add some random time variability
        departure_time += timedelta(minutes=random.randint(-60, 60))
        return_time += timedelta(minutes=random.randint(-60, 60))
        
        # Ensure return time is after departure time
        if return_time <= departure_time:
            return_time += timedelta(days=1)
        
        return {
            'route_miles': route_miles,
            'return_time': return_time.strftime('%H:%M'),
            'departure_time': departure_time.strftime('%H:%M')
        }
