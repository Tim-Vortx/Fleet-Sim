# vortx_fleetsim/app/core/fleet_manager.py
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

from ..models.vehicle import VehicleSpecs, VehicleState
from ..models.route import RouteInfo, Schedule

logger = logging.getLogger(__name__)

class FleetManager:
    def __init__(self):
        """Initialize the fleet manager with your existing functionality"""
        self.vehicle_catalog: Dict[str, VehicleSpecs] = {}
        self.fleet_composition: Dict[str, VehicleState] = {}
        self.active_routes: Dict[str, RouteInfo] = {}
        self.schedule = pd.DataFrame()

    def load_hvip_data(self, csv_path: str) -> None:
        """Load vehicle data from CSV - from your existing fleet_core_algos.py"""
        try:
            logger.info(f"Loading vehicle data from {csv_path}")
            df = pd.read_csv(csv_path)
            
            def extract_numbers(text):
                if pd.isna(text):
                    return []
                numbers = []
                parts = str(text).replace('Up to ', '').replace('<br>', ',').split(',')
                for part in parts:
                    try:
                        cleaned = ''.join(c for c in part if c.isdigit() or c == '.')
                        if cleaned:
                            numbers.append(float(cleaned))
                    except (ValueError, IndexError):
                        continue
                return numbers

            vehicle_count = 0
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
                        
                        variant_key = f"{model_name.lower().replace(' ', '_')}_{int(battery_value)}"
                        
                        # Create VehicleSpecs object
                        vehicle_specs = VehicleSpecs(
                            model=model_name,
                            category=str(row['Category']).strip(),
                            battery_capacities=batteries,
                            ranges=ranges,
                            battery_capacity=battery_value,
                            range=range_value,
                            vehicle_class=[g.strip() for g in str(row['GVWR']).split(';') if g.strip()],
                            vehicle_types=[t.strip() for t in str(row['Vehicle Types']).split(';') if t.strip()],
                            efficiency=battery_value / range_value
                        )
                        
                        self.vehicle_catalog[variant_key] = vehicle_specs
                        vehicle_count += 1
                        logger.debug(f"Added vehicle: {variant_key}")
                
                except Exception as row_error:
                    logger.error(f"Error processing row for model {row.get('Model', 'Unknown')}: {str(row_error)}")
                    continue
            
            logger.info(f"Successfully loaded {vehicle_count} vehicles")
        
        except Exception as e:
            logger.error(f"Error loading vehicle data: {str(e)}")
            raise

    def update_vehicle_state(self, 
                           vehicle_id: str, 
                           status: Optional[str] = None,
                           location: Optional[Tuple[float, float]] = None,
                           current_soc: Optional[float] = None,
                           **kwargs) -> VehicleState:
        """Update the state of a vehicle"""
        if vehicle_id not in self.fleet_composition:
            raise ValueError(f"Vehicle {vehicle_id} not found in fleet")

        vehicle = self.fleet_composition[vehicle_id]
        
        if status:
            vehicle.status = status
        if location:
            vehicle.location = location
        if current_soc is not None:
            vehicle.current_soc = current_soc
            
        vehicle.last_updated = datetime.now()
        
        return vehicle

    def export_for_charging_optimization(self) -> pd.DataFrame:
        """Your existing export logic for charging optimization"""
        if len(self.schedule) == 0:
            fleet_data = []
            for i in range(5):
                vehicle_schedule = self._generate_random_schedule(
                    base_departure="08:00", 
                    base_return="18:00", 
                    route_miles_range=(50, 200)
                )
                vehicle_schedule['vehicle_id'] = f'Vehicle_{i+1}'
                fleet_data.append(vehicle_schedule)
            
            self.schedule = pd.DataFrame(fleet_data)
            self.schedule['battery_capacity'] = 452
            self.schedule['energy_needed_kwh'] = self.schedule['route_miles'] * 1.65
            self.schedule['remaining_soc'] = 80.0

        # Rest of your existing export logic
        return self.schedule

    # Add your other existing methods here...