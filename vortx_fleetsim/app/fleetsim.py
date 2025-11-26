from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Optional, Literal
import json
from dataclasses import asdict, dataclass
import pandas as pd

from fleet_core_algos import FleetManagementSystem
from v11_opt_demand import ChargingOptimizerDemand
from v11_opt_tou import ChargingOptimizerTOU
from v11_util import load_bev_energy_prices, round_to_interval, INTERVAL_MINUTES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OptimizerType = Literal["demand", "tou"]

@dataclass
class SimulationConfig:
    fleet_size: int
    num_chargers: int
    charger_power: float
    optimizer_type: OptimizerType = "demand"
    demand_limit: float = 1000.0
    simulation_interval: int = INTERVAL_MINUTES
    simulation_hours: int = 48
    energy_prices: Optional[Dict] = None

@dataclass
class VehicleState:
    vehicle_id: str
    status: str  # 'driving', 'charging', 'idle'
    location: tuple[float, float]  # lat, lng
    current_soc: float
    last_updated: datetime
    route_id: Optional[str] = None
    charger_id: Optional[str] = None
    energy_delivered: float = 0.0
    energy_consumed: float = 0.0
    battery_capacity: float = 0.0

@dataclass
class ChargerState:
    charger_id: str
    status: str  # 'available', 'charging', 'maintenance'
    vehicle_id: Optional[str] = None
    power_level: float = 0.0
    energy_delivered: float = 0.0
    last_updated: datetime = None

class ContinuousFleetSimulator:
    def __init__(
        self,
        config: SimulationConfig,
        data_lake_connection = None
    ):
        # Store configuration
        self.config = config
        
        # Core components
        self.fleet_system = FleetManagementSystem()
        self.charging_optimizer = self._create_optimizer(config.optimizer_type)
        
        # State tracking
        self.vehicles: Dict[str, VehicleState] = {}
        self.chargers: Dict[str, ChargerState] = {}
        self.current_time: datetime = None
        self.data_lake = data_lake_connection
        
        # Performance metrics
        self.total_energy_delivered = 0.0
        self.peak_power_demand = 0.0
        self.daily_stats = {}
        
        # Initialize fleet and chargers
        self._initialize_fleet(config.fleet_size)
        self._initialize_chargers(config.num_chargers, config.charger_power)

    def _create_optimizer(self, optimizer_type: OptimizerType):
        """Create the specified charging optimizer."""
        optimizer_params = {
            'num_chargers': self.config.num_chargers,
            'charger_power': self.config.charger_power,
            'demand_limit': self.config.demand_limit,
            'interval_minutes': self.config.simulation_interval,
            'simulation_hours': self.config.simulation_hours,
            'energy_prices': self.config.energy_prices
        }
        
        if optimizer_type == "demand":
            return ChargingOptimizerDemand(**optimizer_params)
        elif optimizer_type == "tou":
            return ChargingOptimizerTOU(**optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    async def change_optimizer(self, new_type: OptimizerType):
        """Change the optimization strategy during runtime."""
        try:
            if new_type == self.config.optimizer_type:
                return
            
            # Create new optimizer
            new_optimizer = self._create_optimizer(new_type)
            
            # Initialize new optimizer with current time
            if self.current_time:
                new_optimizer.initialize_simulation(self.current_time)
            
            # Transfer active charging sessions
            for vehicle_id, session in self.charging_optimizer.active_sessions.items():
                new_optimizer.active_sessions[vehicle_id] = session
            
            # Transfer waiting queue
            new_optimizer.waiting_queue = self.charging_optimizer.waiting_queue.copy()
            
            # Update optimizer and config
            self.charging_optimizer = new_optimizer
            self.config.optimizer_type = new_type
            
            # Log change
            await self._push_to_data_lake('optimizer_changed', {
                'old_type': self.config.optimizer_type,
                'new_type': new_type,
                'timestamp': self.current_time
            })
            
            logger.info(f"Changed optimizer from {self.config.optimizer_type} to {new_type}")
            
        except Exception as e:
            logger.error(f"Error changing optimizer: {str(e)}")
            raise

    def _initialize_fleet(self, fleet_size: int):
        """Initialize fleet with basic vehicle states."""
        for i in range(fleet_size):
            vehicle_id = f"VEH_{i:03d}"
            self.vehicles[vehicle_id] = VehicleState(
                vehicle_id=vehicle_id,
                status='idle',
                location=(34.0522, -118.2437),  # Example: LA coordinates
                current_soc=80.0,
                last_updated=datetime.now(),
                battery_capacity=150.0  # Example capacity, should be configurable
            )
            
    def _initialize_chargers(self, num_chargers: int, charger_power: float):
        """Initialize charging infrastructure."""
        for i in range(num_chargers):
            charger_id = f"CHG_{i:03d}"
            self.chargers[charger_id] = ChargerState(
                charger_id=charger_id,
                status='available',
                last_updated=datetime.now()
            )

    async def _simulate_vehicle_operations(self, vehicle_id: str):
        """Simulate individual vehicle operations."""
        vehicle = self.vehicles[vehicle_id]
        
        try:
            if vehicle.status == 'driving':
                # Simulate energy consumption
                energy_consumption = self._calculate_route_energy(vehicle)
                vehicle.energy_consumed += energy_consumption
                vehicle.current_soc = max(0, vehicle.current_soc - (energy_consumption / vehicle.battery_capacity * 100))
                
                if self._is_route_complete(vehicle):
                    vehicle.status = 'idle'
                    await self._push_to_data_lake('vehicle_route_completed', {
                        'vehicle_id': vehicle_id,
                        'route_id': vehicle.route_id,
                        'energy_consumed': vehicle.energy_consumed,
                        'end_soc': vehicle.current_soc,
                        'timestamp': self.current_time
                    })
                    
            elif vehicle.status == 'charging':
                charging_session = self.charging_optimizer.active_sessions.get(vehicle_id)
                if charging_session:
                    energy_delivered = charging_session.power_profile[-1] * (self.config.simulation_interval / 60)
                    vehicle.energy_delivered += energy_delivered
                    vehicle.current_soc = charging_session.current_soc
                    
                    await self._push_to_data_lake('charging_session_update', {
                        'vehicle_id': vehicle_id,
                        'charger_id': vehicle.charger_id,
                        'energy_delivered': energy_delivered,
                        'current_soc': vehicle.current_soc,
                        'timestamp': self.current_time,
                        'optimizer_type': self.config.optimizer_type
                    })
                    
            elif vehicle.status == 'idle':
                if vehicle.current_soc < 80.0:
                    await self._request_charging(vehicle)
                elif self._check_route_availability(vehicle):
                    await self._assign_route(vehicle)
                    
        except Exception as e:
            logger.error(f"Error simulating vehicle {vehicle_id}: {str(e)}")
            
        finally:
            vehicle.last_updated = self.current_time

    async def _request_charging(self, vehicle: VehicleState):
        """Request charging for a vehicle."""
        try:
            # Create charging request
            request = {
                'vehicle_id': vehicle.vehicle_id,
                'current_soc': vehicle.current_soc,
                'target_soc': 95.0,
                'arrival_time': self.current_time,
                'departure_time': self.current_time + timedelta(hours=8),
                'battery_capacity': vehicle.battery_capacity
            }
            
            # Add to charging optimizer
            self.charging_optimizer.add_charging_request(request, self.current_time)
            
            # Update vehicle state
            vehicle.status = 'charging'
            await self._push_to_data_lake('charging_request', {
                **request,
                'optimizer_type': self.config.optimizer_type
            })
            
        except Exception as e:
            logger.error(f"Error requesting charging for vehicle {vehicle.vehicle_id}: {str(e)}")

    # ... (rest of the simulator code remains the same, just update data lake events to include optimizer_type)

# Example usage
if __name__ == "__main__":
    config = SimulationConfig(
        fleet_size=10,
        num_chargers=5,
        charger_power=150.0,
        optimizer_type="tou"  # or "demand"
    )
    
    simulator = ContinuousFleetSimulator(config)
    asyncio.run(simulator.run_continuous_simulation())