# vortx_fleetsim/app/simulator/simulation.py
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Literal
import asyncio
from dataclasses import asdict

from ..models.vehicle import VehicleState
from ..models.route import RouteInfo
from ..core.fleet_manager import FleetManager

logger = logging.getLogger(__name__)

OptimizerType = Literal["demand", "tou"]

class FleetSimulator:
    def __init__(
        self,
        fleet_manager: FleetManager,
        data_lake_connection = None
    ):
        # Store manager and configuration
        self.fleet_manager = fleet_manager
        self.data_lake = data_lake_connection
        
        # Timing and state
        self.current_time: Optional[datetime] = None
        self.interval_minutes = 5
        
        # Performance metrics
        self.total_energy_delivered = 0.0
        self.total_energy_consumed = 0.0
        self.peak_power_demand = 0.0
        self.daily_stats = {}

    async def _push_to_data_lake(self, event_type: str, data: dict):
        """Push event data to data lake if connection exists."""
        if self.data_lake:
            await self.data_lake.push_event(event_type, data)
        else:
            # If no data lake connection, just log the event
            logger.debug(f"Event {event_type}: {data}")

    def _check_route_availability(self, vehicle: VehicleState) -> bool:
        """Check if there are routes available for the vehicle."""
        if not hasattr(self.fleet_manager, 'get_available_routes'):
            return False
            
        available_routes = self.fleet_manager.get_available_routes()
        return len(available_routes) > 0

    async def _assign_route(self, vehicle: VehicleState):
        """Assign a route to the vehicle."""
        try:
            if hasattr(self.fleet_manager, 'assign_route'):
                route = self.fleet_manager.assign_route(vehicle.vehicle_id)
                if route:
                    vehicle.route_id = route.route_id
                    vehicle.status = 'driving'
                    await self._push_to_data_lake('route_assigned', {
                        'vehicle_id': vehicle.vehicle_id,
                        'route_id': route.route_id,
                        'timestamp': self.current_time
                    })
        except Exception as e:
            logger.error(f"Error assigning route to vehicle {vehicle.vehicle_id}: {str(e)}")

    def _calculate_route_energy(self, vehicle: VehicleState) -> float:
        """Calculate energy consumption for current route segment."""
        try:
            if not vehicle.route_id:
                return 0.0
            
            # Get route details if available
            route = self.fleet_manager.active_routes.get(vehicle.route_id)
            if route and hasattr(route, 'energy_consumption'):
                return route.energy_consumption
            
            # Default consumption rate (kWh/mile)
            default_consumption_rate = 1.5
            return default_consumption_rate * 15  # Assume 15 miles per interval
            
        except Exception as e:
            logger.error(f"Error calculating route energy: {str(e)}")
            return 0.0

    def _is_route_complete(self, vehicle: VehicleState) -> bool:
        """Check if current route is complete."""
        try:
            if not vehicle.route_id:
                return True
                
            route = self.fleet_manager.active_routes.get(vehicle.route_id)
            if route:
                return route.end_time <= self.current_time
            
            # Fallback: Simple time-based completion
            route_start_time = vehicle.last_updated
            route_duration = timedelta(hours=2)  # Default 2-hour route
            return self.current_time - route_start_time >= route_duration
            
        except Exception as e:
            logger.error(f"Error checking route completion: {str(e)}")
            return True

    async def _request_charging(self, vehicle: VehicleState):
        """Request charging for a vehicle."""
        try:
            # Create charging request
            request = {
                'vehicle_id': vehicle.vehicle_id,
                'current_soc': vehicle.current_soc,
                'target_soc': 90.0,  # Default target SOC
                'battery_capacity': vehicle.battery_capacity,
                'arrival_time': self.current_time
            }
            
            # Send request to charging service
            response = await self._send_charging_request(request)
            
            if response and response.get('status') == 'accepted':
                vehicle.status = 'charging'
                vehicle.charger_id = response.get('charger_id')
                await self._push_to_data_lake('charging_started', {
                    'vehicle_id': vehicle.vehicle_id,
                    'charger_id': vehicle.charger_id,
                    'timestamp': self.current_time
                })
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error requesting charging: {str(e)}")
            return False

    async def _simulate_vehicle_operations(self, vehicle_id: str):
        """Simulate individual vehicle operations."""
        vehicle = self.fleet_manager.fleet_composition.get(vehicle_id)
        if not vehicle:
            return
        
        try:
            if vehicle.status == 'driving':
                # Calculate and apply energy consumption
                energy_consumption = self._calculate_route_energy(vehicle)
                vehicle.energy_consumed += energy_consumption
                vehicle.current_soc = max(0, vehicle.current_soc - 
                                        (energy_consumption / vehicle.battery_capacity * 100))
                
                # Check if route is complete
                if self._is_route_complete(vehicle):
                    vehicle.status = 'idle'
                    vehicle.route_id = None
                    await self._push_to_data_lake('route_completed', {
                        'vehicle_id': vehicle_id,
                        'energy_consumed': vehicle.energy_consumed,
                        'final_soc': vehicle.current_soc,
                        'timestamp': self.current_time
                    })
                    
            elif vehicle.status == 'charging':
                # Update charging status from charging service
                charging_status = await self._get_charging_status(vehicle)
                if charging_status:
                    vehicle.current_soc = charging_status.get('current_soc', vehicle.current_soc)
                    vehicle.energy_delivered = charging_status.get('energy_delivered', 0.0)
                    
                    if charging_status.get('status') == 'complete':
                        vehicle.status = 'idle'
                        vehicle.charger_id = None
                    
            elif vehicle.status == 'idle':
                # Check if vehicle needs charging or can take a new route
                if vehicle.current_soc < 80.0:
                    await self._request_charging(vehicle)
                elif self._check_route_availability(vehicle):
                    await self._assign_route(vehicle)
                    
        except Exception as e:
            logger.error(f"Error simulating vehicle {vehicle_id}: {str(e)}")
            
        finally:
            vehicle.last_updated = self.current_time

    async def run_continuous_simulation(self):
        """Main simulation loop."""
        try:
            if not self.current_time:
                self.current_time = datetime.now()
                self.current_time = self.current_time.replace(
                    minute=self.current_time.minute - (self.current_time.minute % self.interval_minutes),
                    second=0,
                    microsecond=0
                )

            while True:
                # Simulate each vehicle
                for vehicle_id in list(self.fleet_manager.fleet_composition.keys()):
                    await self._simulate_vehicle_operations(vehicle_id)
                
                # Update simulation time
                self.current_time += timedelta(minutes=self.interval_minutes)
                
                # Update daily statistics if needed
                if self.current_time.hour == 0 and self.current_time.minute == 0:
                    self._update_daily_stats()
                
                # Event logging
                await self._push_to_data_lake('simulation_step', {
                    'timestamp': self.current_time,
                    'fleet_status': {
                        vid: {
                            'status': v.status,
                            'soc': v.current_soc,
                            'energy_consumed': v.energy_consumed,
                            'energy_delivered': v.energy_delivered
                        }
                        for vid, v in self.fleet_manager.fleet_composition.items()
                    }
                })

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
            raise

    async def _send_charging_request(self, request: dict):
        """Send charging request to charging service."""
        # In microservices architecture, this would make an HTTP request
        # For now, returning mock response
        return {
            'status': 'accepted',
            'charger_id': f'CHG_{hash(request["vehicle_id"]) % 10:03d}'
        }

    async def _get_charging_status(self, vehicle: VehicleState):
        """Get charging status from charging service."""
        # In microservices architecture, this would make an HTTP request
        # For now, returning mock status
        if vehicle.charger_id:
            return {
                'status': 'charging',
                'current_soc': min(vehicle.current_soc + 2.0, 100.0),  # Simple increment
                'energy_delivered': 5.0  # kWh per interval
            }
        return None

    def _update_daily_stats(self):
        """Update daily statistics."""
        current_date = self.current_time.date()
        self.daily_stats[current_date] = {
            'total_energy_delivered': self.total_energy_delivered,
            'total_energy_consumed': self.total_energy_consumed,
            'vehicles_charged': len([v for v in self.fleet_manager.fleet_composition.values() 
                                   if v.energy_delivered > 0]),
            'routes_completed': len([v for v in self.fleet_manager.fleet_composition.values() 
                                   if not v.route_id and v.energy_consumed > 0])
        }