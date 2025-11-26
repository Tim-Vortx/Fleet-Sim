# vortx_fleetsim/app/core/route_manager.py
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..models.route import RouteInfo
from ..models.vehicle import VehicleState

logger = logging.getLogger(__name__)

class RouteManager:
    def __init__(self):
        self.active_routes: Dict[str, RouteInfo] = {}
        self.completed_routes: List[RouteInfo] = []
        self.route_templates: Dict[str, RouteInfo] = {}
        self.vehicle_assignments: Dict[str, str] = {}  # vehicle_id -> route_id

    def assign_route(self, vehicle_id: str, route_specs: Optional[Dict] = None) -> Optional[RouteInfo]:
        """
        Assign a route to a vehicle based on specifications or default template.
        
        Args:
            vehicle_id: Identifier for the vehicle
            route_specs: Optional specifications for route generation
        
        Returns:
            RouteInfo if successfully assigned, None otherwise
        """
        try:
            # Generate or select route
            if route_specs:
                route = self._generate_route(route_specs)
            else:
                route = self._get_next_available_route()

            if not route:
                logger.warning(f"No suitable route found for vehicle {vehicle_id}")
                return None

            # Update route with vehicle assignment
            route.vehicle_id = vehicle_id
            
            # Store assignments
            self.active_routes[route.route_id] = route
            self.vehicle_assignments[vehicle_id] = route.route_id
            
            logger.info(f"Assigned route {route.route_id} to vehicle {vehicle_id}")
            return route

        except Exception as e:
            logger.error(f"Error assigning route to vehicle {vehicle_id}: {str(e)}")
            return None

    def get_route_status(self, route_id: str) -> Optional[RouteInfo]:
        """Get current status of a route"""
        return self.active_routes.get(route_id)

    def complete_route(self, route_id: str) -> bool:
        """Mark a route as completed"""
        try:
            if route_id in self.active_routes:
                route = self.active_routes.pop(route_id)
                route.end_time = datetime.now()
                self.completed_routes.append(route)
                
                # Clear vehicle assignment
                if route.vehicle_id:
                    self.vehicle_assignments.pop(route.vehicle_id, None)
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error completing route {route_id}: {str(e)}")
            return False

    def _generate_route(self, specs: Dict) -> RouteInfo:
        """
        Generate a new route based on specifications.
        
        Args:
            specs: Dictionary containing route specifications:
                - distance: float (miles)
                - start_time: datetime
                - end_time: datetime
                - stops: List[Tuple[float, float]] optional
                - energy_consumption: float optional
        """
        try:
            route_id = f"R{len(self.active_routes) + len(self.completed_routes) + 1:04d}"
            
            # Get required specs
            distance = float(specs.get('distance', 100.0))
            start_time = specs.get('start_time', datetime.now())
            
            # Calculate default end time if not provided
            if 'end_time' not in specs:
                # Assume average speed of 30 mph
                duration_hours = distance / 30.0
                end_time = start_time + timedelta(hours=duration_hours)
            else:
                end_time = specs['end_time']

            # Generate stops if not provided
            stops = specs.get('stops', self._generate_random_stops(distance))

            # Calculate energy consumption if not provided
            # Default assumption: 1.5 kWh per mile
            energy_consumption = specs.get('energy_consumption', distance * 1.5)

            return RouteInfo(
                route_id=route_id,
                distance=distance,
                start_time=start_time,
                end_time=end_time,
                energy_consumption=energy_consumption,
                stops=stops
            )

        except Exception as e:
            logger.error(f"Error generating route: {str(e)}")
            raise

    def _generate_random_stops(self, route_distance: float) -> List[Tuple[float, float]]:
        """Generate random stops for a route based on distance"""
        num_stops = max(2, int(route_distance / 50))  # At least start and end
        
        # Generate random stops in LA area
        la_bounds = {
            'lat': (33.7037, 34.3373),  # LA County rough boundaries
            'lng': (-118.6682, -118.1553)
        }
        
        stops = []
        for _ in range(num_stops):
            lat = np.random.uniform(la_bounds['lat'][0], la_bounds['lat'][1])
            lng = np.random.uniform(la_bounds['lng'][0], la_bounds['lng'][1])
            stops.append((lat, lng))
            
        return stops

    def _get_next_available_route(self) -> Optional[RouteInfo]:
        """Get next available route from templates or generate new one"""
        # First try to get from templates
        for route_id, template in self.route_templates.items():
            if route_id not in self.active_routes:
                # Create new route from template
                new_route = RouteInfo(
                    route_id=f"R{len(self.active_routes) + len(self.completed_routes) + 1:04d}",
                    distance=template.distance,
                    start_time=datetime.now(),
                    end_time=datetime.now() + timedelta(hours=template.distance/30.0),
                    energy_consumption=template.energy_consumption,
                    stops=template.stops
                )
                return new_route

        # If no template available, generate new route
        return self._generate_route({
            'distance': np.random.uniform(50, 200)
        })

    def get_vehicle_route(self, vehicle_id: str) -> Optional[RouteInfo]:
        """Get the current route for a vehicle"""
        route_id = self.vehicle_assignments.get(vehicle_id)
        if route_id:
            return self.active_routes.get(route_id)
        return None

    def calculate_completion_percentage(self, route_id: str, current_time: datetime) -> float:
        """Calculate the completion percentage of a route"""
        route = self.active_routes.get(route_id)
        if not route:
            return 0.0
            
        total_duration = (route.end_time - route.start_time).total_seconds()
        elapsed_time = (current_time - route.start_time).total_seconds()
        
        return min(100.0, max(0.0, (elapsed_time / total_duration) * 100))

    def import_route_templates(self, templates: List[Dict]) -> None:
        """Import route templates from configuration"""
        try:
            for template in templates:
                route_info = self._generate_route(template)
                self.route_templates[route_info.route_id] = route_info
        except Exception as e:
            logger.error(f"Error importing route templates: {str(e)}")

    def get_route_metrics(self) -> Dict:
        """Get metrics about routes"""
        return {
            'active_routes': len(self.active_routes),
            'completed_routes': len(self.completed_routes),
            'total_distance': sum(r.distance for r in self.completed_routes),
            'total_energy': sum(r.energy_consumption for r in self.completed_routes),
            'average_distance': np.mean([r.distance for r in self.completed_routes]) if self.completed_routes else 0,
            'average_duration': np.mean([
                (r.end_time - r.start_time).total_seconds() / 3600 
                for r in self.completed_routes
            ]) if self.completed_routes else 0
        }