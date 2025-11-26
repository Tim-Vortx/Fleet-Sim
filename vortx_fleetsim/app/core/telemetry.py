# vortx_fleetsim/app/core/telemetry.py
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import asdict

from ..models.vehicle import VehicleState
from ..models.route import RouteInfo

logger = logging.getLogger(__name__)

class TelemetryManager:
    def __init__(self, data_lake_connection=None):
        # State tracking
        self.vehicle_states: Dict[str, List[VehicleState]] = {}  # Historical states
        self.current_states: Dict[str, VehicleState] = {}       # Current states
        self.data_lake = data_lake_connection
        
        # Performance metrics
        self.energy_metrics: Dict[str, Dict[str, float]] = {}
        self.route_metrics: Dict[str, Dict[str, Any]] = {}
        self.charging_metrics: Dict[str, Dict[str, Any]] = {}

    async def record_state(self, vehicle_state: VehicleState) -> None:
        """Record a new vehicle state"""
        try:
            vehicle_id = vehicle_state.vehicle_id
            
            # Store current state
            self.current_states[vehicle_id] = vehicle_state
            
            # Add to historical states
            if vehicle_id not in self.vehicle_states:
                self.vehicle_states[vehicle_id] = []
            self.vehicle_states[vehicle_id].append(vehicle_state)
            
            # Update metrics
            self._update_metrics(vehicle_state)
            
            # Push to data lake if connected
            if self.data_lake:
                await self.data_lake.push_event('vehicle_state_update', {
                    'vehicle_id': vehicle_id,
                    'timestamp': vehicle_state.last_updated,
                    'state': asdict(vehicle_state)
                })
                
        except Exception as e:
            logger.error(f"Error recording state for vehicle {vehicle_state.vehicle_id}: {str(e)}")

    def _update_metrics(self, state: VehicleState) -> None:
        """Update metrics based on new state"""
        vehicle_id = state.vehicle_id
        
        # Initialize metrics if needed
        if vehicle_id not in self.energy_metrics:
            self.energy_metrics[vehicle_id] = {
                'total_energy_consumed': 0.0,
                'total_energy_delivered': 0.0,
                'charging_sessions': 0,
                'route_energy_usage': []
            }
            
        # Update energy metrics
        metrics = self.energy_metrics[vehicle_id]
        metrics['total_energy_consumed'] += state.energy_consumed
        metrics['total_energy_delivered'] += state.energy_delivered
        
        # Update route metrics
        if state.route_id and state.status == 'driving':
            if vehicle_id not in self.route_metrics:
                self.route_metrics[vehicle_id] = {
                    'total_routes': 0,
                    'current_route': None,
                    'route_history': []
                }
            
            if self.route_metrics[vehicle_id]['current_route'] != state.route_id:
                self.route_metrics[vehicle_id]['total_routes'] += 1
                self.route_metrics[vehicle_id]['current_route'] = state.route_id
        
        # Update charging metrics
        if state.status == 'charging':
            if vehicle_id not in self.charging_metrics:
                self.charging_metrics[vehicle_id] = {
                    'total_charging_time': 0,
                    'charging_sessions': 0,
                    'energy_per_session': []
                }
            
            metrics = self.charging_metrics[vehicle_id]
            if not metrics.get('current_session_start'):
                metrics['current_session_start'] = state.last_updated
                metrics['charging_sessions'] += 1
        elif self.charging_metrics.get(vehicle_id, {}).get('current_session_start'):
            # Charging session ended
            start_time = self.charging_metrics[vehicle_id]['current_session_start']
            session_duration = (state.last_updated - start_time).total_seconds() / 3600
            self.charging_metrics[vehicle_id]['total_charging_time'] += session_duration
            self.charging_metrics[vehicle_id]['current_session_start'] = None

    def get_vehicle_metrics(self, vehicle_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a vehicle"""
        return {
            'energy': self.energy_metrics.get(vehicle_id, {}),
            'routes': self.route_metrics.get(vehicle_id, {}),
            'charging': self.charging_metrics.get(vehicle_id, {})
        }

    def get_fleet_metrics(self) -> Dict[str, Any]:
        """Get fleet-wide metrics"""
        total_energy_consumed = sum(m['total_energy_consumed'] 
                                  for m in self.energy_metrics.values())
        total_energy_delivered = sum(m['total_energy_delivered'] 
                                   for m in self.energy_metrics.values())
        total_routes = sum(m['total_routes'] 
                          for m in self.route_metrics.values())
        total_charging_time = sum(m['total_charging_time'] 
                                for m in self.charging_metrics.values())

        return {
            'total_energy_consumed': total_energy_consumed,
            'total_energy_delivered': total_energy_delivered,
            'total_routes': total_routes,
            'total_charging_time': total_charging_time,
            'active_vehicles': len([s for s in self.current_states.values() 
                                  if s.status != 'idle']),
            'charging_vehicles': len([s for s in self.current_states.values() 
                                    if s.status == 'charging']),
            'driving_vehicles': len([s for s in self.current_states.values() 
                                   if s.status == 'driving'])
        }

    def get_historical_states(
        self, 
        vehicle_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[VehicleState]:
        """Get historical states for a vehicle within a time window"""
        states = self.vehicle_states.get(vehicle_id, [])
        if not states:
            return []
            
        if start_time:
            states = [s for s in states if s.last_updated >= start_time]
        if end_time:
            states = [s for s in states if s.last_updated <= end_time]
            
        return states

    def get_efficiency_metrics(self, vehicle_id: str) -> Dict[str, float]:
        """Calculate efficiency metrics for a vehicle"""
        energy_metrics = self.energy_metrics.get(vehicle_id, {})
        route_metrics = self.route_metrics.get(vehicle_id, {})
        
        total_energy = energy_metrics.get('total_energy_consumed', 0)
        total_routes = route_metrics.get('total_routes', 0)
        
        if total_routes > 0:
            return {
                'energy_per_route': total_energy / total_routes,
                'total_efficiency': total_energy / max(1, total_routes),
                'charging_efficiency': (
                    energy_metrics.get('total_energy_delivered', 0) /
                    max(1, energy_metrics.get('charging_sessions', 1))
                )
            }
        return {
            'energy_per_route': 0.0,
            'total_efficiency': 0.0,
            'charging_efficiency': 0.0
        }

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report"""
        return {
            'fleet_metrics': self.get_fleet_metrics(),
            'vehicle_summaries': {
                vid: {
                    'current_state': asdict(state),
                    'metrics': self.get_vehicle_metrics(vid),
                    'efficiency': self.get_efficiency_metrics(vid)
                }
                for vid, state in self.current_states.items()
            },
            'timestamp': datetime.now()
        }

    async def start_monitoring(self) -> None:
        """Start monitoring and periodic reporting"""
        while True:
            try:
                summary = self.generate_summary_report()
                if self.data_lake:
                    await self.data_lake.push_event('telemetry_summary', summary)
                    
            except Exception as e:
                logger.error(f"Error in telemetry monitoring: {str(e)}")
                
            finally:
                await asyncio.sleep(60)  # Update every minute