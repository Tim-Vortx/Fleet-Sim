from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import logging
import traceback
from heapq import heappush, heappop
import pandas as pd
from copy import deepcopy
from collections import defaultdict

# Import required functions from v5_util
from v11_util import (
    load_bev_energy_prices,
    calculate_load_curve,
    calculate_charging_cost,
    calculate_charger_utilization,
    round_to_interval,
    INTERVAL_MINUTES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChargingSession:
    vehicle_id: str
    charger_id: int
    start_time: datetime
    target_end_time: datetime
    initial_soc: float
    target_soc: float
    battery_capacity: float
    max_charging_rate: float
    availability: List[Tuple[datetime, datetime]]
    power_profile: List[float]
    current_soc: float
    route_distance: float
    energy_needed: float
    energy_delivered: float = 0.0
    completion_time: Optional[datetime] = None
    buffer_minutes: int = 15
    charging_start: Optional[datetime] = None
    last_vehicle_end_time: Optional[datetime] = None
    base_departure: Optional[datetime] = None
    base_return: Optional[datetime] = None

    def validate(self) -> bool:
        """Validate session data."""
        try:
            # Basic validation
            if not isinstance(self.start_time, datetime) or not isinstance(self.target_end_time, datetime):
                logger.error(f"Invalid time format in session for vehicle {self.vehicle_id}")
                return False

            if not (0 <= self.initial_soc <= 100) or not (0 <= self.target_soc <= 100):
                logger.error(f"Invalid SOC values in session for vehicle {self.vehicle_id}")
                return False

            if self.battery_capacity <= 0 or self.max_charging_rate <= 0:
                logger.error(f"Invalid capacity or charging rate in session for vehicle {self.vehicle_id}")
                return False

            # Validate charging window
            if self.base_return is not None and self.base_departure is not None:
                charging_window = self.base_departure - self.base_return
                window_hours = charging_window.total_seconds() / 3600

                # Ensure charging window is reasonable (between 0 and 24 hours)
                if not (0 <= window_hours <= 24):
                    logger.error(f"Invalid charging window duration for vehicle {self.vehicle_id}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Session validation error for vehicle {self.vehicle_id}: {str(e)}")
            return False

@dataclass
class ChargingRequest:
    vehicle_id: str
    start_time: datetime
    target_end_time: datetime
    initial_soc: float
    target_soc: float
    battery_capacity: float
    max_charging_rate: float
    availability: List[Tuple[datetime, datetime]]
    route_distance: float
    energy_needed: float
    efficiency: float
    priority: float = 0.0
    base_departure: Optional[datetime] = None
    base_return: Optional[datetime] = None

    def validate_times(self) -> bool:
        """Validate time-related fields."""
        try:
            if not isinstance(self.start_time, datetime) or not isinstance(self.target_end_time, datetime):
                logger.error(f"Invalid time format for vehicle {self.vehicle_id}")
                return False

            if self.start_time >= self.target_end_time:
                logger.error(f"Start time must be before end time for vehicle {self.vehicle_id}")
                return False

            # Validate availability windows
            for start, end in self.availability:
                if not isinstance(start, datetime) or not isinstance(end, datetime):
                    logger.error(f"Invalid availability window format for vehicle {self.vehicle_id}")
                    return False
                if start >= end:
                    logger.error(f"Invalid availability window (start >= end) for vehicle {self.vehicle_id}")
                    return False

            return True
        except Exception as e:
            logger.error(f"Time validation error for vehicle {self.vehicle_id}: {str(e)}")
            return False


class OpportunityChargingSystem:
    def __init__(
        self,
        num_chargers: int,
        charger_power: float,
        demand_limit: float = None,
        interval_minutes: int = INTERVAL_MINUTES,
        simulation_hours: int = 48,
        energy_prices: Optional[Dict] = None
    ):
        # Core initialization
        self.num_chargers = num_chargers
        self.max_charger_power = charger_power
        self.demand_limit = num_chargers * charger_power
        self.interval_minutes = interval_minutes
        self.energy_prices = energy_prices or load_bev_energy_prices()
        self.simulation_hours = simulation_hours

        # Initialize core data structures
        self.chargers = {i: None for i in range(num_chargers)}
        self.active_sessions: Dict[str, ChargingSession] = {}
        self.completed_sessions: List[ChargingSession] = []
        self.waiting_list: List[ChargingRequest] = []
        self.time_slots: Optional[pd.DatetimeIndex] = None
        
        # Track last vehicle end time for each charger
        self.charger_last_use = {i: None for i in range(num_chargers)}

        # Performance metrics
        self.peak_demand: float = 0.0
        self.total_energy_delivered: float = 0.0
        self.load_curve: List[float] = []
        
        # Known vehicles set for tracking
        self.known_vehicles: Set[str] = set()

        # Initialize charging failures tracking
        self.charging_failures = {
            "missed_soc": 0,
            "missed_departure": 0,
            "total_vehicles": 0
        }
        
        self.simulation_start = None

    def initialize_simulation(self, start_time: datetime) -> None:
        """Initialize simulation with start time and create time slots."""
        try:
            if not isinstance(start_time, datetime):
                raise ValueError("Invalid start time format")
                
            self.simulation_start = round_to_interval(start_time, self.interval_minutes)
            
            # Create time slots for the simulation period
            num_intervals = (self.simulation_hours * 60) // self.interval_minutes
            self.time_slots = []
            
            for i in range(num_intervals):
                slot_time = self.simulation_start + timedelta(minutes=i * self.interval_minutes)
                self.time_slots.append(slot_time)
            
            # Initialize load curve
            self.load_curve = [0.0] * len(self.time_slots)
            
            # Reset performance metrics
            self.peak_demand = 0.0
            self.total_energy_delivered = 0.0
            
        except Exception as e:
            logger.error(f"Error initializing simulation: {str(e)}")
            raise

    def add_charging_request(self, request: ChargingRequest, current_time: datetime) -> None:
        """Add a new charging request to the system."""
        try:
            if request.vehicle_id in self.known_vehicles:
                return
            
            # Initialize simulation if needed
            if self.simulation_start is None:
                self.initialize_simulation(current_time)
            
            # Validate request timing
            if not request.validate_times():
                logger.error(f"Invalid request times for vehicle {request.vehicle_id}")
                return
            
            # Round times to intervals
            request.start_time = round_to_interval(request.start_time, self.interval_minutes)
            request.target_end_time = round_to_interval(request.target_end_time, self.interval_minutes)
            
            # Try to find an available charger
            available_charger = self._find_available_charger(current_time)
            
            if available_charger is not None:
                if self._start_charging_session(request, available_charger, current_time):
                    self.known_vehicles.add(request.vehicle_id)
            else:
                self.waiting_list.append(request)
                self.known_vehicles.add(request.vehicle_id)
                
        except Exception as e:
            logger.error(f"Error adding charging request for vehicle {request.vehicle_id}: {str(e)}")

    def simulate_time_step(self, current_time: datetime) -> Dict:
        """Simulate one time step of the charging system."""
        try:
            current_time = round_to_interval(current_time, self.interval_minutes)
            
            # Update priorities for waiting vehicles
            self._update_priorities(current_time)
            
            # Process completed sessions
            completed_vehicle_ids = []
            for vehicle_id, session in self.active_sessions.items():
                if self._check_session_completion(session, current_time):
                    completed_vehicle_ids.append(vehicle_id)
            
            for vehicle_id in completed_vehicle_ids:
                if vehicle_id in self.active_sessions:
                    del self.active_sessions[vehicle_id]
            
            # Start new sessions if possible
            while self.waiting_list:
                available_charger = self._find_available_charger(current_time)
                if available_charger is None:
                    break
                
                request = self.waiting_list.pop(0)
                self._start_charging_session(request, available_charger, current_time)
            
            # Calculate and apply power allocation
            active_sessions = list(self.active_sessions.values())
            power_allocation = self._calculate_power_allocation(active_sessions, current_time)
            
            # Track total demand for this time step
            current_demand = sum(power_allocation.values())
            
            # Update sessions with allocated power
            for vehicle_id, power in power_allocation.items():
                session = self.active_sessions[vehicle_id]
                
                # Calculate energy delivered
                hours = self.interval_minutes / 60
                energy_delivered = power * hours
                
                # Update session state
                session.energy_delivered += energy_delivered
                session.current_soc += (energy_delivered / session.battery_capacity * 100)
                session.power_profile.append(power)
                
                # Track total energy delivered
                self.total_energy_delivered += energy_delivered
            
            # Update metrics
            time_slot_index = self._get_time_slot_index(current_time)
            if time_slot_index is not None and time_slot_index < len(self.load_curve):
                self.load_curve[time_slot_index] = current_demand
                self.peak_demand = max(self.peak_demand, current_demand)
            
            return {
                'current_load': current_demand,
                'waiting_vehicles': [req.vehicle_id for req in self.waiting_list],
                'active_sessions': [
                    {
                        'vehicle_id': vid,
                        'current_soc': session.current_soc,
                        'power': session.power_profile[-1] if session.power_profile else 0
                    }
                    for vid, session in self.active_sessions.items()
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in simulate_time_step: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'current_load': 0,
                'waiting_vehicles': [],
                'active_sessions': []
            }

    def _find_available_charger(self, current_time: datetime) -> Optional[int]:
        """Find first available charger with sufficient buffer time."""
        best_charger = None
        max_buffer_time = 0

        for charger_id, vehicle_id in self.chargers.items():
            if vehicle_id is None:
                last_use_time = self.charger_last_use[charger_id]
                if last_use_time is None:
                    return charger_id
                
                buffer_time = (current_time - last_use_time).total_seconds() / 60
                if buffer_time >= 15:
                    if buffer_time > max_buffer_time:
                        max_buffer_time = buffer_time
                        best_charger = charger_id
                
        return best_charger

    def _start_charging_session(
        self, 
        request: ChargingRequest, 
        charger_id: int, 
        current_time: datetime
    ) -> bool:
        """Initialize a new charging session and assign it to a charger."""
        try:
            # Create charging session
            session = ChargingSession(
                vehicle_id=request.vehicle_id,
                charger_id=charger_id,
                start_time=current_time,
                target_end_time=request.target_end_time,
                initial_soc=request.initial_soc,
                target_soc=request.target_soc,
                battery_capacity=request.battery_capacity,
                max_charging_rate=min(request.max_charging_rate, self.max_charger_power),
                availability=request.availability,
                power_profile=[],
                current_soc=request.initial_soc,
                route_distance=request.route_distance,
                energy_needed=request.energy_needed,
                charging_start=current_time,
                last_vehicle_end_time=self.charger_last_use[charger_id],
                base_departure=request.base_departure,
                base_return=request.base_return
            )

            if not session.validate():
                logger.error(f"Invalid session data for vehicle {request.vehicle_id}")
                return False
            
            self.active_sessions[request.vehicle_id] = session
            self.chargers[charger_id] = request.vehicle_id
            return True
            
        except Exception as e:
            logger.error(f"Error starting charging session: {str(e)}")
            return False

    def _check_session_completion(self, session: ChargingSession, current_time: datetime) -> bool:
            """Check if a charging session is complete."""
            try:
                if session.vehicle_id not in self.known_vehicles:
                    self.charging_failures["total_vehicles"] += 1
                    self.known_vehicles.add(session.vehicle_id)

                if session.current_soc >= session.target_soc:
                    session.completion_time = current_time
                    self.completed_sessions.append(session)
                    charger_id = next(cid for cid, vid in self.chargers.items() 
                                    if vid == session.vehicle_id)
                    self.charger_last_use[charger_id] = current_time
                    self.chargers[charger_id] = None
                    return True
                
                departure_time = session.base_departure or session.target_end_time
                if current_time >= departure_time:
                    if session.current_soc < session.target_soc:
                        self.charging_failures["missed_departure"] += 1
                    session.completion_time = current_time
                    self.completed_sessions.append(session)
                    charger_id = next(cid for cid, vid in self.chargers.items() 
                                    if vid == session.vehicle_id)
                    self.charger_last_use[charger_id] = current_time
                    self.chargers[charger_id] = None
                    return True
                
                return False
            
            except Exception as e:
                logger.error(f"Error checking session completion: {str(e)}")
                return False

    def _calculate_power_allocation(self, sessions: List[ChargingSession], current_time: datetime) -> Dict[str, float]:
        """Calculate power allocation for active charging sessions."""
        if not sessions:
            return {}
        
        try:
            power_allocation = {}
            for session in sessions:
                if session.current_soc >= session.target_soc:
                    power_allocation[session.vehicle_id] = 0.0
                    continue

                power_allocation[session.vehicle_id] = session.max_charging_rate
                
            return power_allocation
            
        except Exception as e:
            logger.error(f"Error calculating power allocation: {str(e)}")
            return {session.vehicle_id: 0.0 for session in sessions}

    def _update_priorities(self, current_time: datetime) -> None:
        """Update priorities for vehicles in the waiting list."""
        if not self.waiting_list:
            return

        try:
            # Sort by earliest departure time
            self.waiting_list.sort(key=lambda x: x.target_end_time)
        except Exception as e:
            logger.error(f"Error updating priorities: {str(e)}")

    def _get_time_slot_index(self, current_time: datetime) -> Optional[int]:
        """Get the index of the current time slot."""
        try:
            if not self.time_slots:
                return None
            
            minutes_diff = int((current_time - self.simulation_start).total_seconds() / 60)
            index = minutes_diff // self.interval_minutes
            
            if 0 <= index < len(self.time_slots):
                return index
            return None
            
        except Exception as e:
            logger.error(f"Error getting time slot index: {str(e)}")
            return None

    def get_results(self) -> dict:
        """Get final simulation results including cost calculations."""
        # Initialize total cost
        total_cost = 0.0
        per_vehicle_costs = {}
        cost_breakdown = defaultdict(lambda: {'energy': 0.0, 'cost': 0.0})

        # Calculate costs for each completed session
        for session in self.completed_sessions + list(self.active_sessions.values()):
            vehicle_cost = 0.0
            vehicle_energy = 0.0

            if session.charging_start:
                current_time = session.charging_start
                for power in session.power_profile:
                    if power > 0:
                        # Calculate energy for this interval
                        hours = self.interval_minutes / 60
                        energy = power * hours
                        
                        # Get price for this hour
                        hour = current_time.hour
                        price = next(p for (start, end), p in self.energy_prices.items() 
                                if start <= hour < end)
                        
                        # Calculate cost
                        interval_cost = energy * price
                        
                        # Track costs
                        total_cost += interval_cost
                        vehicle_cost += interval_cost
                        vehicle_energy += energy

                        # Track by TOU period
                        period = f"{hour:02d}-{(hour+1):02d}"
                        cost_breakdown[period]['energy'] += energy
                        cost_breakdown[period]['cost'] += interval_cost

                    current_time += timedelta(minutes=self.interval_minutes)

            # Store per-vehicle costs
            per_vehicle_costs[session.vehicle_id] = {
                "total_cost": vehicle_cost,
                "avg_rate": vehicle_cost / vehicle_energy if vehicle_energy > 0 else 0
            }

        # Prepare base results
        results = {
            "completed_sessions": len(self.completed_sessions),
            "peak_demand": self.peak_demand,
            "total_energy_delivered": self.total_energy_delivered,
            "total_cost": total_cost,
            "charging_failures": self.charging_failures.copy(),
            "vehicle_summaries": [],
            "per_vehicle_costs": per_vehicle_costs,
            "cost_breakdown": dict(cost_breakdown),
            "unscheduled_vehicles": []
        }

        # Add vehicle summaries
        for session in self.completed_sessions + list(self.active_sessions.values()):
            charging_duration = None
            avg_charging_rate = 0
            
            if session.charging_start and session.completion_time:
                charging_duration = (
                    session.completion_time - session.charging_start
                ).total_seconds() / 3600
                
                if charging_duration > 0:
                    avg_charging_rate = session.energy_delivered / charging_duration

            summary = {
                "vehicle_id": session.vehicle_id,
                "energy_delivered": session.energy_delivered,
                "final_soc": session.current_soc,
                "target_soc": session.target_soc,
                "completed": session.current_soc >= session.target_soc,
                "start_time": session.charging_start,
                "end_time": session.completion_time,
                "avg_charging_rate": avg_charging_rate,
                "base_departure": session.base_departure,
                "base_return": session.base_return
            }
            results["vehicle_summaries"].append(summary)

        # Get unscheduled vehicles
        unscheduled = []
        for vehicle_id in self.known_vehicles:
            if vehicle_id not in {s.vehicle_id for s in self.completed_sessions}:
                # Only add to unscheduled if not in waiting list
                if vehicle_id not in {r.vehicle_id for r in self.waiting_list}:
                    unscheduled.append(vehicle_id)
        results["unscheduled_vehicles"] = unscheduled

        # Log results
        logger.info("\n=== Final Simulation Summary ===")
        logger.info(f"Total vehicles completed charging: {len(self.completed_sessions)}")
        logger.info(f"Total energy delivered: {self.total_energy_delivered:.2f} kWh")
        logger.info(f"Total charging cost: ${total_cost:.2f}")
        logger.info(f"Peak demand during simulation: {self.peak_demand:.2f} kW")

        # Detailed per-vehicle summary
        for summary in results["vehicle_summaries"]:
            status = "Completed" if summary["completed"] else "Incomplete"
            start_time_str = summary["start_time"].strftime("%Y-%m-%d %H:%M:%S") if summary["start_time"] else "N/A"
            end_time_str = summary["end_time"].strftime("%Y-%m-%d %H:%M:%S") if summary["end_time"] else "N/A"
            return_str = summary["base_return"].strftime("%Y-%m-%d %H:%M:%S") if summary["base_return"] else "N/A"
            departure_str = summary["base_departure"].strftime("%Y-%m-%d %H:%M:%S") if summary["base_departure"] else "N/A"
            
            logger.info(
                f"Vehicle {summary['vehicle_id']}: {status}, "
                f"Depot Return: {return_str}, "          # When vehicle returns current day
                f"Charging Start: {start_time_str}, "
                f"Charging End: {end_time_str}, "
                f"Next Departure: {departure_str}, "     # Vehicle's next day departure
                f"Energy Delivered: {summary['energy_delivered']:.2f}kWh, "
                f"Avg Rate: {summary['avg_charging_rate']:.2f}kW, "
                f"Final SOC: {summary['final_soc']:.1f}%"
            )

        return results
