from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, NamedTuple
import numpy as np
import logging
import traceback
from heapq import heappush, heappop
import pandas as pd
import os
from copy import deepcopy
from collections import defaultdict

# Import required functions from v5_util
from v11_util import (
    load_bev_energy_prices,
    calculate_load_curve,
    calculate_charging_cost,
    calculate_charger_utilization,
    round_to_interval,
    INTERVAL_MINUTES,
    RATE_FILE
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class OptimizationResult(NamedTuple):
    """Stores the result of a single optimization pass."""
    power_profile: Dict[str, List[float]]  # Vehicle ID to power profile mapping
    completion_times: Dict[str, datetime]   # Vehicle ID to completion time mapping
    peak_demand: float
    total_energy: float
    score: float  # Overall optimization score

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

    def __lt__(self, other):
        return self.priority < other.priority

    def get_normalized_schedule(self) -> Tuple[datetime, datetime]:
        """
        Get charging window between current return and next departure.
        
        base_return = When vehicle returns/arrives (current day)
        base_departure = When vehicle needs to depart next (next day)
        """
        if self.base_return is None or self.base_departure is None:
            return self.start_time, self.target_end_time

        # Charging window starts when vehicle returns
        charge_start = self.base_return
        
        # Charging must finish before next departure 
        charge_end = self.base_departure
        
        # Validation to ensure charge_end is after charge_start
        if (charge_end - charge_start).total_seconds() <= 0:
            logger.error(f"Invalid charging window for vehicle {self.vehicle_id}: "
                        f"End time is before start time")
            
        return charge_start, charge_end

    def validate_times(self) -> bool:
        """Validate time-related fields with comprehensive overnight handling."""
        try:
            if not isinstance(self.start_time, datetime) or not isinstance(self.target_end_time, datetime):
                logger.error(f"Invalid time format for vehicle {self.vehicle_id}")
                return False

            if self.start_time >= self.target_end_time:
                logger.error(f"Start time must be before end time for vehicle {self.vehicle_id}")
                return False

            # Validate base_return and base_departure if provided
            if self.base_return is not None and self.base_departure is not None:
                return_time, departure_time = self.get_normalized_schedule()
                time_difference = (departure_time - return_time).total_seconds() / 3600

                # Ensure the charging window is reasonable (between 0 and 24 hours)
                if not (0 <= time_difference <= 24):
                    logger.error(f"Invalid charging window for vehicle {self.vehicle_id}: "
                               f"Duration {time_difference:.1f} hours exceeds limits")
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

    def calculate_priority(self, current_time: datetime, reverse: bool = False) -> float:
        """Calculate priority score based on energy needs and time constraints."""
        try:
            current_time = round_to_interval(current_time, INTERVAL_MINUTES)
            
            # If base_departure/return not set, use target_end_time/start_time
            actual_departure = self.base_departure if self.base_departure is not None else self.target_end_time
            actual_return = self.base_return if self.base_return is not None else self.start_time

            # Calculate time to departure
            time_to_departure = max(0.1, (actual_departure - current_time).total_seconds() / 3600)
            charging_window = max(0.1, (actual_departure - actual_return).total_seconds() / 3600)
            time_position = 1.0 - (time_to_departure / charging_window)

            # Calculate urgency
            if not reverse:
                urgency_factor = np.exp(-6.0 * (time_to_departure / 24.0))
            else:
                urgency_factor = 1.0 - np.exp(-6.0 * ((24.0 - time_to_departure) / 24.0))

            # Calculate energy needs
            current_energy = self.battery_capacity * (self.initial_soc / 100)
            target_energy = self.battery_capacity * (self.target_soc / 100)
            remaining_energy = target_energy - current_energy
            min_power_needed = remaining_energy / time_to_departure

            # Calculate departure urgency
            departure_urgency = 1.0
            if time_to_departure < 4.0:
                departure_urgency = 2.0
            elif time_to_departure < 8.0:
                departure_urgency = 1.5

            # Set priority
            if not reverse:
                self.priority = (
                    0.2 * (remaining_energy / self.battery_capacity) +
                    0.6 * urgency_factor * min_power_needed +
                    0.2 * departure_urgency
                )
            else:
                self.priority = (
                    0.4 * (remaining_energy / self.battery_capacity) +
                    0.4 * urgency_factor +
                    0.2 * (1.0 / departure_urgency)
                )

            return self.priority
        except Exception as e:
            logger.error(f"Error calculating priority for vehicle {self.vehicle_id}: {str(e)}")
            return 0.0
        
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

    def get_adjusted_times(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get adjusted return/departure times handling both same day and overnight."""
        if self.base_return is None or self.base_departure is None:
            return None, None

        adjusted_return = self.base_return
        adjusted_departure = self.base_departure

        # If return time is later than departure time, it's a same-day scenario
        if self.base_return > self.base_departure:
            # No adjustment needed for same day
            pass
        else:
            # Overnight scenario - adjust departure to next day
            adjusted_departure = self.base_departure + timedelta(days=1)

        return adjusted_return, adjusted_departure

    def get_normalized_schedule(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get normalized schedule times that handle all charging scenarios.
        Returns (return_time, departure_time) with proper day alignment.
        """
        if self.base_return is None or self.base_departure is None:
            return None, None

        # Convert times to timestamps for comparison
        return_ts = self.base_return.timestamp()
        departure_ts = self.base_departure.timestamp()
        
        # If departure is before return, assume it's next-day departure
        if departure_ts < return_ts:
            normalized_departure = self.base_departure + timedelta(days=1)
            return self.base_return, normalized_departure
            
        return self.base_return, self.base_departure

    def validate(self) -> bool:
        """Validate session data with comprehensive overnight handling."""
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
                return_time, departure_time = self.get_normalized_schedule()
                if return_time and departure_time:
                    # Calculate charging window duration
                    charging_window = departure_time - return_time
                    window_hours = charging_window.total_seconds() / 3600

                    # Ensure charging window is reasonable
                    if window_hours <= 0:
                        logger.error(f"Invalid charging window for vehicle {self.vehicle_id}: "
                                   f"Negative duration")
                        return False
                    
                    if window_hours > 24:
                        logger.error(f"Invalid charging window for vehicle {self.vehicle_id}: "
                                   f"Duration {window_hours:.1f} hours exceeds 24-hour limit")
                        return False

                    # Check minimum charging window
                    min_charging_time = timedelta(minutes=30)
                    if charging_window < min_charging_time:
                        logger.error(f"Charging window too short for vehicle {self.vehicle_id}")
                        return False

            return True
        except Exception as e:
            logger.error(f"Session validation error for vehicle {self.vehicle_id}: {str(e)}")
            return False

    def get_charging_window(self) -> Tuple[datetime, datetime]:
        """Get the actual charging window considering all scenarios."""
        return_time, departure_time = self.get_normalized_schedule()
        if return_time is None or departure_time is None:
            return self.start_time, self.target_end_time
        return return_time, departure_time
    
class ChargingOptimizerDemand:
    def __init__(
        self,
        num_chargers: int,
        charger_power: float,
        demand_limit: float = 1000.0,
        interval_minutes: int = INTERVAL_MINUTES,
        simulation_hours: int = 48,
        energy_prices: Optional[Dict] = None
    ):
        # Core initialization
        self.num_chargers = num_chargers
        self.max_charger_power = charger_power
        self.demand_limit = demand_limit
        self.energy_prices = energy_prices or load_bev_energy_prices()
        self.interval_minutes = interval_minutes
        self.max_power_change_rate = 0.20
        self.simulation_hours = simulation_hours

        # Initialize core data structures
        self.active_sessions: Dict[str, ChargingSession] = {}
        self.charger_assignments: Dict[int, Optional[str]] = {i: None for i in range(num_chargers)}
        self.completed_sessions: List[ChargingSession] = []
        self.waiting_queue: List[Tuple[float, ChargingRequest]] = []
        self.known_vehicles: Set[str] = set()
        
        # Track last vehicle end time for each charger
        self.charger_last_use: Dict[int, Optional[datetime]] = {i: None for i in range(num_chargers)}

        self.simulation_start: Optional[datetime] = None
        self.time_slots: Optional[List[datetime]] = None

        # Performance metrics
        self.peak_demand: float = 0.0
        self.total_energy_delivered: float = 0.0
        self.load_curve: List[float] = []

        # Initialize charging failures with all required keys
        self.charging_failures = {
            "missed_soc": 0,
            "missed_departure": 0,
            "total_vehicles": 0
        }

        # Optimization weights - increased backward weight
        self.forward_weight = 0.3  # Reduced from 0.5
        self.backward_weight = 0.7  # Increased from 0.5
        
        # Store optimization results
        self.forward_result: Optional[OptimizationResult] = None
        self.backward_result: Optional[OptimizationResult] = None
        self.combined_result: Optional[OptimizationResult] = None

    def _calculate_optimization_score(
        self,
        peak_demand: float,
        total_energy: float,
        completion_times: Dict[str, datetime],
        requests: List[ChargingRequest]
    ) -> float:
        """Calculate a score for the optimization result."""
        try:
            # Initialize score components
            demand_score = 1.0 - (peak_demand / self.demand_limit)  # Lower peak demand is better
            
            # Calculate timing score
            timing_scores = []
            for request in requests:
                completion_time = completion_times.get(request.vehicle_id)
                if completion_time:
                    # Calculate how close to deadline the charging completed
                    time_to_deadline = (request.target_end_time - completion_time).total_seconds() / 3600
                    # Score is better if charging completes closer to deadline
                    timing_score = 1.0 / (1.0 + abs(time_to_deadline))
                    # Add bonus for completing very close to deadline
                    if abs(time_to_deadline) < 1.0:  # Within 1 hour of deadline
                        timing_score *= 1.5
                    timing_scores.append(timing_score)
            
            avg_timing_score = sum(timing_scores) / len(timing_scores) if timing_scores else 0.0
            
            # Combine scores with weights heavily favoring timing
            final_score = (
                0.3 * demand_score +      # Weight for demand management (reduced from 0.4)
                0.7 * avg_timing_score    # Weight for timing optimization (increased from 0.6)
            )
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {str(e)}")
            return 0.0

    def optimize_charging_schedule(self, requests: List[ChargingRequest], start_time: datetime) -> OptimizationResult:
        """Run single-pass optimization with improved demand management."""
        try:
            # Create a copy of the optimizer state
            optimizer_state = deepcopy(self)
            optimizer_state.initialize_simulation(start_time)
            current_time = start_time

            # Sort requests by a combination of start time and energy needs
            sorted_requests = sorted(requests, key=lambda x: (
                x.start_time,
                x.energy_needed / (x.target_end_time - x.start_time).total_seconds()  # Rate needed
            ))

            # Process all requests
            for request in sorted_requests:
                optimizer_state.add_charging_request(request, current_time)
                while optimizer_state.active_sessions or optimizer_state.waiting_queue:
                    optimizer_state.simulate_time_step(current_time)
                    current_time += timedelta(minutes=self.interval_minutes)

            # Calculate metrics
            power_profiles = {
                session.vehicle_id: session.power_profile
                for session in optimizer_state.completed_sessions
            }
            completion_times = {
                session.vehicle_id: session.completion_time
                for session in optimizer_state.completed_sessions
            }

            # Calculate optimization score
            score = self._calculate_optimization_score(
                optimizer_state.peak_demand,
                optimizer_state.total_energy_delivered,
                completion_times,
                requests
            )

            return OptimizationResult(
                power_profile=power_profiles,
                completion_times=completion_times,
                peak_demand=optimizer_state.peak_demand,
                total_energy=optimizer_state.total_energy_delivered,
                score=score
            )
                
        except Exception as e:
            logger.error(f"Error in optimize_charging_schedule: {str(e)}")
            return self._run_forward_optimization(requests, start_time)

    def initialize_simulation(self, start_time: datetime) -> None:
        """Initialize simulation with start time and create time slots."""
        try:
            if not isinstance(start_time, datetime):
                raise ValueError("Invalid start time format")
                
            self.simulation_start = round_to_interval(start_time, self.interval_minutes)
            
            # Create time slots for the entire simulation period
            num_intervals = (self.simulation_hours * 60) // self.interval_minutes
            self.time_slots = []
            
            for i in range(num_intervals):
                slot_time = self.simulation_start + timedelta(minutes=i * self.interval_minutes)
                self.time_slots.append(slot_time)
            
            # Initialize load curve array
            self.load_curve = [0.0] * len(self.time_slots)
            
            # Reset performance metrics
            self.peak_demand = 0.0
            self.total_energy_delivered = 0.0
        except Exception as e:
            logger.error(f"Error initializing simulation: {str(e)}")
            raise

    def _update_priorities(self, current_time: datetime) -> None:
        """Update priorities for vehicles in the waiting queue."""
        if not self.waiting_queue:
            return
            
        try:
            # Create new queue with updated priorities
            updated_queue = []
            while self.waiting_queue:
                _, request = heappop(self.waiting_queue)
                request.calculate_priority(current_time)
                heappush(updated_queue, (-request.priority, request))
                
            self.waiting_queue = updated_queue
        except Exception as e:
            logger.error(f"Error updating priorities: {str(e)}")

    def _find_available_charger(self, current_time: datetime) -> Optional[int]:
            """Find first available charger with at least 15-minute buffer."""
            # First try to find chargers with sufficient buffer
            best_charger = None
            max_buffer_time = 0

            for charger_id, vehicle_id in self.charger_assignments.items():
                if vehicle_id is None:
                    last_use_time = self.charger_last_use[charger_id]
                    if last_use_time is None:
                        return charger_id  # Return immediately if never used
                    
                    # Calculate buffer time
                    time_since_last_use = (current_time - last_use_time).total_seconds() / 60
                    
                    if time_since_last_use >= 15:  # Required minimum buffer
                        if time_since_last_use > max_buffer_time:
                            max_buffer_time = time_since_last_use
                            best_charger = charger_id
                    else:
                        logger.debug(f"Charger {charger_id} available but only has {time_since_last_use:.1f} minute buffer")
            
            return best_charger

    def _start_charging_session(
        self, 
        request: ChargingRequest, 
        charger_id: int, 
        current_time: datetime
    ) -> bool:
        """Initialize a new charging session and assign it to a charger."""
        try:
            # Validate request times before creating session
            if not request.validate_times():
                logger.error(f"Invalid request times for vehicle {request.vehicle_id}")
                return False

            # Check minimum buffer time
            last_use_time = self.charger_last_use[charger_id]
            if last_use_time is not None:
                buffer_time = (current_time - last_use_time).total_seconds() / 60
                if buffer_time < 15:
                    logger.debug(f"Cannot start session for vehicle {request.vehicle_id} - "
                            f"Minimum buffer time not met ({buffer_time:.1f} minutes)")
                    return False
                else:
                    logger.debug(f"Starting session for vehicle {request.vehicle_id} after "
                            f"{buffer_time:.1f} minute buffer")

            # Create session with depot return and next departure times
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
                last_vehicle_end_time=last_use_time,
                base_departure=request.base_departure,  # Vehicle's next departure time
                base_return=request.base_return        # Vehicle's current return time
            )

            # Log creation of session with times
            logger.debug(
                f"Creating charging session for vehicle {request.vehicle_id}: "
                f"Return: {request.base_return.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"Next Departure: {request.base_departure.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Validate session before adding
            if not session.validate():
                logger.error(f"Invalid session data for vehicle {request.vehicle_id}")
                return False
            
            self.active_sessions[request.vehicle_id] = session
            self.charger_assignments[charger_id] = request.vehicle_id
            return True
            
        except Exception as e:
            logger.error(f"Error starting charging session for vehicle {request.vehicle_id}: {str(e)}")
            return False

    def add_charging_request(self, request: ChargingRequest, current_time: datetime) -> None:
        """Add a new charging request to the system."""
        try:
            if request.vehicle_id in self.known_vehicles:
                return
            
            # Initialize simulation if this is the first request
            if self.simulation_start is None:
                self.initialize_simulation(current_time)
            
            # Validate request timing
            if not request.validate_times():
                logger.error(f"Invalid request times for vehicle {request.vehicle_id}")
                return
            
            request.start_time = round_to_interval(request.start_time, self.interval_minutes)
            request.target_end_time = round_to_interval(request.target_end_time, self.interval_minutes)
            
            # Calculate initial priority
            request.calculate_priority(current_time)
            
            # Try to find an available charger
            available_charger = self._find_available_charger(current_time)
            
            if available_charger is not None:
                if self._start_charging_session(request, available_charger, current_time):
                    self.known_vehicles.add(request.vehicle_id)
            else:
                heappush(self.waiting_queue, (-request.priority, request))
                self.known_vehicles.add(request.vehicle_id)
                
        except Exception as e:
            logger.error(f"Error adding charging request for vehicle {request.vehicle_id}: {str(e)}")

    def _should_delay_charging(self, session: ChargingSession, current_time: datetime) -> bool:
        """Determine if charging should be delayed based on departure constraints."""
        try:
            # Calculate time remaining until actual departure
            time_to_departure = (session.base_departure - current_time).total_seconds() / 3600
            total_window = (session.base_departure - session.base_return).total_seconds() / 3600
            
            # Calculate minimum charging time needed
            remaining_energy = session.energy_needed - session.energy_delivered
            min_charging_time = remaining_energy / session.max_charging_rate
            
            # Calculate safety margins based on departure time
            safety_margin = min(1.0, min_charging_time / time_to_departure)
            
            # Never delay if too close to departure
            if time_to_departure < (min_charging_time * 1.2):  # Need at least 20% buffer
                return False
                
            # Calculate dynamic delay threshold based on time to departure
            base_threshold = 0.3
            if time_to_departure < 6:  # Within 6 hours of departure
                base_threshold = 0.1  # Much less likely to delay
            elif time_to_departure < 12:  # Within 12 hours
                base_threshold = 0.2  # Somewhat less likely to delay
                
            # Calculate current demand
            current_demand = sum(s.power_profile[-1] if s.power_profile else 0 
                            for s in self.active_sessions.values())
            
            # Only delay if we have sufficient time and current demand is high
            demand_threshold = (base_threshold + 0.6 * (1.0 - safety_margin)) * self.demand_limit
            
            return current_demand > demand_threshold and time_to_departure > (min_charging_time * 2.0)
            
        except Exception as e:
            logger.error(f"Error in should_delay_charging: {str(e)}")
            return False

    def _calculate_power_allocation(self, sessions: List[ChargingSession], current_time: datetime) -> Dict[str, float]:
        """Calculate optimal power allocation for all active charging sessions."""
        if not sessions:
            return {}
        
        try:
            # Calculate session needs and priorities
            session_needs = []
            for session in sessions:
                if session.current_soc >= session.target_soc:
                    continue

                # Calculate remaining energy needed
                energy_remaining = session.energy_needed - session.energy_delivered
                
                # Get departure time - should be base_departure since that's when vehicle needs to leave
                departure_time = session.base_departure
                if departure_time is None:
                    departure_time = session.target_end_time

                # Calculate time remaining until departure
                time_to_departure = max(0.1, (departure_time - current_time).total_seconds() / 3600)
                
                # Calculate minimum power needed to reach target
                min_power_needed = energy_remaining / time_to_departure
                
                # If time is very short, increase power allocation priority
                urgency_multiplier = 1.0
                if time_to_departure < 1.0:  # Less than 1 hour
                    urgency_multiplier = 3.0
                elif time_to_departure < 2.0:  # Less than 2 hours
                    urgency_multiplier = 2.0
                
                # Calculate optimal power with safety margin
                optimal_power = min(
                    session.max_charging_rate,
                    min_power_needed * 1.2  # 20% safety margin
                )
                
                session_needs.append({
                    'session': session,
                    'min_power': min_power_needed * urgency_multiplier,
                    'optimal_power': optimal_power,
                    'time_to_departure': time_to_departure,
                    'energy_remaining': energy_remaining,
                    'urgency': urgency_multiplier
                })

            # Sort by urgency and time to departure
            session_needs.sort(key=lambda x: (-x['urgency'], x['time_to_departure']))

            # Initialize power allocation
            power_allocation = {session.vehicle_id: 0.0 for session in sessions}
            remaining_demand = self.demand_limit

            # First pass: Allocate minimum required power to all sessions
            for need in session_needs:
                if remaining_demand <= 0:
                    break

                session = need['session']
                required_power = min(need['min_power'], session.max_charging_rate)
                allocated_power = min(required_power, remaining_demand)
                
                if allocated_power > 0:
                    power_allocation[session.vehicle_id] = allocated_power
                    remaining_demand -= allocated_power

            # Second pass: Distribute remaining power to maximize charging rates
            if remaining_demand > 0:
                for need in session_needs:
                    if remaining_demand <= 0:
                        break

                    session = need['session']
                    current_power = power_allocation[session.vehicle_id]
                    max_additional = min(
                        session.max_charging_rate - current_power,
                        remaining_demand
                    )
                    
                    if max_additional > 0:
                        power_allocation[session.vehicle_id] += max_additional
                        remaining_demand -= max_additional

            # Log power allocations for debugging
            logger.debug(f"Power allocations at {current_time}:")
            for vid, power in power_allocation.items():
                if power > 0:
                    logger.debug(f"Vehicle {vid}: {power:.1f}kW")

            return power_allocation
            
        except Exception as e:
            logger.error(f"Error calculating power allocation: {str(e)}")
            return {session.vehicle_id: 0.0 for session in sessions}

    def _check_session_completion(self, session: ChargingSession, current_time: datetime) -> bool:
        """Check if a charging session is complete."""
        try:
            # Track total vehicles
            if session.vehicle_id not in self.known_vehicles:
                self.charging_failures["total_vehicles"] += 1
                self.known_vehicles.add(session.vehicle_id)

            # Check if target SOC reached
            if session.current_soc >= session.target_soc:
                session.completion_time = current_time
                self.completed_sessions.append(session)
                charger_id = next(cid for cid, vid in self.charger_assignments.items() 
                                if vid == session.vehicle_id)
                
                # Update last use time for the charger
                self.charger_last_use[charger_id] = current_time
                self.charger_assignments[charger_id] = None
                
                logger.info(f"Vehicle {session.vehicle_id} completed charging at {session.current_soc:.1f}% SOC")
                return True
            
            # Check if we've passed the departure time
            # Use target_end_time as fallback if base_departure isn't set
            departure_time = session.target_end_time  # Default to target_end_time
            if session.base_departure is not None:
                departure_time = session.base_departure

            if current_time >= departure_time:
                if session.current_soc < session.target_soc:
                    self.charging_failures["missed_departure"] += 1
                    logger.warning(f"Vehicle {session.vehicle_id} not charged before departure time")
                session.completion_time = current_time
                self.completed_sessions.append(session)
                charger_id = next(cid for cid, vid in self.charger_assignments.items() 
                                if vid == session.vehicle_id)
                self.charger_last_use[charger_id] = current_time
                self.charger_assignments[charger_id] = None
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking session completion for vehicle {session.vehicle_id}: {str(e)}")
            return False

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
            while self.waiting_queue:
                available_charger = self._find_available_charger(current_time)
                if available_charger is None:
                    break
                    
                _, request = heappop(self.waiting_queue)
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
            
            # Check for timed out vehicles
            self._handle_timed_out_vehicles(current_time)
            
            return {
                'current_load': current_demand,
                'waiting_vehicles': [req.vehicle_id for _, req in self.waiting_queue],
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

    def _handle_timed_out_vehicles(self, current_time: datetime) -> None:
        """Handle vehicles that are stuck or exceeded their time window."""
        try:
            to_remove = []
            for vehicle_id, session in self.active_sessions.items():
                # Track total vehicles
                if vehicle_id not in self.known_vehicles:
                    self.charging_failures["total_vehicles"] += 1
                    self.known_vehicles.add(vehicle_id)
                
                # Check completion conditions
                if session.current_soc >= session.target_soc:
                    logger.info(f"Vehicle {vehicle_id} completed charging at {session.current_soc:.1f}% SOC")
                    session.completion_time = current_time
                    self.completed_sessions.append(session)
                    to_remove.append(vehicle_id)
                    continue
                
                # Check for vehicles that didn't reach target SOC
                if all(end < current_time for _, end in session.availability):
                    logger.warning(f"Vehicle {vehicle_id} out of time at {session.current_soc:.1f}% SOC (Target: {session.target_soc}%)")
                    self.charging_failures["missed_soc"] += 1
                    session.completion_time = current_time
                    self.completed_sessions.append(session)
                    to_remove.append(vehicle_id)
                    
            # Remove completed sessions
            for vehicle_id in to_remove:
                del self.active_sessions[vehicle_id]
                charger_id = next((cid for cid, vid in self.charger_assignments.items() if vid == vehicle_id), None)
                if charger_id is not None:
                    # Update last use time for the charger
                    self.charger_last_use[charger_id] = current_time
                    self.charger_assignments[charger_id] = None
        except Exception as e:
            logger.error(f"Error handling timed out vehicles: {str(e)}")

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
            "charging_failures": {
                "missed_soc": self.charging_failures["missed_soc"],
                "missed_departure": self.charging_failures["missed_departure"],
                "total_vehicles": self.charging_failures["total_vehicles"],
                "failure_rate": (
                    (self.charging_failures["missed_soc"] + self.charging_failures["missed_departure"]) /
                    max(1, self.charging_failures["total_vehicles"]) * 100
                )
            },
            "vehicle_summaries": [],
            "per_vehicle_costs": per_vehicle_costs,
            "cost_breakdown": dict(cost_breakdown)
        }

        # Add vehicle summaries
        for session in self.completed_sessions + list(self.active_sessions.values()):
            charging_duration = None
            avg_charging_rate = 0
            
            if session.charging_start and session.completion_time:
                charging_duration = (session.completion_time - session.charging_start).total_seconds() / 3600
                if charging_duration > 0:
                    avg_charging_rate = min(
                        session.energy_delivered / charging_duration,
                        session.max_charging_rate,
                        self.max_charger_power
                    )

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

