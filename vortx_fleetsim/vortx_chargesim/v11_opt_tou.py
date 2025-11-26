# v11_opt_tou.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, NamedTuple
import numpy as np
import logging
import traceback
from heapq import heappush, heappop
from collections import defaultdict
import pandas as pd
from copy import deepcopy

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
logging.basicConfig(level=logging.DEBUG)
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

    def calculate_priority(self, current_time: datetime) -> float:
        """
        Calculate priority score based on energy needs, time constraints, and rates 
        across the full charging window.
        """
        try:
            # Get full charging window
            arrival_time = self.base_return or self.start_time
            departure_time = self.base_departure or self.target_end_time
            
            # Ensure we're not looking before current time
            start_analysis = max(arrival_time, current_time)
            
            # Calculate time metrics
            time_to_departure = (departure_time - current_time).total_seconds() / 3600
            total_window_hours = (departure_time - start_analysis).total_seconds() / 3600
            
            # Calculate minimum charging time needed
            min_charging_time = self.energy_needed / self.max_charging_rate
            
            # Calculate base urgency factor
            if time_to_departure <= min_charging_time:
                urgency_factor = 1.0  # Most urgent
            elif time_to_departure <= (min_charging_time * 1.5):
                urgency_factor = 0.8  # Very urgent
            elif time_to_departure <= (min_charging_time * 2.0):
                urgency_factor = 0.6  # Moderately urgent
            else:
                urgency_factor = 0.4  # Less urgent
                
            # Analyze rates across the window
            current_price = self._get_energy_price(current_time)
            lowest_price = current_price
            check_time = start_analysis
            
            while check_time < departure_time:
                rate = self._get_energy_price(check_time)
                lowest_price = min(lowest_price, rate)
                check_time += timedelta(minutes=self.interval_minutes)
            
            # Calculate rate advantage factor
            price_ratio = current_price / lowest_price if lowest_price > 0 else 1.0
            rate_factor = 1.0 / price_ratio  # Higher priority for better current rates
            
            # Calculate energy needs factor
            energy_factor = self.energy_needed / (self.battery_capacity * (self.target_soc - self.initial_soc) / 100)
            
            # Window flexibility factor - lower priority if we have more flexibility
            flexibility_factor = min_charging_time / total_window_hours if total_window_hours > 0 else 1.0
            
            # Combine factors with weights
            priority = (
                0.4 * urgency_factor +     # Time urgency
                0.3 * rate_factor +        # Rate optimization
                0.2 * energy_factor +      # Energy needs
                0.1 * flexibility_factor   # Window flexibility
            )
            
            logger.debug(f"Priority calculation for vehicle {self.vehicle_id}:"
                    f"\n  Time to departure: {time_to_departure:.1f}h"
                    f"\n  Min charging time: {min_charging_time:.1f}h"
                    f"\n  Current rate: ${current_price:.4f}"
                    f"\n  Best available rate: ${lowest_price:.4f}"
                    f"\n  Urgency factor: {urgency_factor:.2f}"
                    f"\n  Rate factor: {rate_factor:.2f}"
                    f"\n  Energy factor: {energy_factor:.2f}"
                    f"\n  Flexibility factor: {flexibility_factor:.2f}"
                    f"\n  Final priority: {priority:.2f}")
            
            return priority
            
        except Exception as e:
            logger.error(f"Error calculating priority for vehicle {self.vehicle_id}: {str(e)}")
            return 0.0
        

class OptimizationResult(NamedTuple):
    """Stores the result of a single optimization pass."""
    power_profile: Dict[str, List[float]]  # Vehicle ID to power profile mapping
    completion_times: Dict[str, datetime]   # Vehicle ID to completion time mapping
    peak_demand: float
    total_energy: float
    total_cost: float
    score: float  # Overall optimization score

class ChargingOptimizerTOU:
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
        self.interval_minutes = interval_minutes
        self.simulation_hours = simulation_hours
        
        # Load energy prices if not provided
        self.energy_prices = energy_prices or load_bev_energy_prices()

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
        self.total_cost: float = 0.0
        self.load_curve: List[float] = []

        # Initialize charging failures
        self.charging_failures = {
            "missed_soc": 0,
            "missed_departure": 0,
            "total_vehicles": 0
        }

    def _get_energy_price(self, time: datetime) -> float:
        """Get the energy price for a given time."""
        try:
            hour = time.hour
            # Find the matching time period and return its price
            for (start_hour, end_hour), price in self.energy_prices.items():
                if start_hour <= hour < end_hour:
                    return price
            # Default to off-peak rate if no matching period found
            return self.energy_prices.get((21, 24), 0.18462)  # Default to off-peak rate
        except Exception as e:
            logger.error(f"Error getting energy price: {str(e)}")
            return 0.18462  # Default to off-peak rate if there's an error

    def _calculate_charging_cost(self, power: float, time: datetime) -> float:
        """Calculate the cost of charging at a given power level and time."""
        energy = power * (self.interval_minutes / 60)  # Convert to hours
        rate = self._get_energy_price(time)
        return energy * rate

    def _calculate_optimization_score(
        self,
        peak_demand: float,
        total_energy: float,
        total_cost: float,
        completion_times: Dict[str, datetime],
        requests: List[ChargingRequest]
    ) -> float:
        """Calculate a score for the optimization result with cost consideration."""
        try:
            # Calculate demand score (lower is better)
            demand_score = 1.0 - (peak_demand / self.demand_limit)
            
            # Calculate cost score (lower is better)
            min_rate = min(self.energy_prices.values())
            theoretical_min_cost = total_energy * min_rate
            cost_score = theoretical_min_cost / max(total_cost, theoretical_min_cost)
            
            # Calculate timing score - higher weight for meeting deadlines
            timing_scores = []
            for request in requests:
                completion_time = completion_times.get(request.vehicle_id)
                if completion_time:
                    time_to_deadline = (request.target_end_time - completion_time).total_seconds() / 3600
                    timing_score = 1.0 / (1.0 + abs(time_to_deadline))
                    if abs(time_to_deadline) < 1.0:
                        timing_score *= 2.0  # Increased bonus for meeting deadlines
                    timing_scores.append(timing_score)
            
            avg_timing_score = sum(timing_scores) / len(timing_scores) if timing_scores else 0.0
            
            # Combine scores with weights - prioritize timing over cost
            final_score = (
                0.05 * demand_score +    # Small weight for demand management
                0.35 * cost_score +      # Medium weight for cost optimization
                0.60 * avg_timing_score  # Highest weight for timing optimization
            )
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating optimization score: {str(e)}")
            return 0.0

    def _should_delay_initial_charging(
        self, 
        request: ChargingRequest, 
        current_time: datetime
    ) -> bool:
        """
        Determine if we should delay starting the initial charging session by analyzing
        the entire available charging window from arrival to departure.
        """
        try:
            # Get full charging window
            arrival_time = request.base_return or request.start_time
            departure_time = request.base_departure or request.target_end_time
            
            # Ensure we're not looking before current time
            start_analysis = max(arrival_time, current_time)
            
            # Calculate total available window
            total_window_hours = (departure_time - start_analysis).total_seconds() / 3600
            
            # Calculate minimum charging time needed with safety margin
            min_charging_time = (request.energy_needed / request.max_charging_rate) * 1.2  # 20% safety margin
            
            # If we don't have enough time, start immediately
            if total_window_hours < min_charging_time:
                logger.debug(f"Vehicle {request.vehicle_id} must start now: "
                        f"window {total_window_hours:.1f}h < needed {min_charging_time:.1f}h")
                return False
                
            # Analyze entire window in interval steps
            check_time = start_analysis
            window_rates = []  # [(time, rate)]
            current_price = self._get_energy_price(current_time)
            lowest_price = current_price
            best_time = current_time
            
            while check_time < departure_time:
                rate = self._get_energy_price(check_time)
                window_rates.append((check_time, rate))
                if rate < lowest_price:
                    # Found a better rate
                    lowest_price = rate
                    best_time = check_time
                check_time += timedelta(minutes=self.interval_minutes)
                
            if not window_rates:
                return False
                
            # Calculate how long we could wait and still complete charging
            latest_start = departure_time - timedelta(hours=min_charging_time)
            time_until_best = (best_time - current_time).total_seconds() / 3600
            price_ratio = current_price / lowest_price if lowest_price > 0 else 1.0
            
            # Logging for visibility
            logger.debug(f"Vehicle {request.vehicle_id} analysis:"
                    f"\n  Window: {total_window_hours:.1f}h (from {start_analysis} to {departure_time})"
                    f"\n  Current rate: ${current_price:.4f}"
                    f"\n  Best rate: ${lowest_price:.4f} at {best_time}"
                    f"\n  Price ratio: {price_ratio:.2f}"
                    f"\n  Required charging time: {min_charging_time:.1f}h"
                    f"\n  Time until best rate: {time_until_best:.1f}h"
                    f"\n  Latest possible start: {latest_start}")
            
            # Decision logic
            if best_time <= latest_start:  # Can we wait for the better rate?
                savings_threshold = 1.05  # 5% savings minimum
                
                if price_ratio > savings_threshold:
                    logger.info(f"Delaying vehicle {request.vehicle_id} charging:"
                            f" {time_until_best:.1f}h for {((price_ratio-1)*100):.1f}% savings")
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error in should_delay_initial_charging: {str(e)}")
            return False


    def _calculate_power_allocation(self, sessions: List[ChargingSession], current_time: datetime) -> Dict[str, float]:
        """Calculate optimal power allocation considering TOU rates."""
        if not sessions:
            return {}
        
        try:
            current_price = self._get_energy_price(current_time)
            session_needs = []
            
            # Check if we're in peak period
            hour = current_time.hour
            is_peak_period = 16 <= hour < 21
            
            # First pass: Calculate minimum required power for each session
            total_min_power_needed = 0
            for session in sessions:
                if session.current_soc >= session.target_soc:
                    continue

                # Calculate remaining energy needed
                energy_remaining = session.energy_needed - session.energy_delivered
                departure_time = session.base_departure or session.target_end_time
                time_to_departure = max(0.1, (departure_time - current_time).total_seconds() / 3600)
                
                # Calculate minimum power needed to meet deadline
                min_power_needed = energy_remaining / time_to_departure
                min_charging_time = energy_remaining / session.max_charging_rate
                
                # Check if we can meet the deadline with max charging rate
                if min_power_needed > session.max_charging_rate:
                    logger.warning(f"Cannot meet deadline for vehicle {session.vehicle_id}: "
                                 f"needs {min_power_needed:.1f}kW but max is {session.max_charging_rate:.1f}kW")
                    min_power_needed = session.max_charging_rate
                
                total_min_power_needed += min_power_needed
                
                # Check if there's a cheaper period available
                has_cheaper_period = False
                check_time = current_time
                while check_time < departure_time:
                    check_price = self._get_energy_price(check_time)
                    if check_price < current_price:
                        has_cheaper_period = True
                        break
                    check_time += timedelta(hours=1)
                
                # Calculate urgency factor
                urgency = 1.0
                if time_to_departure <= min_charging_time:  # Must charge now
                    urgency = 4.0
                elif time_to_departure <= min_charging_time * 1.5:  # Very urgent
                    urgency = 3.0
                elif time_to_departure <= min_charging_time * 2.0:  # Somewhat urgent
                    urgency = 2.0
                
                session_needs.append({
                    'session': session,
                    'min_power': min_power_needed,
                    'time_to_departure': time_to_departure,
                    'energy_remaining': energy_remaining,
                    'min_charging_time': min_charging_time,
                    'has_cheaper_period': has_cheaper_period,
                    'urgency': urgency
                })

            # If we can't meet minimum power requirements, allocate proportionally
            if total_min_power_needed > self.demand_limit:
                logger.warning(f"Cannot meet charging requirements: need {total_min_power_needed:.1f}kW "
                             f"but limit is {self.demand_limit:.1f}kW")
                scale_factor = self.demand_limit / total_min_power_needed
                return {
                    need['session'].vehicle_id: need['min_power'] * scale_factor
                    for need in session_needs
                }

            # Sort by urgency first, then time to departure
            session_needs.sort(key=lambda x: (-x['urgency'], x['time_to_departure']))

            # Initialize power allocation
            power_allocation = {session.vehicle_id: 0.0 for session in sessions}
            remaining_demand = self.demand_limit

            # First pass: Allocate minimum required power to meet deadlines
            for need in session_needs:
                session = need['session']
                # Always allocate minimum required power to meet deadline
                allocated_power = min(need['min_power'], remaining_demand)
                if allocated_power > 0:
                    power_allocation[session.vehicle_id] = allocated_power
                    remaining_demand -= allocated_power

            # Second pass: If we have remaining power and not in peak period,
            # try to charge more to avoid future peaks
            if remaining_demand > 0 and not is_peak_period:
                # Sort by energy remaining (most energy first)
                session_needs.sort(key=lambda x: (-x['energy_remaining'], x['time_to_departure']))
                
                for need in session_needs:
                    if remaining_demand <= 0:
                        break

                    session = need['session']
                    current_power = power_allocation[session.vehicle_id]
                    
                    # Skip if in peak period and has cheaper period available
                    if is_peak_period and need['has_cheaper_period'] and need['urgency'] < 2.0:
                        continue
                    
                    # Calculate additional power that would help reduce future peaks
                    max_additional = min(
                        session.max_charging_rate - current_power,
                        remaining_demand
                    )
                    
                    # Only allocate additional power if:
                    # 1. Not in peak period, or
                    # 2. No cheaper period available, or
                    # 3. Urgent charging needed
                    if max_additional > 0 and (not is_peak_period or not need['has_cheaper_period'] or need['urgency'] >= 2.0):
                        power_allocation[session.vehicle_id] += max_additional
                        remaining_demand -= max_additional

            return power_allocation
            
        except Exception as e:
            logger.error(f"Error calculating power allocation: {str(e)}")
            return {session.vehicle_id: 0.0 for session in sessions}

    def simulate_time_step(self, current_time: datetime) -> Dict:
        """Simulate one time step of the charging system."""
        try:
            current_time = round_to_interval(current_time, self.interval_minutes)
            
            # Core simulation logic (similar to demand optimizer)
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
            
            # Track demand and costs
            current_demand = 0
            current_cost = 0
            
            # Update sessions with allocated power
            for vehicle_id, power in power_allocation.items():
                session = self.active_sessions[vehicle_id]
                
                # Calculate energy and cost
                hours = self.interval_minutes / 60
                energy_delivered = power * hours
                charging_cost = self._calculate_charging_cost(power, current_time)
                
                # Update session state
                session.energy_delivered += energy_delivered
                session.current_soc += (energy_delivered / session.battery_capacity * 100)
                session.power_profile.append(power)
                
                # Track metrics
                current_demand += power
                current_cost += charging_cost
                self.total_energy_delivered += energy_delivered
                self.total_cost += charging_cost
            
            # Update load curve and peak demand
            time_slot_index = self._get_time_slot_index(current_time)
            if time_slot_index is not None and time_slot_index < len(self.load_curve):
                self.load_curve[time_slot_index] = current_demand
                self.peak_demand = max(self.peak_demand, current_demand)
            
            return {
                'current_load': current_demand,
                'current_cost': current_cost,
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
                'current_cost': 0,
                'waiting_vehicles': [],
                'active_sessions': []
            }

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
            
            # Initialize load curve and cost tracking
            self.load_curve = [0.0] * len(self.time_slots)
            
            # Reset performance metrics
            self.peak_demand = 0.0
            self.total_energy_delivered = 0.0
            self.total_cost = 0.0
            
        except Exception as e:
            logger.error(f"Error initializing simulation: {str(e)}")
            raise

    def _update_priorities(self, current_time: datetime) -> None:
        """Update priorities for vehicles in the waiting queue considering TOU rates."""
        if not self.waiting_queue:
            return
            
        try:
            # Get current energy price and minimum price
            current_price = self._get_energy_price(current_time)
            min_price = min(self.energy_prices.values())
            
            # Check if we're in peak period
            hour = current_time.hour
            is_peak_period = 16 <= hour < 21
            
            # Create new queue with updated priorities
            updated_queue = []
            while self.waiting_queue:
                _, request = heappop(self.waiting_queue)
                
                # Calculate base priority from deadline and energy needs
                base_priority = request.calculate_priority(current_time)
                
                # Calculate time to departure
                time_to_departure = (request.base_departure - current_time).total_seconds() / 3600
                
                # Calculate minimum charging time needed
                energy_needed = request.energy_needed - 0  # Assuming no energy delivered yet
                min_charging_time = energy_needed / request.max_charging_rate
                
                # Check if there's a cheaper period available before departure
                has_cheaper_period = False
                check_time = current_time
                while check_time < request.base_departure:
                    check_price = self._get_energy_price(check_time)
                    if check_price < current_price:
                        has_cheaper_period = True
                        break
                    check_time += timedelta(hours=1)
                
                # Calculate price factor with stronger TOU influence
                price_factor = (min_price / current_price) ** 3  # Exponential scaling
                
                # Strongly discourage charging during peak periods if we have time and cheaper periods
                if is_peak_period and time_to_departure > min_charging_time * 1.5 and has_cheaper_period:
                    price_factor *= 0.1  # Severely reduce priority during peak
                
                # Boost priority for urgent charging regardless of price
                if time_to_departure < min_charging_time * 1.2:
                    price_factor = 1.0  # Ignore price for urgent charging
                
                # Calculate final priority
                adjusted_priority = base_priority * price_factor
                
                heappush(updated_queue, (-adjusted_priority, request))
                
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
            # Validate request times
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
                last_vehicle_end_time=last_use_time,
                base_departure=request.base_departure,
                base_return=request.base_return
            )

            # Validate session
            if not session.validate():
                logger.error(f"Invalid session data for vehicle {request.vehicle_id}")
                return False
            
            self.active_sessions[request.vehicle_id] = session
            self.charger_assignments[charger_id] = request.vehicle_id
            
            logger.info(f"Started charging session for vehicle {request.vehicle_id} at {current_time}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting charging session for vehicle {request.vehicle_id}: {str(e)}")
            return False

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
                
                self.charger_last_use[charger_id] = current_time
                self.charger_assignments[charger_id] = None
                
                logger.info(f"Vehicle {session.vehicle_id} completed charging at {session.current_soc:.1f}% SOC")
                return True
            
            # Check if we've passed the departure time
            departure_time = session.base_departure or session.target_end_time
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

    def add_charging_request(self, request: ChargingRequest, current_time: datetime) -> None:
        """Add a new charging request to the system with potential initial delay."""
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
            
            # Calculate initial priority
            request.calculate_priority(current_time)
            
            # Check if we should delay initial charging
            if self._should_delay_initial_charging(request, current_time):
                logger.info(f"Delaying initial charging for vehicle {request.vehicle_id} due to rate optimization")
                heappush(self.waiting_queue, (-request.priority, request))
                self.known_vehicles.add(request.vehicle_id)
                return
            
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

    def get_results(self) -> dict:
        """Get final simulation results including TOU-specific metrics."""
        try:
            # Calculate base metrics
            results = {
                "completed_sessions": len(self.completed_sessions),
                "peak_demand": self.peak_demand,
                "total_energy_delivered": self.total_energy_delivered,
                "total_cost": self.total_cost,
                "charging_failures": self.charging_failures.copy(),
                "cost_breakdown": {},
                "vehicle_summaries": [],
                "per_vehicle_costs": {}  # Add per-vehicle costs
            }

            # Calculate cost breakdown by TOU period
            period_costs = defaultdict(float)
            period_energy = defaultdict(float)
            vehicle_costs = defaultdict(float)  # Track costs per vehicle

            for session in self.completed_sessions:
                if session.charging_start and session.completion_time:
                    current_time = session.charging_start
                    vehicle_id = session.vehicle_id
                    vehicle_session_cost = 0  # Track cost for this session

                    for i, power in enumerate(session.power_profile):
                        if power > 0:
                            # Get price for this hour
                            current_price = self._get_energy_price(current_time)
                            
                            # Calculate energy and cost
                            energy = power * (self.interval_minutes / 60)
                            cost = energy * current_price
                            
                            # Find the period
                            hour = current_time.hour
                            period = next(
                                (f"{start:02d}-{end:02d}" for (start, end), _ in self.energy_prices.items() 
                                if start <= hour < end),
                                "unknown"
                            )
                            
                            period_energy[period] += energy
                            period_costs[period] += cost
                            vehicle_session_cost += cost  # Add to session cost
                            
                        current_time += timedelta(minutes=self.interval_minutes)

                    # Add session cost to vehicle total
                    vehicle_costs[vehicle_id] += vehicle_session_cost

            # Add cost breakdown to results
            results["cost_breakdown"] = {
                period: {
                    "energy_kwh": energy,
                    "cost": cost,
                    "rate": next(
                        (rate for (start, end), rate in self.energy_prices.items() 
                        if f"{start:02d}-{end:02d}" == period),
                        0.0
                    )
                }
                for period, (energy, cost) in zip(
                    period_energy.keys(),
                    zip(period_energy.values(), period_costs.values())
                )
            }

            # Add per-vehicle costs to results
            results["per_vehicle_costs"] = {
                vehicle_id: {
                    "total_cost": cost,
                    "avg_rate": cost / next(
                        (session.energy_delivered 
                        for session in self.completed_sessions 
                        if session.vehicle_id == vehicle_id), 
                        1  # Default to 1 to avoid division by zero
                    )
                }
                for vehicle_id, cost in vehicle_costs.items()
            }

            # Add detailed vehicle summaries
            for session in self.completed_sessions + list(self.active_sessions.values()):
                # Calculate charging duration and rates
                charging_duration = None
                avg_charging_rate = 0
                
                if session.charging_start and session.completion_time:
                    charging_duration = (
                        session.completion_time - session.charging_start
                    ).total_seconds() / 3600
                    
                    if charging_duration > 0:
                        avg_charging_rate = min(
                            session.energy_delivered / charging_duration,
                            session.max_charging_rate
                        )

                # Get vehicle cost from our calculated per-vehicle costs
                vehicle_cost = results["per_vehicle_costs"].get(session.vehicle_id, {}).get("total_cost", 0)

                summary = {
                    "vehicle_id": session.vehicle_id,
                    "energy_delivered": session.energy_delivered,
                    "charging_cost": vehicle_cost,
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

            # Log detailed results
            logger.info("\n=== Final Simulation Summary (TOU Optimizer) ===")
            logger.info(f"Total vehicles completed: {len(self.completed_sessions)}")
            logger.info(f"Total energy delivered: {self.total_energy_delivered:.2f} kWh")
            logger.info(f"Total charging cost: ${self.total_cost:.2f}")
            logger.info(f"Peak demand: {self.peak_demand:.2f} kW")
            
            # Log cost breakdown
            logger.info("\nCost breakdown by TOU period:")
            for period, data in results["cost_breakdown"].items():
                logger.info(f"{period}: {data['energy_kwh']:.2f} kWh @ "
                        f"${data['rate']:.4f}/kWh = ${data['cost']:.2f}")

            # Log per-vehicle costs
            logger.info("\nCost breakdown by vehicle:")
            for vehicle_id, data in results["per_vehicle_costs"].items():
                logger.info(f"{vehicle_id}: ${data['total_cost']:.2f} "
                        f"(${data['avg_rate']:.4f}/kWh)")

            return results
            
        except Exception as e:
            logger.error(f"Error getting results: {str(e)}")
            return {
                "error": str(e),
                "completed_sessions": 0,
                "peak_demand": 0,
                "total_energy_delivered": 0,
                "total_cost": 0,
                "charging_failures": self.charging_failures.copy(),
                "vehicle_summaries": []
            }
