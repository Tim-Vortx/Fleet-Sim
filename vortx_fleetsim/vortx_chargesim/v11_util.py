import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from collections import defaultdict
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Charging Parameters
CHARGER_POWER = 150  # kW
NUM_CHARGERS = 15
CHARGER_SWITCH_TIME = 15  # minutes
MINUTES_PER_DAY = 1440
INTERVAL_MINUTES = 5  # Changed from 15 to 5 minutes for optimization

# Vehicle Fleet Parameters
NUM_VEHICLES = 25
VEHICLE_TYPE = "Volvo VNR Electric"
BATTERY_CAPACITY_KWH = 452
EFFICIENCY_KWH_PER_MILE = 1.65
INITIAL_SOC_PERCENTAGE = 80

# File paths
RATE_FILE = os.path.join(os.path.dirname(__file__), "data", "bev_rates.csv")
CACHE_FILE = os.path.join("cache", "vehicle_data.csv")

# TOU Period Definitions
TOU_PERIODS = {
    'peak': (16, 21),  # 16:00-21:00
    'super_off_peak': (9, 14),  # 9:00-14:00
    'off_peak': [(0, 9), (14, 16), (21, 24)]  # All other hours
}

# Fleet Operation Hours
FLEET_OPERATION_HOURS = (8, 17)  # 8:00-17:00

@dataclass
class BaseChargingSession:
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

    def get_normalized_schedule(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get normalized schedule times that handle all charging scenarios."""
        if self.base_return is None or self.base_departure is None:
            return None, None

        return_ts = self.base_return.timestamp()
        departure_ts = self.base_departure.timestamp()
        
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
                    charging_window = departure_time - return_time
                    window_hours = charging_window.total_seconds() / 3600

                    if window_hours <= 0:
                        logger.error(f"Invalid charging window for vehicle {self.vehicle_id}")
                        return False
                    
                    if window_hours > 24:
                        logger.error(f"Invalid charging window duration")
                        return False

                    if charging_window < timedelta(minutes=30):
                        logger.error(f"Charging window too short")
                        return False

            return True
        except Exception as e:
            logger.error(f"Session validation error: {str(e)}")
            return False

@dataclass
class BaseChargingRequest:
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
                logger.error(f"Start time must be before end time")
                return False

            for start, end in self.availability:
                if not isinstance(start, datetime) or not isinstance(end, datetime):
                    logger.error(f"Invalid availability window format")
                    return False
                if start >= end:
                    logger.error(f"Invalid availability window (start >= end)")
                    return False

            return True
        except Exception as e:
            logger.error(f"Time validation error: {str(e)}")
            return False

def find_available_charger(
    charger_assignments: Dict[int, Optional[str]],
    charger_last_use: Dict[int, Optional[datetime]],
    current_time: datetime
) -> Optional[int]:
    """Find first available charger with sufficient buffer time."""
    best_charger = None
    max_buffer_time = 0

    for charger_id, vehicle_id in charger_assignments.items():
        if vehicle_id is None:
            last_use_time = charger_last_use[charger_id]
            if last_use_time is None:
                return charger_id
            
            buffer_time = (current_time - last_use_time).total_seconds() / 60
            if buffer_time >= 15:
                if buffer_time > max_buffer_time:
                    max_buffer_time = buffer_time
                    best_charger = charger_id
            
    return best_charger

def get_time_slot_index(current_time: datetime, simulation_start: datetime, interval_minutes: int) -> Optional[int]:
    """Get the index of the current time slot."""
    try:
        minutes_diff = int((current_time - simulation_start).total_seconds() / 60)
        index = minutes_diff // interval_minutes
        
        return index if index >= 0 else None
    except Exception as e:
        logger.error(f"Error getting time slot index: {str(e)}")
        return None

def check_session_completion(
    session: BaseChargingSession,
    current_time: datetime,
    charging_failures: Dict[str, int],
    known_vehicles: Set[str],
    charger_assignments: Dict[int, Optional[str]],
    charger_last_use: Dict[int, Optional[datetime]]
) -> bool:
    """Check if a charging session is complete."""
    try:
        if session.vehicle_id not in known_vehicles:
            charging_failures["total_vehicles"] += 1
            known_vehicles.add(session.vehicle_id)

        if session.current_soc >= session.target_soc:
            session.completion_time = current_time
            charger_id = next(cid for cid, vid in charger_assignments.items() 
                            if vid == session.vehicle_id)
            charger_last_use[charger_id] = current_time
            charger_assignments[charger_id] = None
            return True
        
        departure_time = session.base_departure or session.target_end_time
        if current_time >= departure_time:
            if session.current_soc < session.target_soc:
                charging_failures["missed_departure"] += 1
            session.completion_time = current_time
            charger_id = next(cid for cid, vid in charger_assignments.items() 
                            if vid == session.vehicle_id)
            charger_last_use[charger_id] = current_time
            charger_assignments[charger_id] = None
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking session completion: {str(e)}")
        return False

def get_period_for_hour(hour):
    """Determine which TOU period an hour belongs to."""
    if TOU_PERIODS['peak'][0] <= hour < TOU_PERIODS['peak'][1]:
        return 'peak'
    elif TOU_PERIODS['super_off_peak'][0] <= hour < TOU_PERIODS['super_off_peak'][1]:
        return 'super_off_peak'
    return 'off_peak'

def is_fleet_operating_hour(hour):
    """Check if the given hour is during fleet operation."""
    return FLEET_OPERATION_HOURS[0] <= hour < FLEET_OPERATION_HOURS[1]

def calculate_load_curve(schedule, time_slots):
    """Calculate the load curve from charging sessions with power profiles."""
    num_intervals = len(time_slots)
    load_curve = np.zeros(num_intervals)
    
    for session in schedule:
        try:
            start_time = pd.to_datetime(session['charging_start'])
            end_time = pd.to_datetime(session['charging_end']) if session['charging_end'] else None
            
            if 'Charging_Rates' in session and session['Charging_Rates']:
                power_profile = session['Charging_Rates']
                start_idx = (start_time - time_slots[0]).total_seconds() // (INTERVAL_MINUTES * 60)
                start_idx = int(start_idx)
                
                for i, power in enumerate(power_profile):
                    if start_idx + i < num_intervals:
                        load_curve[start_idx + i] += power
            else:
                charging_rate = session.get('Charging_Rate', 0)
                start_idx = (start_time - time_slots[0]).total_seconds() // (INTERVAL_MINUTES * 60)
                start_idx = int(start_idx)
                
                if end_time:
                    end_idx = (end_time - time_slots[0]).total_seconds() // (INTERVAL_MINUTES * 60)
                    end_idx = int(end_idx)
                else:
                    end_idx = len(time_slots)
                
                if start_idx < num_intervals and end_idx > 0:
                    load_curve[start_idx:end_idx] += charging_rate
            
        except Exception as e:
            logger.error(f"Error processing charging session: {e}")
            continue
    
    return load_curve

def round_to_interval(time, interval_minutes=5):
    """Round a datetime to the nearest interval."""
    minutes = time.hour * 60 + time.minute
    rounded_minutes = round(minutes / interval_minutes) * interval_minutes
    return pd.Timestamp(time.date()).replace(
        hour=rounded_minutes // 60,
        minute=rounded_minutes % 60
    )

def calculate_detailed_charging_costs(schedule, energy_prices):
    """Calculate detailed charging costs for the schedule."""
    # Initialize data structures
    per_vehicle = {}
    per_period = defaultdict(lambda: {'energy': 0.0, 'cost': 0.0, 'hours': 0.0})
    total = {
        'energy': 0.0,
        'cost': 0.0,
        'hours': 0.0,
        'avg_rate': 0.0
    }

    for session in schedule:
        vehicle_id = session['Vehicle_ID']
        start_time = pd.to_datetime(session['charging_start'])
        end_time = pd.to_datetime(session['charging_end']) if session['charging_end'] else start_time + pd.Timedelta(hours=1)
        
        # Skip sessions with no energy charged
        if not session.get('Charging_Rates') and not session.get('Charging_Rate'):
            continue
        
        # Handle overnight charging
        if end_time < start_time:
            end_time += pd.Timedelta(days=1)
        
        duration = (end_time - start_time).total_seconds() / 3600  # hours

        # Initialize vehicle data if not exists
        if vehicle_id not in per_vehicle:
            per_vehicle[vehicle_id] = {
                'energy': 0.0,
                'cost': 0.0,
                'hours': 0.0,
                'sessions': [],
                'avg_rate': 0.0
            }

        # Get charging rates (either variable or constant)
        if 'Charging_Rates' in session and session['Charging_Rates']:
            power_profile = session['Charging_Rates']
        else:
            # Create power profile with constant rate
            intervals = int(duration * 60 / INTERVAL_MINUTES)
            power_profile = [session['Charging_Rate']] * intervals

        # Calculate energy consumed and cost in each TOU period
        current_time = start_time
        session_energy = 0.0
        session_cost = 0.0
        
        for power in power_profile:
            # Calculate energy for this interval (same as optimizer)
            hours = INTERVAL_MINUTES / 60
            energy_delivered = power * hours
            
            # Get price for this hour
            hour = current_time.hour
            price = next(p for (start, end), p in energy_prices.items() 
                        if start <= hour < end)
            
            # Calculate cost for this interval (same as optimizer)
            interval_cost = energy_delivered * price

            # Determine rate period
            period = get_period_for_hour(hour)

            # Update period totals
            per_period[period]['energy'] += energy_delivered
            per_period[period]['cost'] += interval_cost
            per_period[period]['hours'] += hours

            # Update session totals
            session_energy += energy_delivered
            session_cost += interval_cost

            current_time += pd.Timedelta(minutes=INTERVAL_MINUTES)

        # Update vehicle totals
        per_vehicle[vehicle_id]['energy'] += session_energy
        per_vehicle[vehicle_id]['cost'] += session_cost
        per_vehicle[vehicle_id]['hours'] += duration
        per_vehicle[vehicle_id]['sessions'].append({
            'start': start_time.strftime('%Y-%m-%d %H:%M'),
            'end': end_time.strftime('%Y-%m-%d %H:%M'),
            'energy': session_energy,
            'cost': session_cost,
            'duration': duration,
            'avg_rate': session_cost / session_energy if session_energy > 0 else 0
        })

        # Update overall totals
        total['energy'] += session_energy
        total['cost'] += session_cost
        total['hours'] += duration

    # Calculate averages
    total['avg_rate'] = total['cost'] / total['energy'] if total['energy'] > 0 else 0
    for vehicle in per_vehicle.values():
        vehicle['avg_rate'] = vehicle['cost'] / vehicle['energy'] if vehicle['energy'] > 0 else 0

    # Add percentage breakdowns to period costs
    if total['energy'] > 0:
        for period in per_period:
            per_period[period]['energy_pct'] = (per_period[period]['energy'] / total['energy']) * 100
            per_period[period]['cost_pct'] = (per_period[period]['cost'] / total['cost']) * 100
            per_period[period]['avg_rate'] = (per_period[period]['cost'] / per_period[period]['energy'] 
                                            if per_period[period]['energy'] > 0 else 0)

    return {
        'per_vehicle': per_vehicle,
        'per_period': dict(per_period),  # Convert defaultdict to regular dict
        'total': total
    }

def calculate_session_cost(start_idx, charging_time, rate, time_slots, energy_prices):
    """Calculate the cost of a charging session."""
    total_cost = 0
    for k in range(charging_time):
        idx = (start_idx + k) % len(time_slots)
        time = time_slots[idx]
        hour = time.hour
        energy_price = next(price for (start, end), price in energy_prices.items() 
                          if start <= hour < end)
        energy_charged = rate / 60  # kWh per minute
        cost = energy_charged * energy_price
        total_cost += cost
    return total_cost

def calculate_charging_cost(start_time, end_time, energy, energy_prices):
    """Calculate the cost of a charging session."""
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time)
    
    # Handle overnight charging
    if end_time < start_time:
        end_time += pd.Timedelta(days=1)
    
    duration = (end_time - start_time).total_seconds() / 3600  # hours
    total_cost = 0
    current_time = start_time
    
    while current_time < end_time:
        hour = current_time.hour
        rate = next(price for (start, end), price in energy_prices.items() 
                   if start <= hour < end)
        
        next_hour = current_time.replace(minute=0, second=0) + timedelta(hours=1)
        interval_end = min(end_time, next_hour)
        
        interval_duration = (interval_end - current_time).total_seconds() / 3600
        energy_charged = (energy / duration) * interval_duration
        total_cost += energy_charged * rate
        
        current_time = interval_end
    
    return total_cost

def calculate_charger_utilization(schedule, num_chargers):
    """Calculate utilization statistics for chargers."""
    charger_utilization = {i: 0 for i in range(1, num_chargers + 1)}
    
    for session in schedule:
        charger = int(session['Charger_ID'].replace('C', ''))  # Extract number from 'C1', 'C2', etc.
        start_time = pd.to_datetime(session['charging_start'])
        end_time = pd.to_datetime(session['charging_end']) if session['charging_end'] else start_time + pd.Timedelta(hours=1)
        
        # Handle overnight charging
        if end_time < start_time:
            end_time += pd.Timedelta(days=1)
            
        duration = (end_time - start_time).total_seconds() / 3600
        charger_utilization[charger] += duration

    total_time = 24 * num_chargers
    total_used_time = sum(charger_utilization.values())
    total_utilization = (total_used_time / total_time) * 100

    for charger in charger_utilization:
        charger_utilization[charger] = min((charger_utilization[charger] / 24) * 100, 100)

    return charger_utilization, total_utilization

def correct_time_format(time_str):
    """Correct time format for 24:00."""
    if time_str.endswith('24:00'):
        date_part = time_str.split()[0]
        next_day = (pd.to_datetime(date_part) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        return f"{next_day} 00:00"
    return time_str

def generate_random_time(base_time):
    """Generate a random time within 2 hours of the base time."""
    base_hour, base_minute = map(int, base_time.split(":"))
    random_minute_offset = np.random.randint(-120, 121)
    random_time = pd.Timestamp(f"{base_hour:02d}:{base_minute:02d}") + pd.Timedelta(minutes=random_minute_offset)
    return random_time.strftime("%H:%M")

def generate_vehicle_df():
    """Generate a new vehicle dataframe with random routes."""
    # Generate vehicle IDs
    vehicle_ids = [f"Volvo_VNR_{i+1}" for i in range(NUM_VEHICLES)]

    # Set route ranges between 150 and 200 miles
    route_miles = np.random.randint(150, 201, size=NUM_VEHICLES)

    # Generate departure and return times
    departure_times = [generate_random_time("20:00") for _ in range(NUM_VEHICLES)]
    return_times = [generate_random_time("04:00") for _ in range(NUM_VEHICLES)]

    # Create the DataFrame
    vehicle_data = {
        "Vehicle_ID": vehicle_ids,
        "OEM": [VEHICLE_TYPE] * NUM_VEHICLES,
        "Battery_Capacity_kWh": [BATTERY_CAPACITY_KWH] * NUM_VEHICLES,
        "Efficiency_kWh_per_mile": [EFFICIENCY_KWH_PER_MILE] * NUM_VEHICLES,
        "Route_Miles": route_miles,
        "operations_start": departure_times,
        "operations_end": return_times
    }

    vehicle_df = pd.DataFrame(vehicle_data)

    # Set initial SOC and calculate energy metrics
    vehicle_df["Initial_SOC"] = INITIAL_SOC_PERCENTAGE
    vehicle_df["Initial_Energy_kWh"] = vehicle_df["Battery_Capacity_kWh"] * (vehicle_df["Initial_SOC"] / 100)
    vehicle_df["Energy_Consumed_kWh"] = vehicle_df["Efficiency_kWh_per_mile"] * vehicle_df["Route_Miles"]
    
    # Calculate remaining energy and SOC
    vehicle_df["Remaining_Energy_kWh"] = (vehicle_df["Initial_Energy_kWh"] - vehicle_df["Energy_Consumed_kWh"]).clip(lower=0)
    vehicle_df["Remaining_SOC"] = (vehicle_df["Remaining_Energy_kWh"] / vehicle_df["Battery_Capacity_kWh"] * 100).clip(lower=0, upper=100)
    
    # Set target SOC and calculate needed energy
    vehicle_df["Target_SOC"] = INITIAL_SOC_PERCENTAGE
    vehicle_df["Energy_Needed_kWh"] = ((vehicle_df["Target_SOC"] - vehicle_df["Remaining_SOC"]) / 100 * vehicle_df["Battery_Capacity_kWh"]).clip(lower=0)

    return vehicle_df

def load_or_generate_vehicle_data(data_choice='keep'):
    """
    Load existing vehicle data or generate new data based on user choice.

    Parameters:
    - data_choice (str): 'keep' to retain existing data, 'delete' to remove and generate new data.

    Returns:
    - vehicle_df (pd.DataFrame): DataFrame containing vehicle data.
    """
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

    if os.path.exists(CACHE_FILE):
        if data_choice == 'keep':
            print("Keeping existing vehicle data...")
            vehicle_df = pd.read_csv(CACHE_FILE)
        elif data_choice == 'delete':
            print("Deleting existing data and creating new vehicle data...")
            vehicle_df = generate_vehicle_df()
            vehicle_df.to_csv(CACHE_FILE, index=False)
        else:
            print("Invalid data_choice provided. Keeping existing vehicle data by default...")
            vehicle_df = pd.read_csv(CACHE_FILE)
    else:
        print("No cached data found. Generating new vehicle data and caching it...")
        vehicle_df = generate_vehicle_df()
        vehicle_df.to_csv(CACHE_FILE, index=False)

    # Convert time columns to datetime with explicit format
    for col in ['operations_end', 'operations_start']:
        if vehicle_df[col].dtype == 'object':
            vehicle_df[col] = pd.to_datetime(vehicle_df[col], format='%H:%M').dt.time

    print(f"Loaded {len(vehicle_df)} valid vehicle records.")
    return vehicle_df

def create_time_slots(base_date=None):
    """Create time slots for a full day with 5-minute intervals."""
    if base_date is None:
        base_date = datetime.now().date()
    elif isinstance(base_date, str):
        base_date = pd.to_datetime(base_date).date()
    
    return pd.date_range(
        start=base_date, 
        periods=MINUTES_PER_DAY // INTERVAL_MINUTES, 
        freq=f'{INTERVAL_MINUTES}min'
    )

def validate_input_data(vehicles_data, time_slots=None):
    """
    Validate input vehicle data for charging simulation.
    
    Args:
        vehicles_data (pd.DataFrame or list): Vehicle data to validate
        time_slots (list, optional): Time slots for additional validation. Defaults to None.
    
    Returns:
        tuple: (validated vehicles_data, time_slots)
    """
    # Convert list to DataFrame if needed
    if isinstance(vehicles_data, list):
        vehicles_data = pd.DataFrame(vehicles_data)
    
    # Check if DataFrame is empty
    if vehicles_data.empty:
        raise ValueError("Vehicle data is empty")
    
    # Flexible column mapping
    column_mapping = {
        'Route_Miles': ['Route_Miles', 'Route_Distance_miles', 'Range', 'route_miles'],
        'Battery_Capacity_kWh': ['Battery_Capacity_kWh', 'battery_capacity', 'Battery Capacity'],
        'Efficiency_kWh_per_mile': ['Efficiency_kWh_per_mile', 'efficiency', 'Efficiency'],
        'Vehicle_ID': ['Vehicle_ID', 'vehicle_id', 'ID'],
        'operations_start': ['operations_start', 'Departure_Time', 'departure_time'],
        'operations_end': ['operations_end', 'Return_Time', 'return_time']
    }
    
    # Rename columns to standard format
    for standard_col, alt_cols in column_mapping.items():
        for alt_col in alt_cols:
            if alt_col in vehicles_data.columns:
                vehicles_data.rename(columns={alt_col: standard_col}, inplace=True)
                break
    
    # Check if all required columns are present
    required_columns = [
        'Vehicle_ID', 
        'operations_start', 
        'operations_end', 
        'Battery_Capacity_kWh', 
        'Efficiency_kWh_per_mile', 
        'Route_Miles'
    ]
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in vehicles_data.columns]
    if missing_columns:
        # If Route_Miles is missing, try to generate it
        if 'Route_Miles' in missing_columns:
            # Use a default range or generate random route miles
            vehicles_data['Route_Miles'] = np.random.uniform(50, 200, len(vehicles_data))
            missing_columns.remove('Route_Miles')
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for valid data types and ranges
    if not all(vehicles_data['Battery_Capacity_kWh'] > 0):
        raise ValueError("Battery capacity must be positive")
    
    if not all(vehicles_data['Efficiency_kWh_per_mile'] > 0):
        raise ValueError("Energy efficiency must be positive")
    
    if not all(vehicles_data['Route_Miles'] >= 0):
        raise ValueError("Route miles cannot be negative")
    
    # If time_slots is not provided, create default time slots
    if time_slots is None:
        time_slots = create_time_slots()
    
    return vehicles_data, time_slots

def load_bev_energy_prices(rate_option=1):
    """
    Load BEV rates from the provided CSV based on the selected rate option.

    Parameters:
    - rate_option (int): 1 for BEV-1, 2 for BEV-2-S, 3 for BEV-2-P.

    Returns:
    - ENERGY_PRICES (dict): Dictionary with time intervals as keys and prices as values.
    """
    try:
        df = pd.read_csv(RATE_FILE)  # Use the absolute path from RATE_FILE
        rates = {
            'BEV-1': {
                'peak': df.loc[df['Rate Option'] == 'BEV-1', 'Peak Rate ($/kWh)'].values[0],
                'off_peak': df.loc[df['Rate Option'] == 'BEV-1', 'Off-Peak Rate ($/kWh)'].values[0],
                'super_off_peak': df.loc[df['Rate Option'] == 'BEV-1', 'Super Off-Peak Rate ($/kWh)'].values[0]
            },
            'BEV-2-S': {
                'peak': df.loc[df['Rate Option'] == 'BEV-2-S (Secondary)', 'Peak Rate ($/kWh)'].values[0],
                'off_peak': df.loc[df['Rate Option'] == 'BEV-2-S (Secondary)', 'Off-Peak Rate ($/kWh)'].values[0],
                'super_off_peak': df.loc[df['Rate Option'] == 'BEV-2-S (Secondary)', 'Super Off-Peak Rate ($/kWh)'].values[0]
            },
            'BEV-2-P': {
                'peak': df.loc[df['Rate Option'] == 'BEV-2-P (Primary / Transmission)', 'Peak Rate ($/kWh)'].values[0],
                'off_peak': df.loc[df['Rate Option'] == 'BEV-2-P (Primary / Transmission)', 'Off-Peak Rate ($/kWh)'].values[0],
                'super_off_peak': df.loc[df['Rate Option'] == 'BEV-2-P (Primary / Transmission)', 'Super Off-Peak Rate ($/kWh)'].values[0]
            }
        }

        selected_rate = {
            1: 'BEV-1',
            2: 'BEV-2-S',
            3: 'BEV-2-P'
        }.get(rate_option, 'BEV-1')  # Default to BEV-1 if invalid option

        selected_rates = rates[selected_rate]
        ENERGY_PRICES = {
            (0, 9): selected_rates['off_peak'],
            (9, 14): selected_rates['super_off_peak'],
            (14, 16): selected_rates['off_peak'],
            (16, 21): selected_rates['peak'],
            (21, 24): selected_rates['off_peak']
        }

        return ENERGY_PRICES

    except Exception as e:
        logger.error(f"Error loading rates: {e}")
        # Provide a hardcoded fallback energy price dictionary
        return {
            (0, 9): 0.18462,    # Off-peak
            (9, 14): 0.16135,   # Super off-peak
            (14, 16): 0.18462,  # Off-peak
            (16, 21): 0.39785,  # Peak
            (21, 24): 0.18462   # Off-peak
        }
