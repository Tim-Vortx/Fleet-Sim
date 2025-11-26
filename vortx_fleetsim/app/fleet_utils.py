from datetime import datetime, timedelta
import random
import pandas as pd
from typing import List, Dict, Optional, Tuple
from fleet_core_algos import VehicleSpecs

def generate_multi_day_schedule(
    base_date: datetime,
    base_departure_time: str,
    base_return_time: str,
    route_miles_range: tuple,
    time_window_minutes: int = 120,
    simulation_days: int = 1,
    initial_soc: float = 80.0,
    vehicle_variant: Optional[str] = None,
    mid_day_dwell: Optional[Dict[str, str]] = None
) -> List[Dict]:
    """
    Generate a multi-day schedule for a vehicle with cross-day support and dwell periods.
    
    Args:
        base_date (datetime): Starting date for the schedule
        base_departure_time (str): Base departure time in 'HH:MM' format
        base_return_time (str): Base return time in 'HH:MM' format
        route_miles_range (tuple): Min and max miles for route
        time_window_minutes (int): Variability window in minutes
        simulation_days (int): Number of days to generate schedules for
        initial_soc (float): Initial state of charge when vehicle departs
        vehicle_variant (str, optional): Specific vehicle variant to use for efficiency calculations
        mid_day_dwell (dict, optional): Mid-day dwell period with start_time and end_time in 'HH:MM' format
    
    Returns:
        List of schedule dictionaries with full datetime information and availability windows
    """
    schedules = []
    
    for day_offset in range(simulation_days):
        current_date = base_date + timedelta(days=day_offset)
        
        # Parse base times
        base_departure_hour, base_departure_minute = map(int, base_departure_time.split(':'))
        base_return_hour, base_return_minute = map(int, base_return_time.split(':'))
        
        # Create base datetime objects
        base_departure_datetime = current_date.replace(
            hour=base_departure_hour, 
            minute=base_departure_minute, 
            second=0, 
            microsecond=0
        )
        base_return_datetime = current_date.replace(
            hour=base_return_hour, 
            minute=base_return_minute, 
            second=0, 
            microsecond=0
        )
        
        # Add random offsets
        departure_offset = random.randint(-time_window_minutes, time_window_minutes)
        return_offset = random.randint(-time_window_minutes, time_window_minutes)
        
        departure_datetime = base_departure_datetime + timedelta(minutes=departure_offset)
        return_datetime = base_return_datetime + timedelta(minutes=return_offset)
        
        # Ensure return time is after departure time
        if return_datetime <= departure_datetime:
            return_datetime += timedelta(days=1)

        # Generate availability windows
        availability_windows = []
        
        # Morning dwell (start of day until first departure)
        day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        availability_windows.append({
            'start': day_start,
            'end': departure_datetime,
            'type': 'morning_dwell'
        })
        
        # Mid-day dwell if specified
        if mid_day_dwell:
            mid_dwell_start_hour, mid_dwell_start_minute = map(int, mid_day_dwell['start_time'].split(':'))
            mid_dwell_end_hour, mid_dwell_end_minute = map(int, mid_day_dwell['end_time'].split(':'))
            
            mid_dwell_start = current_date.replace(
                hour=mid_dwell_start_hour,
                minute=mid_dwell_start_minute,
                second=0,
                microsecond=0
            )
            mid_dwell_end = current_date.replace(
                hour=mid_dwell_end_hour,
                minute=mid_dwell_end_minute,
                second=0,
                microsecond=0
            )
            
            # Add random offsets to mid-day dwell times
            mid_dwell_start += timedelta(minutes=random.randint(-time_window_minutes//2, time_window_minutes//2))
            mid_dwell_end += timedelta(minutes=random.randint(-time_window_minutes//2, time_window_minutes//2))
            
            availability_windows.append({
                'start': mid_dwell_start,
                'end': mid_dwell_end,
                'type': 'mid_day_dwell'
            })
        
        # Evening dwell (final return until end of day)
        day_end = current_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        if return_datetime.date() == current_date.date():
            availability_windows.append({
                'start': return_datetime,
                'end': day_end,
                'type': 'evening_dwell'
            })
        else:
            # If return is next day, split the evening dwell across days
            next_day_start = day_end + timedelta(microseconds=1)
            availability_windows.append({
                'start': return_datetime,
                'end': next_day_start,
                'type': 'evening_dwell'
            })
        
        # Generate route miles
        route_miles = random.uniform(route_miles_range[0], route_miles_range[1])
        
        schedule = {
            'departure_datetime': departure_datetime,
            'return_datetime': return_datetime,
            'route_miles': round(route_miles, 1),
            'initial_soc': initial_soc,
            'target_soc': 80.0,  # Default target SOC
            'availability_windows': [
                {
                    'start': window['start'].isoformat(),
                    'end': window['end'].isoformat(),
                    'type': window['type']
                }
                for window in availability_windows
            ],
            'dwell_periods': {
                'morning': {
                    'start': '00:00',
                    'end': departure_datetime.strftime('%H:%M')
                },
                'mid_day': {
                    'start': mid_day_dwell['start_time'],
                    'end': mid_day_dwell['end_time']
                } if mid_day_dwell else None,
                'evening': {
                    'start': return_datetime.strftime('%H:%M'),
                    'end': '23:59'
                }
            }
        }
        
        schedules.append(schedule)
    
    return schedules

def extract_numbers(text: str) -> List[float]:
    """
    Extract numerical values from text that may contain ranges or multiple values.
    
    Args:
        text (str): Text containing numbers, possibly with ranges (e.g., "200-300")
        
    Returns:
        List of extracted numbers
    """
    if pd.isna(text):
        return []
    numbers = []
    for part in str(text).replace('Up to ', '').split('<br>'):
        try:
            cleaned = ''.join(c for c in part if c.isdigit() or c in '.-')
            if '-' in cleaned:
                start, end = map(float, cleaned.split('-'))
                numbers.append((start + end) / 2)  # Use average of range
            else:
                numbers.append(float(cleaned))
        except (ValueError, IndexError):
            continue
    return numbers

def extract_vehicle_data(csv_path: str) -> Dict:
    """
    Extract and process vehicle data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing vehicle data
        
    Returns:
        Dict containing processed vehicle data with battery capacities and ranges
    """
    try:
        df = pd.read_csv('./data/hvip_bev_large.csv')
        vehicles = {}
        
        for _, row in df.iterrows():
            try:
                # Extract battery capacities and ranges
                batteries = extract_numbers(row['Battery'])
                ranges = extract_numbers(row['Range'])
                
                if not batteries or not ranges:
                    continue
                
                model_name = str(row['Model']).strip()
                
                # Process each battery capacity and range combination
                for idx in range(min(len(batteries), len(ranges))):
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
                    
                    # Calculate max charging power (80% of battery capacity, capped at 350kW)
                    max_charging_power = min(float(battery_value) * 0.8, 350.0)
                    
                    vehicles[variant_key] = {
                        'name': model_name,
                        'batteryCapacity': float(battery_value),
                        'range': float(range_value),
                        'maxChargingPower': max_charging_power,
                        'category': str(row['Category']).strip(),
                        'vehicleTypes': vehicle_types,
                        'gvwr': gvwr_classes
                    }
            except Exception as row_error:
                print(f"Error processing row for model {row.get('Model', 'Unknown')}: {str(row_error)}")
                continue
        
        return vehicles
        
    except Exception as e:
        print(f"Error extracting vehicle data: {str(e)}")
        return {}
