from flask import Flask, request as flask_request, jsonify, make_response
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback
import os
import sys
import json
from copy import deepcopy

# Import local modules
from fleet_core_algos import FleetManagementSystem
from fleet_utils import generate_multi_day_schedule, extract_vehicle_data
from v11_util import (
    load_bev_energy_prices, 
    create_time_slots, 
    validate_input_data, 
    calculate_load_curve, 
    calculate_charging_cost,
    calculate_charger_utilization,
    calculate_detailed_charging_costs,
    round_to_interval,
    INTERVAL_MINUTES
)
from v11_opportunity import OpportunityChargingSystem
from v11_opt_demand import ChargingOptimizerDemand, ChargingRequest, ChargingSession
from v11_opt_tou import ChargingOptimizerTOU

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Enable CORS with specific origin
CORS(app, 
     resources={r"/*": {
         "origins": "http://localhost:3000",  # Specific origin instead of wildcard
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Accept", "Authorization"],
         "expose_headers": ["Content-Type", "X-Total-Count"],
         "supports_credentials": True,
         "max_age": 600
     }})

# Initialize FleetManagementSystem and load vehicle data
fleet_system = FleetManagementSystem()
VEHICLE_DATA_PATH = './data/hvip_bev_large.csv'

if os.path.exists(VEHICLE_DATA_PATH):
    try:
        fleet_system.load_hvip_data(csv_path=VEHICLE_DATA_PATH)
        logger.info(f"Successfully loaded vehicles from {VEHICLE_DATA_PATH}")
    except Exception as e:
        logger.error(f"Error loading vehicles: {e}")
        logger.error(traceback.format_exc())

def calculate_charging_duration(initial_soc: float, target_soc: float, battery_capacity: float, charger_power: float) -> float:
    """Calculate required charging duration in hours."""
    energy_needed = ((target_soc - initial_soc) / 100) * battery_capacity
    # Add 20% buffer for charging inefficiencies and ramp-up/down times
    return (energy_needed / charger_power) * 1.2

def get_earliest_target_end_time(start_time: datetime, charging_duration: float, next_departure: datetime) -> datetime:
    """Calculate earliest possible target end time that ensures charging completion."""
    # Calculate minimum end time based on charging duration
    min_end_time = start_time + timedelta(hours=charging_duration)
    
    # If charging starts in evening, prioritize overnight charging
    if start_time.hour >= 17:
        # Try to complete charging by early morning (6 AM) next day
        next_morning = (start_time + timedelta(days=1)).replace(hour=6, minute=0)
        if next_morning < next_departure:
            return next_morning
    
    # If charging would overlap with peak period (16:00-21:00), try to shift it
    peak_start = start_time.replace(hour=16, minute=0)
    peak_end = start_time.replace(hour=21, minute=0)
    
    if start_time < peak_start and min_end_time.hour >= 16:
        # First try to complete before peak
        before_peak = peak_start - timedelta(minutes=30)
        if (before_peak - start_time).total_seconds() / 3600 >= charging_duration:
            return before_peak
        # If can't complete before peak, push to overnight
        return peak_end + timedelta(hours=charging_duration)
    
    # For vehicles arriving during peak, delay until peak end
    if peak_start <= start_time <= peak_end:
        return peak_end + timedelta(hours=charging_duration)
    
    # Default to minimum end time if other options aren't feasible
    return min(min_end_time, next_departure - timedelta(minutes=30))

# Helper function to convert fields to JSON serializable format
def convert_to_serializable(obj):
    """Convert objects to JSON serializable format."""
    try:
        if obj is None:
            return None
        elif isinstance(obj, (bool, np.bool_)):
            return 1 if obj else 0
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return convert_to_serializable(obj.__dict__)
        return obj
    except Exception as e:
        logger.error(f"Error converting object to serializable: {str(e)}")
        logger.error(f"Object type: {type(obj)}")
        logger.error(f"Object: {obj}")
        return str(obj)  # Fallback to string representation

@app.route('/vehicles', methods=['GET'])
def get_vehicles():
    """Get available vehicle catalog."""
    try:
        vehicle_data = {}
        for variant_key, specs in fleet_system.vehicle_catalog.items():
            vehicle_data[variant_key] = {
                'name': specs.model,
                'category': specs.category,
                'batteryCapacity': float(specs.battery_capacity),
                'range': float(specs.range),
                'maxChargingPower': min(float(specs.battery_capacity) * 0.8, 350.0),
                'gvwr': specs.vehicle_class,
                'vehicleTypes': specs.vehicle_types,
                'efficiency': specs.efficiency
            }
        return jsonify(vehicle_data)
        
    except Exception as e:
        logger.error(f"Error in get_vehicles: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/generate_random_fleet', methods=['POST'])
def generate_random_fleet():
    """Generate random fleet data based on input parameters."""
    try:
        data = flask_request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Extract and validate vehicle variant
        vehicle_variant = data.get('vehicle_variant', '')
        vehicle_specs = fleet_system.vehicle_catalog.get(vehicle_variant)
        if not vehicle_specs:
            return jsonify({"error": f"Vehicle not found: {vehicle_variant}"}), 400

        # Extract parameters
        num_vehicles = int(data.get('num_vehicles', 1))
        base_departure = data.get('base_departure', '08:00')
        base_return = data.get('base_return', '17:00')
        route_range = data.get('route_range', [50, 200])
        variability = int(data.get('variability', 30))
        base_date = data.get('base_date', datetime.now().strftime('%Y-%m-%d'))
        initial_soc = float(data.get('initial_soc', 80))

        # Generate fleet data
        fleet_data = []
        for i in range(num_vehicles):
            # Calculate route and energy metrics
            route_distance = np.random.uniform(route_range[0], route_range[1])
            energy_needed = vehicle_specs.efficiency * route_distance

            # Generate times with variability
            departure_time = datetime.strptime(f"{base_date} {base_departure}", '%Y-%m-%d %H:%M')
            return_time = datetime.strptime(f"{base_date} {base_return}", '%Y-%m-%d %H:%M')

            # Add variability
            departure_time += timedelta(minutes=np.random.randint(-variability, variability))
            return_time += timedelta(minutes=np.random.randint(-variability, variability))

            # Handle overnight case
            if return_time < departure_time:
                return_time += timedelta(days=1)

            # Calculate SOC changes
            energy_consumed = vehicle_specs.efficiency * route_distance
            final_soc = max(0, min(100, initial_soc - (energy_consumed / vehicle_specs.battery_capacity * 100)))

            vehicle_data = {
                "Vehicle_ID": f"{vehicle_variant}_{i + 1}",
                "Return_Time": return_time.strftime('%H:%M'),
                "Departure_Time": departure_time.strftime('%H:%M'),
                "Route_Distance_miles": round(route_distance, 2),
                "Energy_Needed_kWh": round(energy_needed, 2),
                "Battery_Capacity_kWh": vehicle_specs.battery_capacity,
                "Initial_SOC": initial_soc,
                "Final_SOC": round(final_soc, 2),
                "Efficiency_kWh_per_mile": vehicle_specs.efficiency
            }
            fleet_data.append(vehicle_data)

        return jsonify(fleet_data)
    
    except Exception as e:
        logger.error(f"Error in generate_random_fleet: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    """Run charging simulation based on input parameters."""
    try:
        data = flask_request.get_json()
        params = data.get('params', {})
        vehicles_data = data.get('fleet', [])

        # Extract optimization parameters
        max_charger_power = float(params.get('chargerPower', 150))
        num_chargers = int(params.get('numChargers', 2))
        peak_load_limit = float(params.get('peakLoadLimit', 1500))
        charging_method = params.get('chargingMethod', 'optimize')
        optimization_goal = params.get('optimizationGoal', 'demand')

        # Process vehicle data
        vehicles_for_charging = pd.DataFrame(vehicles_data)
        base_date = datetime.now().date()

        # Convert times to datetime
        vehicles_for_charging['operations_start'] = pd.to_datetime(
            vehicles_for_charging['Departure_Time'].apply(lambda x: f"{base_date} {x}")
        )
        vehicles_for_charging['operations_end'] = pd.to_datetime(
            vehicles_for_charging['Return_Time'].apply(lambda x: f"{base_date} {x}")
        )

        # Handle overnight operations
        overnight_mask = vehicles_for_charging['operations_end'] < vehicles_for_charging['operations_start']
        vehicles_for_charging.loc[overnight_mask, 'operations_end'] += pd.Timedelta(days=1)

        # Calculate next departure time for each vehicle (next day's departure)
        vehicles_for_charging['next_departure'] = vehicles_for_charging['operations_start'].apply(
            lambda x: x + pd.Timedelta(days=1)
        )

        # Find the earliest return time to start simulation
        simulation_start = vehicles_for_charging['operations_end'].min()
        
        # Round simulation start to nearest interval
        minutes_to_round = simulation_start.minute % INTERVAL_MINUTES
        if minutes_to_round != 0:
            simulation_start -= timedelta(minutes=minutes_to_round)

        # Create time slots for exactly 36 hours
        time_slots = pd.date_range(
            start=simulation_start,
            periods=(36 * 60) // INTERVAL_MINUTES,  # 36 hours worth of intervals
            freq=f'{INTERVAL_MINUTES}min'
        )

        # Load energy prices
        energy_prices = load_bev_energy_prices()

        # Initialize optimizer based on method and goal
        optimizer_params = {
            'num_chargers': num_chargers,
            'charger_power': max_charger_power,
            'demand_limit': peak_load_limit,
            'interval_minutes': INTERVAL_MINUTES,
            'energy_prices': energy_prices  # Pass energy prices to all optimizers
        }

        if charging_method == 'opportunity':
            optimizer = OpportunityChargingSystem(**optimizer_params)
        else:
            if optimization_goal == 'demand':
                optimizer = ChargingOptimizerDemand(**optimizer_params)
            else:  # optimization_goal == 'tou'
                optimizer = ChargingOptimizerTOU(**optimizer_params)

        # Initialize simulation
        optimizer.initialize_simulation(simulation_start)

        # Prepare charging requests
        charging_requests = []
        for _, vehicle in vehicles_for_charging.iterrows():
            charging_duration = calculate_charging_duration(
                initial_soc=vehicle['Initial_SOC'],
                target_soc=vehicle['Final_SOC'],
                battery_capacity=vehicle['Battery_Capacity_kWh'],
                charger_power=max_charger_power
            )

            target_end_time = get_earliest_target_end_time(
                start_time=vehicle['operations_end'],
                charging_duration=charging_duration,
                next_departure=vehicle['next_departure']
            )

            availability = [(
                vehicle['operations_end'],
                vehicle['next_departure']
            )]

            request = ChargingRequest(
                vehicle_id=vehicle['Vehicle_ID'],
                start_time=vehicle['operations_end'],
                target_end_time=target_end_time,
                initial_soc=vehicle['Initial_SOC'],
                target_soc=vehicle['Final_SOC'],
                battery_capacity=vehicle['Battery_Capacity_kWh'],
                max_charging_rate=max_charger_power,
                availability=availability,
                route_distance=vehicle['Route_Distance_miles'],
                energy_needed=vehicle['Energy_Needed_kWh'],
                efficiency=vehicle['Efficiency_kWh_per_mile'],
                base_departure=vehicle['next_departure'],
                base_return=vehicle['operations_end']
            )
            charging_requests.append(request)

        # Sort requests by start time
        charging_requests.sort(key=lambda r: r.start_time)

        # Run simulation
        request_index = 0
        current_time = simulation_start
        end_time = time_slots[-1]

        while current_time <= end_time:
            # Add new charging requests
            while request_index < len(charging_requests) and charging_requests[request_index].start_time <= current_time:
                optimizer.add_charging_request(charging_requests[request_index], current_time)
                request_index += 1

            # Simulate time step
            optimizer.simulate_time_step(current_time)
            current_time += timedelta(minutes=INTERVAL_MINUTES)

        # Get results
        results = optimizer.get_results()

        # Process completed sessions into schedule format
        schedule = []
        for session in optimizer.completed_sessions:
            if not session.power_profile:
                continue

            schedule_entry = {
                'Vehicle_ID': session.vehicle_id,
                'Charger_ID': f"C{session.charger_id + 1}",
                'charging_start': session.charging_start.isoformat() if session.charging_start else session.start_time.isoformat(),
                'charging_end': session.completion_time.isoformat() if session.completion_time else None,
                'Initial_SOC': session.initial_soc,
                'Final_SOC': session.current_soc,
                'Energy_Charged': session.energy_delivered,
                'Charging_Rates': session.power_profile
            }
            schedule.append(schedule_entry)

        # Calculate metrics
        load_curve = calculate_load_curve(schedule, time_slots)
        charger_utilization, total_utilization = calculate_charger_utilization(
            schedule, num_chargers
        )

        # Generate time labels
        time_labels = []
        current_label_time = simulation_start
        for _ in range(len(time_slots)):
            day_number = (current_label_time - simulation_start).days + 1
            time_str = current_label_time.strftime('%H:%M')
            time_labels.append(f"D{day_number} {time_str}")
            current_label_time += timedelta(minutes=INTERVAL_MINUTES)

        # Format load curve
        if isinstance(load_curve, np.ndarray):
            load_curve = load_curve.tolist()
        elif not isinstance(load_curve, list):
            load_curve = list(load_curve)

        # Prepare response
        response_data = {
            "schedule": schedule,
            "load_curve": load_curve,
            "time_labels": time_labels,
            "interval_minutes": INTERVAL_MINUTES,
            "simulation_start": simulation_start.isoformat(),
            "unscheduled_vehicles": results.get('unscheduled_vehicles', []),
            "peak_demand": results.get("peak_demand", 0),
            "charging_costs": {
                "total": {
                    "energy": results.get("total_energy_delivered", 0),
                    "cost": results.get("total_cost", 0),
                    "hours": 0.0,  # Not used in frontend
                    "avg_rate": (
                        results.get("total_cost", 0) / results.get("total_energy_delivered", 1) 
                        if results.get("total_energy_delivered", 0) > 0 else 0
                    )
                },
                "per_vehicle": results.get("per_vehicle_costs", {}),
                "per_period": results.get("cost_breakdown", {})
            },
            "charger_utilization": {
                "total": total_utilization,
                "per_charger": charger_utilization
            },
            "charging_failures": results.get("charging_failures", {})
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in simulate: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/generate_multi_day_schedule', methods=['POST'])
def generate_multi_day_schedule_route():
    """Generate multi-day schedule for vehicles."""
    try:
        data = flask_request.get_json()
        
        # Extract mid-day dwell period if provided
        mid_day_dwell = None
        if data.get('mid_day_dwell'):
            mid_day_dwell = {
                'start_time': data['mid_day_dwell']['start_time'],
                'end_time': data['mid_day_dwell']['end_time']
            }
        
        schedule = generate_multi_day_schedule(
            base_date=datetime.strptime(data['base_date'], '%Y-%m-%d'),
            base_departure_time=data['base_departure_time'],
            base_return_time=data['base_return_time'],
            route_miles_range=tuple(data.get('route_miles_range', [50, 200])),
            time_window_minutes=data.get('time_window_minutes', 120),
            simulation_days=data.get('simulation_days', 1),
            initial_soc=data.get('initial_soc', 80.0),
            vehicle_variant=data.get('vehicle_variant'),
            mid_day_dwell=mid_day_dwell
        )

        # Serialize datetime objects
        serialized_schedule = [{
            **entry,
            'departure_datetime': entry['departure_datetime'].isoformat(),
            'return_datetime': entry['return_datetime'].isoformat(),
            'availability_windows': entry['availability_windows'],
            'dwell_periods': entry['dwell_periods']
        } for entry in schedule]

        return jsonify(serialized_schedule)
    
    except Exception as e:
        logger.error(f"Error in generate_multi_day_schedule: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/simulate_batch', methods=['POST'])
def simulate_batch():
    """Run multiple simulations with different peak load limits."""
    try:
        data = flask_request.get_json()
        params = data.get('params', {})
        vehicles_data = data.get('fleet', [])

        # Extract parameters
        max_charger_power = float(params.get('chargerPower', 150))
        num_chargers = int(params.get('numChargers', 2))
        start_limit = float(params.get('startLimit', 500))
        end_limit = float(params.get('endLimit', 2000))
        step_size = float(params.get('stepSize', 100))
        charging_method = params.get('chargingMethod', 'optimize')
        optimization_goal = params.get('optimizationGoal', 'demand')

        peak_load_limits = np.arange(start_limit, end_limit + step_size, step_size)
        scenarios = []
        lowest_viable_limit = None
        lowest_cost_limit = None
        min_total_cost = float('inf')

        # Load energy prices
        energy_prices = load_bev_energy_prices()

        # Process vehicle data
        vehicles_for_charging = pd.DataFrame(vehicles_data)
        base_date = datetime.now().date()

        # Convert times to datetime
        vehicles_for_charging['operations_start'] = pd.to_datetime(
            vehicles_for_charging['Departure_Time'].apply(lambda x: f"{base_date} {x}")
        )
        vehicles_for_charging['operations_end'] = pd.to_datetime(
            vehicles_for_charging['Return_Time'].apply(lambda x: f"{base_date} {x}")
        )

        # Handle overnight operations
        overnight_mask = vehicles_for_charging['operations_end'] < vehicles_for_charging['operations_start']
        vehicles_for_charging.loc[overnight_mask, 'operations_end'] += pd.Timedelta(days=1)

        # Calculate next departure time (next day)
        vehicles_for_charging['next_departure'] = vehicles_for_charging['operations_start'].apply(
            lambda x: x + pd.Timedelta(days=1)
        )

        # Prepare base charging requests
        base_charging_requests = []
        for _, vehicle in vehicles_for_charging.iterrows():
            # Calculate charging duration
            charging_duration = calculate_charging_duration(
                initial_soc=vehicle['Initial_SOC'],
                target_soc=vehicle['Final_SOC'],
                battery_capacity=vehicle['Battery_Capacity_kWh'],
                charger_power=max_charger_power
            )

            # Calculate optimal end time
            target_end_time = get_earliest_target_end_time(
                start_time=vehicle['operations_end'],
                charging_duration=charging_duration,
                next_departure=vehicle['next_departure']
            )

            # Create availability windows
            availability = [(
                vehicle['operations_end'],
                vehicle['next_departure']
            )]

            # Create charging request
            request = ChargingRequest(
                vehicle_id=vehicle['Vehicle_ID'],
                start_time=vehicle['operations_end'],
                target_end_time=target_end_time,
                initial_soc=vehicle['Initial_SOC'],
                target_soc=vehicle['Final_SOC'],
                battery_capacity=vehicle['Battery_Capacity_kWh'],
                max_charging_rate=max_charger_power,
                availability=availability,
                route_distance=vehicle['Route_Distance_miles'],
                energy_needed=vehicle['Energy_Needed_kWh'],
                efficiency=vehicle['Efficiency_kWh_per_mile'],
                base_departure=vehicle['next_departure'],
                base_return=vehicle['operations_end']
            )
            base_charging_requests.append(request)

        # Sort requests by start time
        base_charging_requests.sort(key=lambda r: r.start_time)
        simulation_start = min(r.start_time for r in base_charging_requests)
        simulation_start = round_to_interval(simulation_start, INTERVAL_MINUTES)

        # Test each peak load limit
        for limit in peak_load_limits:
            # Initialize optimizer based on method and goal
            optimizer_params = {
                'num_chargers': num_chargers,
                'charger_power': max_charger_power,
                'demand_limit': limit,
                'interval_minutes': INTERVAL_MINUTES,
                'energy_prices': energy_prices
            }

            if charging_method == 'opportunity':
                optimizer = OpportunityChargingSystem(**optimizer_params)
            else:
                if optimization_goal == 'demand':
                    optimizer = ChargingOptimizerDemand(**optimizer_params)
                else:  # optimization_goal == 'tou'
                    optimizer = ChargingOptimizerTOU(**optimizer_params)
            
            optimizer.initialize_simulation(simulation_start)
            charging_requests = deepcopy(base_charging_requests)

            # Run simulation
            current_time = simulation_start
            end_time = current_time + timedelta(hours=36)
            request_index = 0

            while current_time <= end_time:
                while (request_index < len(charging_requests) and 
                       charging_requests[request_index].start_time <= current_time):
                    optimizer.add_charging_request(charging_requests[request_index], current_time)
                    request_index += 1

                optimizer.simulate_time_step(current_time)
                current_time += timedelta(minutes=INTERVAL_MINUTES)

            # Get and process results
            results = optimizer.get_results()
            vehicle_summaries = convert_to_serializable(results.get("vehicle_summaries", []))

            # Process vehicle summaries
            fully_charged_vehicles = len([
                v for v in vehicle_summaries 
                if float(v["final_soc"]) >= float(v["target_soc"])
            ])
            total_vehicles = len(vehicle_summaries)
            total_cost = float(results["total_cost"])

            # Track lowest cost scenario that charges all vehicles
            if fully_charged_vehicles == total_vehicles:
                if lowest_viable_limit is None or limit < lowest_viable_limit:
                    lowest_viable_limit = float(limit)
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    lowest_cost_limit = float(limit)

            # Create serializable scenario summary
            scenario = {
                "peak_limit": float(limit),
                "peak_demand": float(results["peak_demand"]),
                "total_energy": float(results["total_energy_delivered"]),
                "total": {  # Match the structure from /simulate endpoint
                    "energy": float(results["total_energy_delivered"]),
                    "cost": float(results["total_cost"]),
                    "hours": 0.0,  # Not used in frontend
                    "avg_rate": float(results["total_cost"]) / float(results["total_energy_delivered"]) if results["total_energy_delivered"] > 0 else 0
                },
                "vehicles_completed": int(fully_charged_vehicles),
                "total_vehicles": int(total_vehicles),
                "all_vehicles_charged": 1 if fully_charged_vehicles == total_vehicles else 0,
                "cost_breakdown": results.get("cost_breakdown", {}),
                "vehicle_summaries": vehicle_summaries,
                "load_curve": [float(x) for x in optimizer.load_curve]
            }
            scenarios.append(scenario)

        # Prepare response
        response_data = {
            "scenarios": scenarios,
            "lowest_viable_limit": float(lowest_viable_limit) if lowest_viable_limit is not None else None,
            "lowest_cost_limit": float(lowest_cost_limit) if lowest_cost_limit is not None else None,
            "min_total_cost": float(min_total_cost) if min_total_cost != float('inf') else None,
            "simulation_parameters": {
                "start_limit": float(start_limit),
                "end_limit": float(end_limit),
                "step_size": float(step_size),
                "num_chargers": int(num_chargers),
                "charger_power": float(max_charger_power)
            }
        }

        return jsonify(convert_to_serializable(response_data))

    except Exception as e:
        logger.error(f"Error in simulate_batch_tou: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "error_details": traceback.format_exc()
        }), 500

@app.after_request
def after_request(response):
    """Ensure CORS headers are set correctly for all responses."""
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')  # Specific origin instead of wildcard
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)  # Changed port to 8000
