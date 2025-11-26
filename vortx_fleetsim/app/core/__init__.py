# vortx_fleetsim/app/core/__init__.py
from .fleet_manager import FleetManager
from .route_manager import RouteManager
from .telemetry import TelemetryManager

import os
import logging

# Configure logging for the core module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Default data file paths
DEFAULT_VEHICLE_DATA = os.path.join(DATA_DIR, 'hvip_bev_large.csv')
DEFAULT_RATES_DATA = os.path.join(DATA_DIR, 'bev_rates.csv')
DEFAULT_CACHE_DIR = os.path.join(DATA_DIR, 'cache')

# Create cache directory if it doesn't exist
if not os.path.exists(DEFAULT_CACHE_DIR):
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

# Version info
__version__ = '0.1.0'

# Export main classes
__all__ = [
    'FleetManager',
    'RouteManager',
    'TelemetryManager',
]

def init_fleet_system(
    vehicle_data_path: str = DEFAULT_VEHICLE_DATA,
    rates_data_path: str = DEFAULT_RATES_DATA,
    data_lake_connection = None
) -> tuple[FleetManager, RouteManager, TelemetryManager]:
    """
    Initialize the core fleet management system.
    
    Args:
        vehicle_data_path: Path to vehicle catalog data
        rates_data_path: Path to energy rates data
        data_lake_connection: Optional connection to data lake

    Returns:
        Tuple of (FleetManager, RouteManager, TelemetryManager)
    """
    try:
        # Initialize managers
        fleet_manager = FleetManager()
        route_manager = RouteManager()
        telemetry_manager = TelemetryManager(data_lake_connection)

        # Load vehicle data
        if os.path.exists(vehicle_data_path):
            fleet_manager.load_hvip_data(vehicle_data_path)
            logger.info(f"Loaded vehicle data from {vehicle_data_path}")
        else:
            logger.warning(f"Vehicle data file not found: {vehicle_data_path}")

        # Connect components
        fleet_manager.route_manager = route_manager
        fleet_manager.telemetry_manager = telemetry_manager

        return fleet_manager, route_manager, telemetry_manager

    except Exception as e:
        logger.error(f"Error initializing fleet system: {str(e)}")
        raise

def get_data_paths() -> dict:
    """Get standard data file paths"""
    return {
        'vehicle_data': DEFAULT_VEHICLE_DATA,
        'rates_data': DEFAULT_RATES_DATA,
        'cache_dir': DEFAULT_CACHE_DIR
    }