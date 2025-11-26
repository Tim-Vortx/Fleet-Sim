from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum

class VehicleStatus(Enum):
    """Vehicle operational status."""
    DRIVING = 'driving'
    CHARGING = 'charging'
    IDLE = 'idle'
    MAINTENANCE = 'maintenance'
    OUT_OF_SERVICE = 'out_of_service'

@dataclass
class VehicleSpecs:
    """Vehicle specifications."""
    model: str
    category: str
    battery_capacities: List[float]
    ranges: List[float]
    battery_capacity: float = field(default=0.0)
    range: float = field(default=0.0)
    vehicle_class: List[str] = field(default_factory=list)
    vehicle_types: List[str] = field(default_factory=list)
    efficiency: float = field(default=0.0)
    max_charging_power: float = field(default=0.0)
    charging_curve: Dict[float, float] = field(default_factory=dict)  # SoC to max power mapping

    def __post_init__(self):
        if self.battery_capacity not in self.battery_capacities:
            raise ValueError(f"Selected battery capacity {self.battery_capacity} not in available capacities {self.battery_capacities}")
        if not self.ranges:
            raise ValueError("Vehicle must have at least one range value")
        if self.efficiency <= 0:
            self.efficiency = self.battery_capacity / self.range if self.range else 2.0
        if self.max_charging_power <= 0:
            # Default to 80% of battery capacity, capped at 350kW
            self.max_charging_power = min(float(self.battery_capacity) * 0.8, 350.0)
        if not self.charging_curve:
            # Create default charging curve if none provided
            self._create_default_charging_curve()

    def _create_default_charging_curve(self):
        """Create a default charging curve based on battery capacity."""
        self.charging_curve = {
            0.0: self.max_charging_power,    # 0-20%: max power
            20.0: self.max_charging_power,
            50.0: self.max_charging_power * 0.8,  # 20-50%: 80% of max
            80.0: self.max_charging_power * 0.5,  # 50-80%: 50% of max
            90.0: self.max_charging_power * 0.3,  # 80-90%: 30% of max
            100.0: self.max_charging_power * 0.1   # 90-100%: 10% of max
        }

    def get_max_charging_power(self, current_soc: float) -> float:
        """Get maximum charging power at current SoC."""
        if not self.charging_curve:
            return self.max_charging_power

        # Find the closest SoC points and interpolate
        soc_points = sorted(self.charging_curve.keys())
        for i in range(len(soc_points) - 1):
            if soc_points[i] <= current_soc <= soc_points[i + 1]:
                soc_low, soc_high = soc_points[i], soc_points[i + 1]
                power_low = self.charging_curve[soc_low]
                power_high = self.charging_curve[soc_high]
                
                # Linear interpolation
                soc_ratio = (current_soc - soc_low) / (soc_high - soc_low)
                return power_low + soc_ratio * (power_high - power_low)
        
        return self.charging_curve[soc_points[-1]]

@dataclass
class VehicleState:
    """Current state of a vehicle."""
    vehicle_id: str
    status: VehicleStatus
    location: Tuple[float, float]  # lat, lng
    current_soc: float
    last_updated: datetime
    route_id: Optional[str] = None
    charger_id: Optional[str] = None
    energy_delivered: float = 0.0
    energy_consumed: float = 0.0
    battery_capacity: float = 0.0
    charging_power: float = 0.0
    remaining_range: float = 0.0
    maintenance_due: bool = False
    telemetry_active: bool = True

    def __post_init__(self):
        # Convert string status to enum if needed
        if isinstance(self.status, str):
            self.status = VehicleStatus(self.status)
        
        # Validate SOC range
        if not 0 <= self.current_soc <= 100:
            raise ValueError("State of charge must be between 0 and 100")
        
        # Validate energy values
        if self.energy_delivered < 0 or self.energy_consumed < 0:
            raise ValueError("Energy values cannot be negative")

    def update_state(self, 
                    new_status: Optional[Union[str, VehicleStatus]] = None,
                    new_location: Optional[Tuple[float, float]] = None,
                    new_soc: Optional[float] = None,
                    energy_delta: Optional[float] = None,
                    power: Optional[float] = None) -> None:
        """Update vehicle state with new values."""
        if new_status:
            self.status = VehicleStatus(new_status) if isinstance(new_status, str) else new_status
        
        if new_location:
            self.location = new_location
        
        if new_soc is not None:
            if not 0 <= new_soc <= 100:
                raise ValueError("State of charge must be between 0 and 100")
            self.current_soc = new_soc
        
        if energy_delta:
            if self.status == VehicleStatus.CHARGING:
                self.energy_delivered += energy_delta
            elif self.status == VehicleStatus.DRIVING:
                self.energy_consumed += abs(energy_delta)
        
        if power is not None:
            self.charging_power = max(0, power)
        
        self.last_updated = datetime.now()

    def calculate_range(self, efficiency: float) -> float:
        """Calculate remaining range based on current SOC and efficiency."""
        energy_remaining = (self.current_soc / 100) * self.battery_capacity
        self.remaining_range = energy_remaining / efficiency
        return self.remaining_range

    def to_dict(self) -> Dict:
        """Convert vehicle state to dictionary format."""
        return {
            "vehicle_id": self.vehicle_id,
            "status": self.status.value,
            "location": self.location,
            "current_soc": self.current_soc,
            "last_updated": self.last_updated.isoformat(),
            "route_id": self.route_id,
            "charger_id": self.charger_id,
            "energy_delivered": self.energy_delivered,
            "energy_consumed": self.energy_consumed,
            "battery_capacity": self.battery_capacity,
            "charging_power": self.charging_power,
            "remaining_range": self.remaining_range,
            "maintenance_due": self.maintenance_due,
            "telemetry_active": self.telemetry_active
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VehicleState':
        """Create vehicle state from dictionary format."""
        return cls(
            vehicle_id=data["vehicle_id"],
            status=data["status"],
            location=tuple(data["location"]),
            current_soc=data["current_soc"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            route_id=data.get("route_id"),
            charger_id=data.get("charger_id"),
            energy_delivered=data.get("energy_delivered", 0.0),
            energy_consumed=data.get("energy_consumed", 0.0),
            battery_capacity=data.get("battery_capacity", 0.0),
            charging_power=data.get("charging_power", 0.0),
            remaining_range=data.get("remaining_range", 0.0),
            maintenance_due=data.get("maintenance_due", False),
            telemetry_active=data.get("telemetry_active", True)
        )