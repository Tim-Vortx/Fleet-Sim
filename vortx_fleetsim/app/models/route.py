from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

@dataclass
class RouteWindow:
    """Time window for a route with start and end times."""
    start_time: datetime
    end_time: datetime
    window_type: str = field(default="operation")  # operation, dwell, charging
    
    def __post_init__(self):
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
    
    def duration(self) -> timedelta:
        """Calculate duration of the time window."""
        return self.end_time - self.start_time
    
    def overlaps(self, other: 'RouteWindow') -> bool:
        """Check if this window overlaps with another."""
        return (self.start_time < other.end_time and 
                self.end_time > other.start_time)
    
    def contains(self, time: datetime) -> bool:
        """Check if a specific time falls within this window."""
        return self.start_time <= time <= self.end_time

@dataclass
class Route:
    """Represents a vehicle route with schedule and energy requirements."""
    route_id: str
    vehicle_id: str
    departure_time: datetime
    return_time: datetime
    distance_miles: float
    energy_required: float
    initial_soc: float = field(default=100.0)
    target_soc: float = field(default=80.0)
    route_windows: List[RouteWindow] = field(default_factory=list)
    dwell_windows: List[RouteWindow] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.route_windows:
            # Create default operation window if none provided
            self.route_windows = [RouteWindow(
                start_time=self.departure_time,
                end_time=self.return_time,
                window_type="operation"
            )]
        
        # Validate time windows
        self._validate_windows()
    
    def _validate_windows(self):
        """Validate all time windows are properly ordered and non-overlapping."""
        # Check operation windows
        for i in range(len(self.route_windows)):
            if i > 0 and self.route_windows[i].start_time <= self.route_windows[i-1].end_time:
                raise ValueError("Route windows must be in chronological order without overlap")
        
        # Check dwell windows
        for i in range(len(self.dwell_windows)):
            if i > 0 and self.dwell_windows[i].start_time <= self.dwell_windows[i-1].end_time:
                raise ValueError("Dwell windows must be in chronological order without overlap")
            
            # Check dwell windows don't overlap with operation windows
            for op_window in self.route_windows:
                if self.dwell_windows[i].overlaps(op_window):
                    raise ValueError("Dwell windows cannot overlap with operation windows")
    
    def add_dwell_window(self, start: datetime, end: datetime):
        """Add a new dwell window to the route."""
        new_window = RouteWindow(start, end, "dwell")
        
        # Check for overlaps with existing windows
        for window in self.route_windows + self.dwell_windows:
            if new_window.overlaps(window):
                raise ValueError(f"New dwell window overlaps with existing {window.window_type} window")
        
        self.dwell_windows.append(new_window)
        self.dwell_windows.sort(key=lambda x: x.start_time)
    
    def get_availability_windows(self) -> List[Tuple[datetime, datetime]]:
        """Get all time windows when vehicle is available for charging."""
        all_windows = sorted(
            self.route_windows + self.dwell_windows,
            key=lambda x: x.start_time
        )
        
        availability = []
        current_time = self.departure_time
        
        for window in all_windows:
            # Add gap before window if exists
            if current_time < window.start_time:
                availability.append((current_time, window.start_time))
            
            # Add dwell window as available time
            if window.window_type == "dwell":
                availability.append((window.start_time, window.end_time))
            
            current_time = window.end_time
        
        # Add final gap if exists
        if current_time < self.return_time:
            availability.append((current_time, self.return_time))
        
        return availability
    
    def get_total_available_time(self) -> timedelta:
        """Calculate total time available for charging."""
        return sum(
            (end - start for start, end in self.get_availability_windows()),
            timedelta()
        )
    
    def is_available_at(self, time: datetime) -> bool:
        """Check if vehicle is available for charging at specific time."""
        return any(
            start <= time <= end
            for start, end in self.get_availability_windows()
        )
    
    def calculate_min_charging_power(self, charger_efficiency: float = 0.9) -> float:
        """
        Calculate minimum required charging power to meet schedule.
        
        Args:
            charger_efficiency: Efficiency factor of charger (0-1)
            
        Returns:
            Float: Minimum required charging power in kW
        """
        total_available_time = self.get_total_available_time().total_seconds() / 3600  # hours
        if total_available_time <= 0:
            return float('inf')
        
        # Calculate energy needed including efficiency losses
        energy_needed = (self.energy_required / charger_efficiency)
        
        return energy_needed / total_available_time
    
    def to_dict(self) -> Dict:
        """Convert route to dictionary format."""
        return {
            "route_id": self.route_id,
            "vehicle_id": self.vehicle_id,
            "departure_time": self.departure_time.isoformat(),
            "return_time": self.return_time.isoformat(),
            "distance_miles": self.distance_miles,
            "energy_required": self.energy_required,
            "initial_soc": self.initial_soc,
            "target_soc": self.target_soc,
            "route_windows": [
                {
                    "start_time": w.start_time.isoformat(),
                    "end_time": w.end_time.isoformat(),
                    "window_type": w.window_type
                }
                for w in self.route_windows
            ],
            "dwell_windows": [
                {
                    "start_time": w.start_time.isoformat(),
                    "end_time": w.end_time.isoformat(),
                    "window_type": w.window_type
                }
                for w in self.dwell_windows
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Route':
        """Create route from dictionary format."""
        route = cls(
            route_id=data["route_id"],
            vehicle_id=data["vehicle_id"],
            departure_time=datetime.fromisoformat(data["departure_time"]),
            return_time=datetime.fromisoformat(data["return_time"]),
            distance_miles=data["distance_miles"],
            energy_required=data["energy_required"],
            initial_soc=data.get("initial_soc", 100.0),
            target_soc=data.get("target_soc", 80.0)
        )
        
        # Add route windows
        route.route_windows = [
            RouteWindow(
                start_time=datetime.fromisoformat(w["start_time"]),
                end_time=datetime.fromisoformat(w["end_time"]),
                window_type=w["window_type"]
            )
            for w in data.get("route_windows", [])
        ]
        
        # Add dwell windows
        route.dwell_windows = [
            RouteWindow(
                start_time=datetime.fromisoformat(w["start_time"]),
                end_time=datetime.fromisoformat(w["end_time"]),
                window_type=w["window_type"]
            )
            for w in data.get("dwell_windows", [])
        ]
        
        return route