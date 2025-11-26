from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import asyncio
from datetime import datetime

from fleetsim import ContinuousFleetSimulator, SimulationConfig

app = FastAPI()

# Store simulator instance
simulator: Optional[ContinuousFleetSimulator] = None
simulation_task: Optional[asyncio.Task] = None

class SimulatorConfig(BaseModel):
    fleet_size: int
    num_chargers: int
    charger_power: float
    optimizer_type: Literal["demand", "tou"]
    demand_limit: float = 1000.0
    simulation_interval: int = 5

@app.post("/simulator/start")
async def start_simulator(config: SimulatorConfig):
    global simulator, simulation_task
    
    if simulator:
        raise HTTPException(400, "Simulator already running")
        
    try:
        sim_config = SimulationConfig(
            fleet_size=config.fleet_size,
            num_chargers=config.num_chargers,
            charger_power=config.charger_power,
            optimizer_type=config.optimizer_type,
            demand_limit=config.demand_limit,
            simulation_interval=config.simulation_interval
        )
        
        simulator = ContinuousFleetSimulator(sim_config)
        simulation_task = asyncio.create_task(simulator.run_continuous_simulation())
        
        return {"status": "started", "config": config.dict()}
        
    except Exception as e:
        raise HTTPException(500, f"Error starting simulator: {str(e)}")

@app.post("/simulator/stop")
async def stop_simulator():
    global simulator, simulation_task
    
    if not simulator:
        raise HTTPException(400, "No simulator running")
        
    try:
        if simulation_task:
            simulation_task.cancel()
            await simulation_task
        simulator = None
        simulation_task = None
        return {"status": "stopped"}
        
    except Exception as e:
        raise HTTPException(500, f"Error stopping simulator: {str(e)}")

@app.post("/simulator/optimizer/{optimizer_type}")
async def change_optimizer(optimizer_type: Literal["demand", "tou"]):
    if not simulator:
        raise HTTPException(400, "No simulator running")
        
    try:
        await simulator.change_optimizer(optimizer_type)
        return {
            "status": "changed",
            "new_optimizer": optimizer_type,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"Error changing optimizer: {str(e)}")

@app.get("/simulator/status")
async def get_status():
    if not simulator:
        return {"status": "stopped"}
        
    return {
        "status": "running",
        "config": {
            "fleet_size": len(simulator.vehicles),
            "num_chargers": len(simulator.chargers),
            "optimizer_type": simulator.config.optimizer_type,
            "current_time": simulator.current_time.isoformat() if simulator.current_time else None,
            "total_energy_delivered": simulator.total_energy_delivered,
            "peak_power_demand": simulator.peak_power_demand
        },
        "vehicles": {
            vid: {
                "status": v.status,
                "soc": v.current_soc,
                "energy_delivered": v.energy_delivered,
                "energy_consumed": v.energy_consumed
            }
            for vid, v in simulator.vehicles.items()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)