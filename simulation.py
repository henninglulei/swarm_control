import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from agent import Agent
from filters import BaseFilter

@dataclass
class SimulationConfig:
    noise_covariance: np.ndarray
    filter: BaseFilter
    desc: str
    dt: float


class Simulation:
    def __init__(self, agents: List[Agent], L: int, config: SimulationConfig, seed: int = None):
        self.agents = agents
        self.laplacian_weights = L
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.error_history = None
        self.timestep = 0
    
    def _step(self) -> None:
        for agent in self.agents:
            
            agent.measure(self.agents, self.config.noise_covariance, self.rng)
            agent.apply_filter()
            agent.update_position(self.laplacian_weights, self.config.dt)

            self.error_history[self.timestep] += np.linalg.norm(agent.position - agent.target_position) ** 2
        self.timestep += 1
    
    def run(self, num_steps: int) -> Dict[str, Any]:
        self.error_history = np.zeros(num_steps)
        
        for _ in range(num_steps):
            self._step()
        
        return self.error_history