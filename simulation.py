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
    T: int  # Number of independent measurements per iteration


class Simulation:
    def __init__(self, agents: List[Agent], L: int, config: SimulationConfig, seed: int = None):
        self.agents = agents
        self.laplacian_weights = L
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.error_history = None
        self.timestep = 0
    
    def _step(self) -> None:
        positions = np.array([agent.position for agent in self.agents])
        num_agents = len(self.agents)
        dim = positions.shape[1]
        
        # Generate T noisy measurements
        T_measurements = []
        real_distances = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        for _ in range(self.config.T):
            # Batch generate all noise at once
            all_noise = self.rng.multivariate_normal(
                mean=np.zeros(dim), 
                cov=self.config.noise_covariance, 
                size=num_agents * num_agents
            ).reshape(num_agents, num_agents, dim)

            # TODO: Set the diagonal to zero noise
            
            T_measurements.append(real_distances + all_noise)

        
        self.last_controls = np.zeros_like(positions)
        for agent_idx, agent in enumerate(self.agents):
            agent_T_measurements = [T_measurements[t][agent_idx] for t in range(self.config.T)]
            agent._last_T_measurements_raw = agent_T_measurements
            agent.apply_filter()
            agent.update_position(self.laplacian_weights, self.config.dt)

            self.error_history[self.timestep] += np.linalg.norm(agent.position - agent.target_position) ** 2
        
        self.timestep += 1
    
    def run(self, num_steps: int) -> Dict[str, Any]:
        self.error_history = np.zeros(num_steps)
        
        for _ in range(num_steps):
            self._step()
        
        return self.error_history