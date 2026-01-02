import numpy as np
from collections import deque

from filters import BaseFilter

class Agent:    
    def __init__(self, id, initial_position, target_position, filter: BaseFilter):
        self.id = id
        self.locked = self.id in {0, 1, 2}

        self.position = np.array(initial_position)
        self.target_position = np.array(target_position)

        self.filter = filter.copy()
        self.filter.reset()

        self._last_T_measurements_raw = None
        self._last_measurement_filtered = None
    
    def apply_filter(self):
        if self._last_T_measurements_raw is None:
            raise RuntimeError("measure or apply_noise must be called before apply_filter")

        self._last_measurement_filtered = self.filter.update(self._last_T_measurements_raw)
    
    def update_position(self, laplacian_weights, dt):
        if self._last_measurement_filtered is None:
            raise RuntimeError("apply_filter must be called before update_position")
        
        if not self.locked:
            adjustment = np.zeros_like(self.position)
            for agent_id in range(len(laplacian_weights)):
                adjustment += laplacian_weights[self.id][agent_id] * self._last_measurement_filtered[agent_id]
            
            self.position += adjustment * dt

    def __str__(self):
        return f"Agent {self.id} at position {self.position}"
                