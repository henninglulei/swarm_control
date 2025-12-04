import numpy as np

class Agent:
    
    class FilterType:
        RLS = 0
        ML = 1
        KALMAN = 2
        MovingAverage = 3
    
    def __init__(self, id, initial_position):
        self.id = id
        self.position = np.array(initial_position)
        self.measurement_history = []
        self.locked = self.id in {0, 1, 2}
        self.position_history = [self.position.copy()]
        
    def measure(self, agents, noise_covariance, use_noise=True):
        """Simulate noisy distance measurements to other agents."""
        num_agents = len(agents)
        measurements = np.zeros((num_agents, 2))
        for i, agent in enumerate(agents):
            if agent is not self:
                distance = self.position - agent.position
                noise = np.random.multivariate_normal(mean=np.zeros(distance.shape[0]), cov=noise_covariance)
                measurements[i, :] = distance + noise if use_noise else distance
        
        self.measurement_history.append(measurements)
        return measurements
    
    def filter(self, filter: 'Agent.FilterType', **filter_params):
        current_measurements = self.measurement_history[-1]
        filtered_measurements = np.zeros_like(current_measurements)
        
        match filter:
            case Agent.FilterType.RLS:
                for i, measurement in enumerate(current_measurements):
                    pass
            
            case Agent.FilterType.MovingAverage:
                for i, measurement in enumerate(current_measurements):
                    filtered_measurements[i] = sum(self.measurement_history[j][i] for j in range(max(0, len(self.measurement_history)-filter_params['window_size']), len(self.measurement_history))) / filter_params['window_size']
        
            case Agent.FilterType.KALMAN:
                for i, measurement in enumerate(current_measurements):
                    pass
                
            case _:
                raise ValueError("Unknown filter type")
            
        self.measurement_history[-1] = filtered_measurements
    
    def update_position(self, laplacian_weights, timestep):
        if not self.locked:
            adjustment = np.zeros_like(self.position)
            for i, weight in enumerate(laplacian_weights):
                adjustment += weight * self.measurement_history[-1][i]
            
            self.position += adjustment * timestep
            
        self.position_history.append(self.position.copy())

    def __str__(self):
        return f"Agent {self.id} at position {self.position}"
                