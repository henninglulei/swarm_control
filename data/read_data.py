from scipy.io import loadmat
import numpy as np

def read_data() -> dict:
    mat = loadmat('data/data.mat')

    K = mat['K'][0][0] # Max. number of iterations
    L = np.array(mat['L']) # Edge weights
    num_links = mat['M'][0][0] # Number of links
    num_agents = mat['N'][0][0] # Number of agents
    noise_covariance = np.array(mat['R']) # Noise covariance matrix
    timestep = mat['dt'][0][0]  # Time step
    initial_positions = np.array(mat['z']) # Current positions
    target_positions = np.array(mat['z_star']) # Goal formation -> target locations
    
    return {
        'K': K,
        'L': L,
        'num_links': num_links,
        'num_agents': num_agents,
        'noise_covariance': noise_covariance,
        'timestep': timestep,
        'initial_positions': initial_positions,
        'target_positions': target_positions
    }