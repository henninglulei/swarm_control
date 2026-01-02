from agent import Agent
from data.read_data import read_data
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from simulation import SimulationConfig, Simulation
from filters import MaximumLikelihoodEstimator, NoOpFilter, RecursiveLeastSquaresFilter

data_dict = read_data()
K, L, num_links, num_agents, noise_covariance, dt, initial_positions, target_positions = data_dict.values()

K = 30_000


rls = SimulationConfig(noise_covariance=noise_covariance,
                    filter=RecursiveLeastSquaresFilter(forgetting_factor=0.99,
                                                    noise_covariance=noise_covariance,
                                                    alpha=1e3),
                    desc='RLS Filter, Forgetting Factor 0.999',
                    dt=dt,
                    T = 1)


rls3 = SimulationConfig(noise_covariance=noise_covariance,
                    filter=RecursiveLeastSquaresFilter(forgetting_factor=0.99,
                                                    noise_covariance=noise_covariance,
                                                    alpha=1e3),
                    desc='RLS Filter, Forgetting Factor 0.999',
                    dt=dt,
                    T = 3)



rls10 = SimulationConfig(noise_covariance=noise_covariance,
                    filter=RecursiveLeastSquaresFilter(forgetting_factor=0.99,
                                                    noise_covariance=noise_covariance,
                                                    alpha=1e3),
                    desc='RLS Filter, Forgetting Factor 0.999',
                    dt=dt,
                    T = 10)


rls30 = SimulationConfig(noise_covariance=noise_covariance,
                    filter=RecursiveLeastSquaresFilter(forgetting_factor=0.99,
                                                    noise_covariance=noise_covariance,
                                                    alpha=1e3),
                    desc='RLS Filter, Forgetting Factor 0.999',
                    dt=dt,
                    T = 30)

                    

no_op = SimulationConfig(noise_covariance=noise_covariance,
                    filter=NoOpFilter(),
                    desc='No Op Filter',
                    dt=dt,
                    T = 1)

mle3 = SimulationConfig(noise_covariance=noise_covariance,
                    filter=MaximumLikelihoodEstimator(),
                    desc='Maximum Likelihood Estimator, T = 3',
                    dt=dt,
                    T = 3)

mle10 = SimulationConfig(noise_covariance=noise_covariance,
                    filter=MaximumLikelihoodEstimator(),
                    desc='Maximum Likelihood Estimator, T = 10',
                    dt=dt,
                    T = 10)

mle30 = SimulationConfig(noise_covariance=noise_covariance,
                    filter=MaximumLikelihoodEstimator(),
                    desc='Maximum Likelihood Estimator, T = 30',
                    dt=dt,
                    T = 30)

mle100 = SimulationConfig(noise_covariance=noise_covariance,
                    filter=MaximumLikelihoodEstimator(),
                    desc='Maximum Likelihood Estimator, T = 100',
                    dt=dt,
                    T = 100)

configurations = [rls3]

seeds = np.arange(100)
error_dict = {config.desc: [] for config in configurations}
for seed in tqdm.tqdm(seeds):

    for config in configurations:
        agents = [Agent(id=i, initial_position=initial_positions[i], target_position=target_positions[i], filter=config.filter) for i in range(num_agents)]
        
        sim = Simulation(agents=agents, L=L, config=config, seed=seed)
        errors = sim.run(num_steps=K)
        error_dict[config.desc].append(errors)

with open('error_dict.npy', 'wb') as f:
    np.save(f, error_dict)
        