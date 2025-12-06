import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from agent import Agent
from simulation import Simulation, SimulationConfig
from filters import NoOpFilter, MovingAverageFilter

def test_anchors_do_not_move():
    K, dt = 3, 0.1
    L = np.eye(3)  # no coupling, so non-anchors also stay still
    initial = np.array([[0,0],[1,0],[0,1]], dtype=float)
    target = initial.copy()
    agents = [Agent(i, initial[i], target[i], filter=NoOpFilter()) for i in range(3)]
    cfg = SimulationConfig(noise_covariance=np.zeros((2,2)), filter=NoOpFilter(), desc="noop", dt=dt)
    sim = Simulation(agents, L, cfg, seed=123)
    errs = sim.run(num_steps=K)
    np.testing.assert_allclose(errs, np.zeros(K))
    for i in range(3):
        np.testing.assert_allclose(agents[i].position, initial[i])

def test_reproducible_run():
    K, dt = 5, 0.1
    L = np.zeros((2,2))
    initial = np.array([[0,0],[1,0]], dtype=float)
    target = np.array([[0,0],[0,0]], dtype=float)
    cfg = SimulationConfig(noise_covariance=np.zeros((2,2)), filter=NoOpFilter(), desc="noop", dt=dt)

    def run_once(seed):
        agents = [Agent(i, initial[i], target[i], filter=NoOpFilter()) for i in range(2)]
        sim = Simulation(agents, L, cfg, seed=seed)
        return sim.run(num_steps=K)

    e1 = run_once(42)
    e2 = run_once(42)
    np.testing.assert_allclose(e1, e2)
    

def test_golden_convergence():
    from data.read_data import read_data
    K, L, _, _, noise_cov, dt, initial, target = read_data().values()
    K = 30  # shorten for test speed
    
    agents = [Agent(i, initial[i], target[i], filter=MovingAverageFilter(5)) for i in range(len(initial))]
    cfg = SimulationConfig(
        noise_covariance=noise_cov * 200,
        filter=MovingAverageFilter(5),
        desc="golden",
        dt=dt
    )
    sim = Simulation(agents, L, cfg, seed=42)
    errors = sim.run(num_steps=K)
    
    print(np.array([agent.position for agent in agents]))
    
    expected_final_positions = np.array([[2.0, 0.0], [1.0, 1.0], [1.0, -1.0], [1.2162748809647952, -0.9576685630618925], [1.6536719343457027, 0.5569518659184408], [0.8072353419299619, -1.77976958863653], [1.1897539966256785, -0.7764461227049858]])
    final_positions = np.array([agent.position for agent in agents])
    np.testing.assert_allclose(final_positions, expected_final_positions, rtol=1e-2)
    
    # Reference generated 2025-12-05 with current implementation
    expected_errors = np.array([28.979595600350727, 28.817512016313014, 28.79459850749182, 28.75328056385725, 28.712556268943057, 28.65005868609594, 28.642788431915605, 28.51647492893312, 28.383821722511797, 28.281785722383297, 28.253419286011223, 28.173474649834276, 28.10924135554787, 27.996412268583548, 27.868528223505926, 27.69789775344219, 27.51588703126984, 27.381440471196004, 27.293092944398357, 27.225118904361974, 27.15055784346773, 27.070797600521892, 26.993295437115275, 26.903722616461277, 26.754691342506284, 26.579193742115162, 26.513529120259793, 26.423572348466905, 26.361139992857762, 26.308738216917288])
    np.testing.assert_allclose(errors, expected_errors, rtol=1e-2)
