import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from agent import Agent
from filters import MovingAverageFilter, NoOpFilter

def test_measure_shape_and_self_zero():
    rng = np.random.default_rng(0)
    a0 = Agent(0, [0,0], [0,0], filter=NoOpFilter())
    a1 = Agent(1, [1,0], [0,0], filter=NoOpFilter())
    a2 = Agent(2, [0,1], [0,0], filter=NoOpFilter())
    meas = a0.measure([a0,a1,a2], noise_covariance=np.zeros((2,2)), rng=rng, use_noise=True)
    assert meas.shape == (3,2)
    np.testing.assert_allclose(meas[0], [0,0])  # self row
    np.testing.assert_allclose(meas[1], [-1,0])
    np.testing.assert_allclose(meas[2], [0,-1])

def test_moving_average_filter_warmup():
    f = MovingAverageFilter(window_size=3)
    m1 = np.array([[1.,0.],[0.,1.]])
    m2 = np.array([[2.,0.],[0.,2.]])
    m3 = np.array([[4.,0.],[0.,4.]])
    np.testing.assert_allclose(f.update(m1), m1)
    np.testing.assert_allclose(f.update(m2), (m1+m2)/2)
    np.testing.assert_allclose(f.update(m3), (m1+m2+m3)/3)

def test_apply_filter_requires_measure():
    a = Agent(0, [0,0], [0,0], filter=MovingAverageFilter(2))
    with pytest.raises(RuntimeError):
        a.apply_filter()
        

def test_filter_classes_different_instances():
    filter = MovingAverageFilter(2)
    agent1 = Agent(0, [0,0], [0,0], filter=filter)
    agent2 = Agent(1, [1,1], [0,0], filter=filter)
    assert agent1.filter is not agent2.filter