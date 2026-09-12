"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000

# TODO 1: test_rejects_negative_rate
def test_rejects_negative_rate():
    """Check that calling simulate(...) with a negative lam raises a ValueError."""
    with pytest.raises(ValueError):
        simulate(1000, -0.4)


# TODO 2: test_matches_law
def test_matches_law():
    """Check that simulation average matches physical law N0 * exp(-lam * t)."""
    n0, rate, dt, steps = 1000, 0.4, 0.05, 200
    t = dt * steps
    expected = n0 * np.exp(-rate * t)

    runs = [simulate(n0, rate, dt=dt, steps=steps, seed=seed)[-1] for seed in range(100)]
    avg_result = np.mean(runs)

    assert avg_result == pytest.approx(expected, rel=1e-1)