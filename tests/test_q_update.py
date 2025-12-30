import numpy as np

from qlearning import q_update


def test_q_update_matches_formula():
    q = np.zeros((3, 2), dtype=float)
    q[1, 0] = 0.5
    q[2, 0] = 1.0
    q[2, 1] = 2.0  # next_max = 2.0

    state, action = 1, 0
    reward = 1.0
    next_state = 2
    lr = 0.8
    gamma = 0.95

    old = q[state, action]
    expected = old + lr * (reward + gamma * 2.0 - old)

    new_val = q_update(q, state, action, reward, next_state, lr, gamma)
    assert np.isclose(new_val, expected)
    assert np.isclose(q[state, action], expected)
