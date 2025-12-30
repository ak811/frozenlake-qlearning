import numpy as np

from policies import epsilon_greedy_action, greedy_action


def test_greedy_action_returns_argmax():
    q = np.array([[0.0, 1.0, 0.5]])
    assert greedy_action(q, 0) == 1


def test_epsilon_greedy_epsilon_zero_is_greedy():
    rng = np.random.default_rng(0)
    q = np.array([[0.1, 0.2, 0.3]])
    a = epsilon_greedy_action(q, 0, epsilon=0.0, rng=rng, n_actions=3)
    assert a == 2


def test_epsilon_greedy_epsilon_one_is_random_within_bounds():
    rng = np.random.default_rng(123)
    q = np.array([[10.0, -1.0, 0.0]])
    for _ in range(100):
        a = epsilon_greedy_action(q, 0, epsilon=1.0, rng=rng, n_actions=3)
        assert 0 <= a < 3
