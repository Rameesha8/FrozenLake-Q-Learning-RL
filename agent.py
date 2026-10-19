import numpy as np

class QLearningAgent:
    """Tabular off-policy Q-learning, implemented without RL frameworks."""
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9997, seed=42):
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.rng = np.random.default_rng(seed)

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        values = self.q[state]
        return int(self.rng.choice(np.flatnonzero(np.isclose(values, values.max())))) if explore else int(np.argmax(values))

    def update(self, state, action, reward, next_state, terminated):
        target = reward if terminated else reward + self.gamma * np.max(self.q[next_state])
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
