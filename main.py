import argparse
import csv
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from agent import QLearningAgent
from utils import plot_training, plot_sensitivity, save_gif


def train(alpha=.1, gamma=.99, episodes=30000, seed=42, epsilon_decay=.9997):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay, seed=seed)
    rewards, epsilons = [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed+episode)
        total = 0
        for _ in range(100):
            action = agent.act(state)
            nxt, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, nxt, terminated)
            state = nxt; total += reward
            if terminated or truncated: break
        rewards.append(total); agent.decay(); epsilons.append(agent.epsilon)
    env.close()
    return agent, rewards, epsilons


def evaluate(agent, episodes=500, seed=100000):
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True)
    successes, lengths = [], []
    for i in range(episodes):
        state, _ = env.reset(seed=seed+i)
        total = 0
        for step in range(100):
            state, reward, terminated, truncated, _ = env.step(agent.act(state, explore=False))
            total += reward
            if terminated or truncated: break
        successes.append(total > 0); lengths.append(step+1)
    env.close()
    return {'success_rate': float(np.mean(successes)), 'mean_steps': float(np.mean(lengths)), 'episodes': episodes}


def run(output='outputs', episodes=30000, seed=42, sensitivity=True):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    agent, rewards, epsilons = train(episodes=episodes, seed=seed)
    np.save(out/'q_table.npy', agent.q)
    plot_training(rewards, epsilons, out/'training.png')
    result = evaluate(agent)
    result.update({'environment':'FrozenLake-v1 8x8 slippery', 'algorithm':'Q-learning', 'training_episodes':episodes, 'alpha':.1, 'gamma':.99, 'epsilon_decay':.9997, 'seed':seed, 'last_1000_training_success_rate':float(np.mean(rewards[-1000:]))})
    env = gym.make('FrozenLake-v1', map_name='8x8', is_slippery=True, render_mode='rgb_array')
    try: result['successful_gif_recorded'] = save_gif(env, agent, out/'agent_demo.gif')
    finally: env.close()
    if sensitivity:
        rows = []
        for j, (a,g) in enumerate([(0.05,.95),(.1,.95),(.1,.99),(.2,.99)]):
            other, _, _ = train(alpha=a, gamma=g, episodes=episodes, seed=seed)
            rows.append({'alpha':a, 'gamma':g, **evaluate(other, seed=200000)})
        with (out/'sensitivity.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        plot_sensitivity(rows, out/'sensitivity.png')
    (out/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=30000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='outputs')
    p.add_argument('--skip-sensitivity', action='store_true')
    args=p.parse_args()
    run(args.output, args.episodes, args.seed, not args.skip_sensitivity)
