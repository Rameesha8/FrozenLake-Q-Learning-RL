from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def moving_average(values, window=200):
    values = np.asarray(values, dtype=float)
    if not len(values): return values
    return np.convolve(values, np.ones(window) / window, mode='valid') if len(values) >= window else np.array([])


def plot_training(rewards, epsilons, path):
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(np.arange(len(rewards)), rewards, alpha=.12, label='Episode reward')
    ma = moving_average(rewards)
    if len(ma): ax[0].plot(np.arange(len(ma)) + 199, ma, label='200-episode success rate', linewidth=2)
    ax[0].set_ylabel('Reward / success'); ax[0].legend(); ax[0].grid(alpha=.25)
    ax[1].plot(epsilons); ax[1].set_ylabel('Epsilon'); ax[1].set_xlabel('Training episode'); ax[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_sensitivity(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"a={r['alpha']}, g={r['gamma']}" for r in rows]
    ax.bar(labels, [r['success_rate'] for r in rows]); ax.set_ylim(0, 1)
    ax.set_ylabel('Evaluation success rate'); ax.tick_params(axis='x', rotation=25)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_gif(env, agent, path, seed=2026, max_steps=100, attempts=30):
    """Record a successful greedy episode if one is found; otherwise record the best attempt."""
    best_frames, best_reward, success = [], -1, False
    for i in range(attempts):
        state, _ = env.reset(seed=seed+i)
        frames = [env.render()]
        total = 0
        for _ in range(max_steps):
            action = agent.act(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render()); total += reward
            if terminated or truncated: break
        if total > best_reward:
            best_frames, best_reward = frames, total
        if total > 0:
            success = True; break
    if not best_frames: raise RuntimeError('No frames were rendered')
    imageio.mimsave(path, best_frames, duration=0.20, loop=0)
    return success
