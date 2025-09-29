import os, json, random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

class GridWorld:
    
    def __init__(self, width: int = 6, height: int = 6, start: Tuple[int, int] = (0,0), 
                 goal: Tuple[int, int] = (5,5), slip: float = 0.05, step_penalty: float = -0.01):
        self.width = width
        self.height = height
        self.start = start
        self.agent = list(start)
        self.goal = goal
        self.slip = slip
        self.step_penalty = step_penalty
        self.n_states = width * height
        self.action_space = [(0,1), (0,-1), (1,0), (-1,0)]
        self.n_actions = len(self.action_space)
        self._validate_positions()

    def _validate_positions(self):
        for pos, name in [(self.start, "start"), (self.goal, "goal")]:
            if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
                raise ValueError(f"{name} position {pos} is outside grid bounds")

    def reset(self) -> int:
        self.agent = list(self.start)
        return self._state_index(tuple(self.agent))

    def _state_index(self, pos: Tuple[int, int]) -> int:
        x, y = pos
        return y * self.width + x

    def _pos_from_index(self, idx: int) -> Tuple[int, int]:
        x = idx % self.width
        y = idx // self.width
        return (x, y)

    def step(self, action: int, extra_reward: float = 0.0, omit_reward: bool = False) -> Tuple[int, float, bool, dict]:
        if np.random.rand() < self.slip:
            action = np.random.randint(self.n_actions)
            
        dx, dy = self.action_space[action]
        new_x = np.clip(self.agent[0] + dx, 0, self.width-1)
        new_y = np.clip(self.agent[1] + dy, 0, self.height-1)
        self.agent = [new_x, new_y]
        
        done = tuple(self.agent) == self.goal
        
        if done:
            reward = 0.0 if omit_reward else (1.0 + extra_reward)
        else:
            reward = self.step_penalty
            
        return self._state_index(tuple(self.agent)), reward, done, {}

    def render_policy(self, policy: np.ndarray) -> np.ndarray:
        grid = np.full((self.height, self.width), ' ', dtype=object)
        arrows = {0: '→', 1: '←', 2: '↓', 3: '↑'}
        
        for s in range(self.n_states):
            pos = self._pos_from_index(s)
            if pos == self.goal:
                grid[pos[1], pos[0]] = 'G'
            elif pos == tuple(self.start):
                grid[pos[1], pos[0]] = 'S'
            else:
                grid[pos[1], pos[0]] = arrows[policy[s]]
                
        return grid

    def visualize_grid(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        for x in range(self.width + 1):
            ax.axvline(x, color='black', linewidth=1)
        for y in range(self.height + 1):
            ax.axhline(y, color='black', linewidth=1)
            
        start_x, start_y = self.start
        goal_x, goal_y = self.goal
        
        ax.add_patch(plt.Rectangle((start_x, start_y), 1, 1, fill=True, color='green', alpha=0.3))
        ax.add_patch(plt.Rectangle((goal_x, goal_y), 1, 1, fill=True, color='red', alpha=0.3))
        
        ax.text(start_x + 0.5, start_y + 0.5, 'S', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(goal_x + 0.5, goal_y + 0.5, 'G', ha='center', va='center', fontsize=20, fontweight='bold')
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title('GridWorld Environment')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        return ax


class ActorCritic:
    
    def __init__(self, n_states: int, n_actions: int, alpha_v: float = 0.2, 
                 alpha_pi: float = 0.05, gamma: float = 0.99):
        self.V = np.zeros(n_states)
        self.preferences = np.zeros((n_states, n_actions))
        self.alpha_v = alpha_v
        self.alpha_pi = alpha_pi
        self.gamma = gamma
        self.n_actions = n_actions

    def policy(self, state: int) -> np.ndarray:
        prefs = self.preferences[state]
        exp_p = np.exp(prefs - np.max(prefs))
        return exp_p / np.sum(exp_p)

    def select_action(self, state: int) -> int:
        probs = self.policy(state)
        return np.random.choice(self.n_actions, p=probs)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        target = reward + (0 if done else self.gamma * self.V[next_state])
        delta = target - self.V[state]
        
        self.V[state] += self.alpha_v * delta
        
        probs = self.policy(state)
        for a in range(self.n_actions):
            grad_log = (1 if a == action else 0) - probs[a]
            self.preferences[state, a] += self.alpha_pi * delta * grad_log
            
        return delta


class ExperimentConfig:
    
    def __init__(self, total_episodes: int = 300, max_steps: int = 150, 
                 window_size: int = 30, smoothing_window: int = 25):
        self.total_episodes = total_episodes
        self.max_steps = max_steps
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        
        self.start_episode = total_episodes // 2
        self.end_episode = self.start_episode + max(1, total_episodes // 10)


def train_with_events(env: GridWorld, agent: ActorCritic, config: ExperimentConfig, 
                     experiment: Optional[str] = None) -> Dict:
    returns = []
    td_errors = []
    events = []
    global_step = 0
    
    for episode in range(config.total_episodes):
        state = env.reset()
        episode_return = 0.0
        
        for step in range(config.max_steps):
            action = agent.select_action(state)
            
            extra_reward = 0.0
            omit_reward = False
            
            if experiment == "unexpected_reward" and config.start_episode <= episode < config.end_episode:
                extra_reward = 2.0
            elif experiment == "reward_omission" and config.start_episode <= episode < config.end_episode:
                omit_reward = True

            next_state, reward, done, _ = env.step(action, extra_reward=extra_reward, omit_reward=omit_reward)
            td_error = agent.update(state, action, reward, next_state, done)
            
            td_errors.append(td_error)
            
            if done:
                event_type = "normal_reward"
                if experiment == "unexpected_reward" and config.start_episode <= episode < config.end_episode:
                    event_type = "unexpected_reward"
                elif experiment == "reward_omission" and config.start_episode <= episode < config.end_episode:
                    event_type = "reward_omission"
                    
                events.append((global_step, event_type, episode, step))
            
            episode_return += reward
            state = next_state
            global_step += 1
            
            if done:
                break
                
        returns.append(episode_return)
    
    return {
        "returns": returns,
        "td_errors": td_errors,
        "events": events,
        "agent": agent
    }


def align_traces(td_errors: List[float], events: List[tuple], window_size: int = 30) -> Dict[str, np.ndarray]:
    traces = {}
    
    for step_idx, event_type, ep, t in events:
        start_idx = max(0, step_idx - window_size)
        end_idx = min(len(td_errors), step_idx + window_size + 1)
        trace = td_errors[start_idx:end_idx]
        
        if len(trace) < 2 * window_size + 1:
            left_pad = window_size - (step_idx - start_idx)
            right_pad = (2 * window_size + 1) - len(trace) - left_pad
            trace = [np.nan] * left_pad + trace + [np.nan] * right_pad
            
        traces.setdefault(event_type, []).append(trace)
    
    traces_mean = {}
    for event_type, event_traces in traces.items():
        arr = np.array(event_traces, dtype=float)
        traces_mean[event_type] = np.nanmean(arr, axis=0)
        
    return traces_mean


def plot_results(experiment_results: Dict, config: ExperimentConfig, output_dir: str):
    
    plt.figure(figsize=(12, 4))
    
    for i, (exp_name, results) in enumerate(experiment_results.items()):
        returns = results["returns"]
        
        plt.subplot(1, 3, i + 1)
        plt.plot(returns, alpha=0.7, label="Raw returns")
        
        if len(returns) > config.smoothing_window:
            smoothed = np.convolve(returns, np.ones(config.smoothing_window)/config.smoothing_window, 
                                 mode='valid')
            plt.plot(range(config.smoothing_window-1, len(returns)), smoothed, 
                    label=f"Smoothed", linewidth=2)
        
        if exp_name != "baseline":
            plt.axvspan(config.start_episode, config.end_episode, 
                       alpha=0.2, color='red', label='Manipulation')
        
        plt.title(f"{exp_name.replace('_', ' ').title()}")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_returns_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(15, 5))
    
    for i, (exp_name, results) in enumerate(experiment_results.items()):
        plt.subplot(1, 3, i + 1)
        
        traces_mean = align_traces(results["td_errors"], results["events"], config.window_size)
        
        for event_type, mean_trace in traces_mean.items():
            x = np.arange(-config.window_size, config.window_size + 1)
            plt.plot(x, mean_trace, label=event_type.replace('_', ' ').title(), linewidth=2)
        
        plt.axvline(0, linestyle='--', color='red', alpha=0.7, label='Event')
        plt.axhline(0, linestyle='-', color='black', alpha=0.5)
        
        plt.title(f"TD Errors: {exp_name.replace('_', ' ').title()}")
        plt.xlabel("Steps Relative to Goal")
        plt.ylabel("Mean TD Error")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_td_errors_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    np.random.seed(42)
    random.seed(42)
    
    output_dir = "/Users/david/Desktop/rl_dopamine_demo"
    config = ExperimentConfig()
    
    os.makedirs(output_dir, exist_ok=True)
    
    experiment_results = {}
    
    for exp_name in ["baseline", "unexpected_reward", "reward_omission"]:
        print(f"Running: {exp_name}")
        
        env = GridWorld(width=6, height=6, start=(0,0), goal=(5,5), slip=0.05)
        agent = ActorCritic(env.n_states, env.n_actions)
        
        results = train_with_events(env, agent, config, 
                                  experiment=exp_name if exp_name != "baseline" else None)
        experiment_results[exp_name] = results
    
    plot_results(experiment_results, config, output_dir)
    
    env_template = GridWorld(width=6, height=6, start=(0,0), goal=(5,5))
    
    for exp_name, results in experiment_results.items():
        policy = results["agent"].preferences.argmax(axis=1)
        policy_grid = env_template.render_policy(policy)
        
        filename = os.path.join(output_dir, f"learned_policy_{exp_name}.txt")
        with open(filename, "w") as f:
            f.write(f"Learned Policy - {exp_name}\n")
            f.write("=" * 40 + "\n")
            for row in policy_grid:
                f.write(" ".join(row) + "\n")
    
    metadata = {
        "experiments": list(experiment_results.keys()),
        "config": {
            "total_episodes": config.total_episodes,
            "max_steps": config.max_steps,
            "window_size": config.window_size,
            "manipulation_episodes": f"{config.start_episode}-{config.end_episode}"
        }
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    env_viz = GridWorld()
    fig, ax = plt.subplots(figsize=(6, 6))
    env_viz.visualize_grid(ax)
    plt.savefig(os.path.join(output_dir, "gridworld_environment.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Done!")
    print(f"Results in: {output_dir}")


if __name__ == "__main__":
    main()