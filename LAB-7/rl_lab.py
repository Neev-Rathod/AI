"""Inspectable RL solutions for the CS F407 warehouse laboratory.

The file deliberately contains the environment, learning algorithms, tests,
and evaluation code instead of hiding them behind an RL framework.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # The tabular part remains usable without PyTorch.
    torch = None
    nn = None

Position = tuple[int, int]
ACTIONS: tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
ACTION_NAMES = ("up", "down", "left", "right", "wait")
OBSTACLES = frozenset({(0, 2), (1, 2), (3, 2), (4, 2)})
START, GOAL = (2, 0), (2, 4)
GAMMA = 0.95


@dataclass(frozen=True)
class StepResult:
    position: Position
    reward: float
    terminated: bool
    truncated: bool
    invalid: bool = False


class Warehouse:
    """The deterministic five-by-five single-robot task."""

    def __init__(self, start: Position = START):
        self.start = start
        self.position = start

    def reset(self, position: Optional[Position] = None) -> Position:
        self.position = self.start if position is None else position
        return self.position

    def step(self, action: int) -> StepResult:
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"action must be in [0, 4], got {action}")
        old = self.position
        dr, dc = ACTIONS[action]
        candidate = (old[0] + dr, old[1] + dc)
        invalid = (
            candidate[0] not in range(5)
            or candidate[1] not in range(5)
            or candidate in OBSTACLES
        )
        self.position = old if invalid else candidate
        if self.position == GOAL:
            return StepResult(self.position, 10.0, True, False, invalid)
        return StepResult(self.position, -1.0 if invalid else -0.1, False, False, invalid)


def state_index(position: Position) -> int:
    return 5 * position[0] + position[1]


def epsilon_greedy(q_values: np.ndarray, state: int, epsilon: float,
                   rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[1]))
    best = np.flatnonzero(q_values[state] == q_values[state].max())
    return int(rng.choice(best))


def q_update(q_values: np.ndarray, state: int, action: int, reward: float,
             next_state: int, terminated: bool, alpha: float = 0.2,
             gamma: float = GAMMA) -> float:
    target = reward if terminated else reward + gamma * float(q_values[next_state].max())
    q_values[state, action] += alpha * (target - q_values[state, action])
    return target


def epsilon_schedule(episode: int, decay_episodes: int = 1200) -> float:
    return max(0.05, 1.0 - 0.95 * episode / decay_episodes)


def train_q_table(episodes: int = 1500, seed: int = 7):
    rng = np.random.default_rng(seed)
    q_values = np.zeros((25, 5), dtype=np.float64)
    returns, successes, epsilons = [], [], []
    for episode in range(episodes):
        env = Warehouse()
        state = state_index(env.reset())
        total, reached = 0.0, False
        for _ in range(40):
            epsilon = epsilon_schedule(episode)
            action = epsilon_greedy(q_values, state, epsilon, rng)
            result = env.step(action)
            next_state = state_index(result.position)
            q_update(q_values, state, action, result.reward, next_state,
                     result.terminated)
            total += result.reward
            state = next_state
            if result.terminated:
                reached = True
                break
        returns.append(total)
        successes.append(reached)
        epsilons.append(epsilon)
    return q_values, {"returns": returns, "successes": successes, "epsilons": epsilons}


def evaluate_q_table(q_values: np.ndarray, start: Position = START,
                     max_steps: int = 40):
    env = Warehouse(start)
    env.reset()
    positions = [env.position]
    invalid_moves = 0
    total = 0.0
    terminated = truncated = False
    for _ in range(max_steps):
        state = state_index(env.position)
        action = int(np.flatnonzero(q_values[state] == q_values[state].max())[0])
        result = env.step(action)
        positions.append(result.position)
        total += result.reward
        invalid_moves += result.invalid
        if result.terminated:
            terminated = True
            break
    else:
        truncated = True
    return {"positions": positions, "steps": len(positions) - 1, "return": total,
            "success": terminated, "terminated": terminated, "truncated": truncated,
            "invalid_moves": invalid_moves}


def random_evaluation(runs: int = 100, seed: int = 7):
    rng = np.random.default_rng(seed)
    outcomes = []
    for _ in range(runs):
        env = Warehouse()
        invalid = total = 0
        success = False
        for step in range(1, 41):
            result = env.step(int(rng.integers(5)))
            total += result.reward
            invalid += result.invalid
            if result.terminated:
                success = True
                break
        outcomes.append((success, step, invalid, total))
    successful = [x[1] for x in outcomes if x[0]]
    return {"successes": sum(x[0] for x in outcomes), "timeouts": runs - sum(x[0] for x in outcomes),
            "success_fraction": sum(x[0] for x in outcomes) / runs,
            "mean_success_steps": float(np.mean(successful)) if successful else None,
            "invalid_moves": sum(x[2] for x in outcomes)}


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(self, *transition):
        self.data.append(transition)

    def sample(self, size: int, rng: np.random.Generator):
        indices = rng.choice(len(self.data), size, replace=False)
        return [self.data[int(i)] for i in indices]

    def __len__(self):
        return len(self.data)


class DQN(nn.Module if nn is not None else object):
    def __init__(self, input_size: int = 25, output_size: int = 5):
        if nn is None:
            raise RuntimeError("DQN requires PyTorch; install torch to run this section")
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, output_size))

    def forward(self, state):
        return self.network(state)


class JointDQN(DQN):
    """The same architecture for two concatenated one-hot robot positions."""

    def __init__(self):
        super().__init__(input_size=50, output_size=25)


def one_hot(index: int, size: int = 25) -> np.ndarray:
    value = np.zeros(size, dtype=np.float32)
    value[index] = 1.0
    return value


def dqn_loss(online: nn.Module, target: nn.Module, batch, gamma: float = GAMMA):
    states, actions, rewards, next_states, terminated = zip(*batch)
    device = next(online.parameters()).device
    states_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
    next_t = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    terminated_t = torch.tensor(terminated, dtype=torch.float32, device=device)
    chosen = online(states_t).gather(1, actions_t[:, None]).squeeze(1)
    with torch.no_grad():
        target_values = rewards_t + gamma * (1.0 - terminated_t) * target(next_t).max(1).values
    return nn.functional.mse_loss(chosen, target_values)


def train_dqn(steps: int = 12000, seed: int = 7):
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online, target = DQN(), DQN()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
    replay = ReplayBuffer(20000)
    env, state = Warehouse(), None
    state = state_index(env.reset())
    updates = 0
    for step in range(steps):
        epsilon = max(0.05, 1.0 - 0.95 * step / (0.8 * steps))
        if rng.random() < epsilon:
            action = int(rng.integers(5))
        else:
            with torch.no_grad():
                action = int(online(torch.tensor(one_hot(state))).argmax())
        result = env.step(action)
        next_state = state_index(result.position)
        replay.add(one_hot(state), action, result.reward, one_hot(next_state), result.terminated)
        state = next_state
        if result.terminated or (step + 1) % 40 == 0:
            state = state_index(env.reset())
        if step >= 500 and (step + 1) % 4 == 0:
            batch = replay.sample(64, rng)
            loss = dqn_loss(online, target, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            if updates % 250 == 0:
                target.load_state_dict(online.state_dict())
    return online


def evaluate_dqn(model: nn.Module, start: Position = START, max_steps: int = 40):
    env = Warehouse(start)
    positions, invalid = [env.reset()], 0
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            action = int(model(torch.tensor(one_hot(state_index(env.position)))).argmax())
        result = env.step(action)
        positions.append(result.position)
        invalid += result.invalid
        if result.terminated:
            return {"positions": positions, "steps": step, "success": True,
                    "truncated": False, "invalid_moves": invalid}
    return {"positions": positions, "steps": max_steps, "success": False,
            "truncated": True, "invalid_moves": invalid}


def joint_one_hot(positions: tuple[Position, Position]) -> np.ndarray:
    return np.concatenate((one_hot(state_index(positions[0])),
                           one_hot(state_index(positions[1]))))


def train_joint_dqn(steps: int = 60000, seed: int = 7):
    """Train the two-robot controller with the same replay/target settings."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online, target = JointDQN(), JointDQN()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=0.001)
    replay = ReplayBuffer(20000)
    env = TwoRobotWarehouse()
    positions = env.reset()
    for step in range(steps):
        epsilon = max(0.05, 1.0 - 0.95 * step / (0.8 * steps))
        state = joint_one_hot(positions)
        if rng.random() < epsilon:
            action = int(rng.integers(25))
        else:
            with torch.no_grad():
                action = int(online(torch.tensor(state)).argmax())
        result = env.step(action)
        next_positions = result.positions
        replay.add(state, action, result.reward, joint_one_hot(next_positions),
                   result.terminated)
        positions = env.reset() if result.terminated or (step + 1) % 40 == 0 else next_positions
        if step >= 500 and (step + 1) % 4 == 0:
            loss = dqn_loss(online, target, replay.sample(64, rng))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1 - 500) // 4 % 250 == 0:
                target.load_state_dict(online.state_dict())
    return online


def evaluate_joint_dqn(model: nn.Module, positions=((2, 0), (2, 4)), max_steps=40):
    env = TwoRobotWarehouse(positions)
    current = env.reset()
    trajectory, invalid = [current], 0
    for step in range(1, max_steps + 1):
        with torch.no_grad():
            action = int(model(torch.tensor(joint_one_hot(current))).argmax())
        result = env.step(action)
        invalid += result.invalid_moves
        trajectory.append(result.positions)
        current = result.positions
        if result.terminated:
            return {"trajectory": trajectory, "steps": step, "success": not result.collision,
                    "collision": result.collision, "truncated": False, "invalid_moves": invalid}
    return {"trajectory": trajectory, "steps": max_steps, "success": False,
            "collision": False, "truncated": True, "invalid_moves": invalid}


@dataclass(frozen=True)
class JointStep:
    positions: tuple[Position, Position]
    reward: float
    terminated: bool
    truncated: bool
    collision: bool
    invalid_moves: int


class TwoRobotWarehouse:
    def __init__(self, positions: tuple[Position, Position] = (START, GOAL),
                 goals: tuple[Position, Position] = (GOAL, START)):
        self.start, self.goals = positions, goals
        self.positions = positions
        self.arrived = [positions[i] == goals[i] for i in range(2)]

    def reset(self, positions=None):
        self.positions = self.start if positions is None else positions
        self.arrived = [self.positions[i] == self.goals[i] for i in range(2)]
        return self.positions

    def _propose(self, position, action, arrived):
        if arrived:
            return position, False
        dr, dc = ACTIONS[action]
        candidate = (position[0] + dr, position[1] + dc)
        invalid = candidate[0] not in range(5) or candidate[1] not in range(5) or candidate in OBSTACLES
        return (position if invalid else candidate), invalid

    def step(self, joint_action: int) -> JointStep:
        if not 0 <= joint_action < 25:
            raise ValueError("joint action must be in [0, 24]")
        action_a, action_b = divmod(joint_action, 5)
        old_a, old_b = self.positions
        new_a, invalid_a = self._propose(old_a, action_a, self.arrived[0])
        new_b, invalid_b = self._propose(old_b, action_b, self.arrived[1])
        collision = new_a == new_b or (new_a == old_b and new_b == old_a and old_a != old_b)
        if collision:
            return JointStep(self.positions, -10.0, True, False, True, 0)
        self.positions = (new_a, new_b)
        newly_arrived = sum(not self.arrived[i] and self.positions[i] == self.goals[i] for i in range(2))
        self.arrived = [self.arrived[i] or self.positions[i] == self.goals[i] for i in range(2)]
        complete = all(self.arrived)
        reward = -0.1 - int(invalid_a and not self.arrived[0]) - int(invalid_b and not self.arrived[1])
        reward += 5 * newly_arrived + (5 if complete else 0)
        return JointStep(self.positions, reward, complete, False, False,
                         int(invalid_a and not self.arrived[0]) + int(invalid_b and not self.arrived[1]))


def moving_average(values: Iterable[float], window: int = 50) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def run_tests() -> None:
    expected = [((0, 0), 0, (0, 0), -1), ((1, 1), 3, (1, 1), -1),
                ((2, 1), 3, (2, 2), -0.1), ((2, 1), 4, (2, 1), -0.1),
                ((2, 3), 3, (2, 4), 10)]
    for position, action, next_position, reward in expected:
        result = Warehouse().step(action) if False else None
        env = Warehouse(position)
        result = env.step(action)
        assert (result.position, result.reward, result.terminated) == (next_position, reward, reward == 10)
    q = np.zeros((25, 5))
    q[0, 0] = 0.4
    q[1] = 2
    assert q_update(q, 0, 0, 1, 1, False) == 1 + GAMMA * 2
    assert np.isclose(q[0, 0], 0.4 + 0.2 * (1 + GAMMA * 2 - 0.4))
    q[0, 1] = 0
    q_update(q, 0, 1, 5, 1, True)
    assert np.isclose(q[0, 1], 1.0)
    env = Warehouse()
    for _ in range(40):
        result = env.step(4)
    assert not result.terminated
    assert env.position == START
    two = TwoRobotWarehouse(((2, 1), (2, 3)))
    assert two.step(3 * 5 + 2).collision  # right/left, same-cell attempt
    two = TwoRobotWarehouse(((2, 1), (2, 2)))
    assert two.step(3 * 5 + 2).collision  # right/left swap
    two = TwoRobotWarehouse(((2, 1), GOAL))
    result = two.step(4 * 5 + 4)
    assert not result.collision and result.positions == ((2, 1), GOAL)
    print("All environment, truncation, TD-update, and collision tests passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    run_tests()
    q_values, history = train_q_table()
    q_result = evaluate_q_table(q_values)
    print("Q-table greedy:", q_result)
    print("Random baseline:", random_evaluation())
    if not args.skip_dqn and torch is not None:
        model = train_dqn()
        print("DQN greedy:", evaluate_dqn(model))
    try:
        import matplotlib.pyplot as plt
        args.output.mkdir(parents=True, exist_ok=True)
        plt.plot(moving_average(history["returns"]), label="return (50-episode mean)")
        plt.plot(moving_average(history["successes"]), label="success (50-episode mean)")
        plt.legend()
        plt.xlabel("episode")
        plt.tight_layout()
        plt.savefig(args.output / "q_learning_curves.png", dpi=150)
        plt.close()
    except ImportError:
        print("Matplotlib unavailable; skipped the training plot.")


if __name__ == "__main__":
    main()
