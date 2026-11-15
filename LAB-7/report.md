# Lab 7: Reinforcement Learning and Deep RL

## Implementation

[`rl_lab.py`](./rl_lab.py) implements the specified five-by-five deterministic
warehouse, tabular Q-learning, a replay-buffer DQN, and the two-robot
simultaneous-motion environment. It uses no path planner or RL library.

The single-robot MDP state is the current `(row, column)` (encoded as
`5 * row + column`), the action set is `(up, down, left, right, wait)`, and
the transition is deterministic. The fixed map and goal make position
Markov-sufficient. The shortest route is four right moves through `(2, 1)`,
`(2, 2)`, and `(2, 3)`. Search would calculate a route; Q-learning estimates
action values from reward-bearing experience.

The table has shape `25 x 5`, learning rate `alpha=0.2`, discount `gamma=0.95`,
seed 7, 1,500 episodes, and a 40-step truncation limit. Epsilon decreases
linearly from 1 to 0.05 over 1,200 episodes. Evaluation uses greedy
tie-breaking, performs no updates, and keeps the pre-reset next state in every
update. A 40-step cutoff is truncation, not termination, so it still bootstraps.

## Validation

Running `python rl_lab.py --skip-dqn` checks invalid moves, rewards, goal
termination, truncation, synthetic TD targets, collision/swap detection, and
the no-repeat-arrival rule. The checked non-terminal target is
`1 + 0.95*2 = 2.9`, giving `Q <- 0.4 + 0.2*(2.9-0.4) = 0.9`.
For a terminal reward of 5 the target is 5, independent of the next state,
so an initially zero entry becomes 1.0.

The script prints the learned greedy trajectory and a 100-run uniform-random
baseline. It saves the 50-episode moving-average plot as
`q_learning_curves.png`. Results should be regenerated rather than copied:
random-policy and neural-network outcomes depend on the stated seed and
installed package versions.

## DQN

The DQN uses a 25-entry one-hot state, two 64-unit ReLU layers, and five raw
Q-value outputs. `ReplayBuffer` stores transitions, `dqn_loss` gathers
`Q_theta(s, recorded_action)`, and a `torch.no_grad()` target network computes
`r + gamma*(1-terminated)*max Q_target(s')`. It collects 500 steps, samples
batches of 64, updates every four steps, synchronizes every 250 optimizer
updates, uses Adam at 0.001, and trains for 12,000 steps.

## Two robots

The state is the ordered pair `(pA,pB)`, represented for a DQN by two
concatenated 25-entry one-hot vectors. A joint action is encoded
`5*aA+aB`, giving `5^2=25` outputs. Both proposals are computed from the old
state. Same-cell and swap attempts are collisions; a robot at its own goal
stays and cannot earn another arrival reward. Collision has reward -10.
Otherwise the team receives `-0.1`, invalid-move penalties for non-arrived
robots, +5 per newly reached goal, and +5 when both have arrived.

Observing only A is insufficient because A's safe action depends on B's
position and intended simultaneous move. For three robots the joint output
would be `5^3=125`. A collision-free hand-designed route must make one robot
yield outside the doorway; such a route is not substituted for a learned
trajectory.

## Reflection

The map, coordinates, actions, rewards, termination/truncation distinction,
state encodings, network sizes, replay settings, and target update rule were
specified before implementation. Tests enforce these details rather than
trusting a plausible trajectory. The most important implementation correction
is to use `terminated` (not the 40-step `truncated` flag) when suppressing
bootstrapping. The two-robot environment also tests simultaneous proposals,
same-cell collisions, swaps, and persistent occupancy at a goal.
