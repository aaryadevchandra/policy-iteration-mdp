# Gridworld Policy Evaluation

A simple reinforcement learning project that implements **policy evaluation** on a `10 × 10` Gridworld using NumPy.

The agent starts from a random cell and evaluates a random policy where each action has equal probability:

- left: `0.25`
- right: `0.25`
- up: `0.25`
- down: `0.25`

After evaluating the state-value function, the agent extracts a greedy path by repeatedly moving to the neighboring state with the highest value.

---

## Project Overview

This project demonstrates the core idea of **Dynamic Programming in Reinforcement Learning**.

The value of each state is computed using the Bellman expectation equation:

```math
V(s) = \sum_a \pi(a|s) [r + \gamma V(s')]
```

where:

- `V(s)` is the value of state `s`
- `π(a|s)` is the probability of taking action `a` in state `s`
- `r` is the reward
- `γ` is the discount factor
- `s'` is the next state

The value table is updated repeatedly until convergence.

---

## Gridworld Setup

The environment is a `10 × 10` grid.

```text
S . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . .
. . . . . . . . . T
```

- `S` is the random start state
- `T` is the terminal state
- Every non-terminal cell has reward `-1`
- Terminal cell has reward `+1`
- Discount factor used: `0.5`

---

## Features

- Implements policy evaluation from scratch
- Uses NumPy for value computation
- Stores all value-table updates across iterations
- Visualizes value-table evolution using OpenCV
- Extracts a greedy path from the final value table
- Saves the final path visualization as a local video
- Useful for understanding Bellman updates visually

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/gridworld-policy-evaluation.git
cd gridworld-policy-evaluation
```

Install dependencies:

```bash
pip install numpy opencv-python
```

---

## Running the Project

Run the main script:

```bash
python main.py
```

This will:

1. Create a `10 × 10` Gridworld
2. Initialize a random policy
3. Run iterative policy evaluation
4. Visualize the value-table updates
5. Extract a greedy path from the final value table
6. Save the path visualization video

---

## Visualization

### Value Table Evolution

The value table is visualized as a heatmap.

Each cell shows the estimated value of that state. The values update over multiple iterations until convergence.

```python
visualize_value_iteration(value_table_renditions)
```

---

### Greedy Path Visualization

After policy evaluation, the agent follows the neighboring state with the highest value.

The generated path video is saved locally as:

```text
assets/path_visualization.mp4
```

To include the video in the README:

```html
<video src="assets/path_visualization.mp4" controls width="600"></video>
```

For better GitHub compatibility, convert the video to GIF and use:

```md
![Path Visualization](assets/path_visualization.gif)
```

---

## Example Output

```text
start: [2, 4]
terminal: [9, 9]

Final path:
[[2, 4], [3, 4], [4, 4], [5, 4], ..., [9, 9]]
```

The exact path may change because the start position is randomly selected.

---

## Project Structure

```text
gridworld-policy-evaluation/
│
├── main.py
├── visualization.py
├── assets/
│   └── path_visualization.mp4
│
└── README.md
```

---

## Core Algorithm

### 1. Initialize Value Table

```python
value_table = np.zeros_like(grid)
```

All state values are initialized to zero.

---

### 2. Evaluate Policy

For each state, the value is updated based on the expected return under the current policy.

```math
V(s) = \sum_a \pi(a|s) [r + \gamma V(s')]
```

Since the policy is random, each action contributes equally.

---

### 3. Repeat Until Convergence

The value table is updated until the difference between two consecutive value tables becomes very small.

```python
if np.linalg.norm(V_new - V_old) < delta:
    break
```

---

### 4. Extract Greedy Path

After evaluation, the agent chooses the neighboring state with the highest value.

```python
best_action = argmax(V(next_state))
```

This creates a greedy path from the start state to the terminal state.

---

## Important Note

The value table is produced by evaluating a **random policy**.

The greedy path extraction step is not the same as following the random policy. It is closer to a simple policy improvement step because the agent chooses the best neighboring value.

To make this a full reinforcement learning control algorithm, the next step would be implementing:

- Policy Iteration
- Value Iteration
- Monte Carlo Control
- Q-Learning

---

## Concepts Covered

- Markov Decision Processes
- State-value functions
- Bellman expectation equation
- Iterative policy evaluation
- Discounted returns
- Greedy policy improvement
- Gridworld visualization

---

## Future Improvements

- Add proper policy iteration
- Add value iteration
- Add stochastic transitions
- Add obstacles/walls
- Add multiple terminal states
- Add GIF export directly
- Compare random policy vs optimal policy
- Visualize policy arrows for each state

---

## License

This project is for learning and experimentation.
