import numpy as np

grid = np.zeros((10, 10), dtype=np.float32)
grid[:, :] = -1
grid[-1, -1] = 1

# define policy
# say we have a random policy, each with 1/4th prob
state_policies = {}

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        state_policies[f"{i}_{j}"] = {
            "left": 0.25,
            "right": 0.25,
            "up": 0.25,
            "down": 0.25,
        }

# policy evaluation
# run through rvery state and finf the value of that particulat state
# value function is expected return
start_coord = (np.random.randint(0, grid.shape[0]), np.random.randint(0, grid.shape[1]))
discount = 0.5

value_table = np.zeros_like(grid)
value_table[:, :] = 0.0

# to store all the versions of thev value table for the stopping condition
value_table_renditions = [value_table]


def state_value(
    value_table,
    coord: tuple[int, int],
    discount=discount,
    state_policies=state_policies,
    grid=grid,
) -> float:
    value = 0.0
    coord_i, coord_j = coord
    action_probs = state_policies[f"{coord_i}_{coord_j}"]
    for action in action_probs:
        if action == "left" and coord_j > 0:
            value += action_probs[action] * (
                grid[coord_i, coord_j] + discount * value_table[coord_i, coord_j - 1]
            )
        if action == "right" and coord_j < grid.shape[1] - 1:
            value += action_probs[action] * (
                grid[coord_i, coord_j] + discount * value_table[coord_i, coord_j + 1]
            )
        if action == "up" and coord_i > 0:
            value += action_probs[action] * (
                grid[coord_i, coord_j] + discount * value_table[coord_i - 1, coord_j]
            )
        if action == "down" and coord_i < grid.shape[0] - 1:
            value += action_probs[action] * (
                grid[coord_i, coord_j] + discount * value_table[coord_i + 1, coord_j]
            )

    return value


def evaluate_policy(value_table_renditions, grid, delta=0.5):

    k = 0

    while True:
        curr_value_table = np.zeros_like(grid)
        if (
            k > 2
            and np.linalg.norm(
                value_table_renditions[k - 2] - value_table_renditions[k - 1]
            )
            < delta
        ):
            break

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if i == grid.shape[0] - 1 and j == grid.shape[1] - 1:
                    continue
                curr_value_table[i, j] = state_value(value_table_renditions[-1], (i, j))

        value_table_renditions.append(curr_value_table.copy())
        k += 1

    return value_table_renditions


value_table_renditions = evaluate_policy(value_table_renditions, grid)
from visualization import visualize_value_iteration

visualize_value_iteration(value_table_renditions)
