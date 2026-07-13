import numpy as np

np.set_printoptions(precision=15, suppress=True)

terminal_coord = [9, 9]

grid = np.zeros((10, 10), dtype=np.float64)
grid[:, :] = -1
grid[terminal_coord[0], terminal_coord[1]] = 1

start_coord = [np.random.randint(0, grid.shape[0]), np.random.randint(0, grid.shape[1])]

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
discount = 0.8

value_table = np.zeros_like(grid, dtype=np.float64)
value_table[:, :] = 0.0

# to store all the versions of thev value table for the stopping condition
# multiple epochs/ iteratoins
value_table_renditions = [value_table]


# returns state vlaue
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


def evaluate_policy(value_table_renditions, grid, delta=1e-10):

    k = 0

    while True:
        # go over every state and evalueate value for each of those states
        # and store in the value. table for the currnt iteration
        # rpeat till theupdate in the value tables is less than delta (some small number / threshold)

        curr_value_table = np.zeros_like(grid, dtype=np.float64)
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

final_value_table = value_table_renditions[-1]
fvt = final_value_table

curr_coord = start_coord

path = [curr_coord.copy()]

rows, cols = fvt.shape

while curr_coord != terminal_coord:

    vals = []
    if curr_coord[1] < grid.shape[1] - 1:
        vals.append((fvt[curr_coord[0], curr_coord[1] + 1], "right"))

    if curr_coord[1] > 0:
        vals.append((fvt[curr_coord[0], curr_coord[1] - 1], "left"))

    if curr_coord[0] > 0:
        vals.append((fvt[curr_coord[0] - 1, curr_coord[1]], "up"))

    if curr_coord[0] < grid.shape[0] - 1:
        vals.append((fvt[curr_coord[0] + 1, curr_coord[1]], "down"))

    curr_coord_action_max_value, best_action = max(vals, key=lambda x: x[0])

    # right
    if best_action == "right":
        curr_coord[1] += 1
    # left
    if best_action == "left":
        curr_coord[1] -= 1
    # down
    if best_action == "down":
        curr_coord[0] += 1
    # up
    if best_action == "up":
        curr_coord[0] -= 1

    path.append(curr_coord.copy())
    print(path)

print(path)


from visualization import visualize_path_on_grid

visualize_path_on_grid(
    fvt=fvt,
    path=path,
    terminal_coord=terminal_coord,
    cell_size=70,
    delay=400,
)
