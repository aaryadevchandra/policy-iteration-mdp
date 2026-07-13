import cv2
import numpy as np


def visualize_value_iteration(value_table_renditions, cell_size=70, delay=300):
    """
    Visualizes how the value table changes over iterations.

    value_table_renditions: list of 2D numpy arrays
    cell_size: pixel size of each grid cell
    delay: milliseconds between frames
    """

    # Use global min/max so colors are consistent across all iterations
    all_values = np.stack(value_table_renditions)
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    rows, cols = value_table_renditions[0].shape

    for iteration, value_table in enumerate(value_table_renditions):

        # Normalize values to [0, 255]
        normalized = (value_table - global_min) / (global_max - global_min + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)

        # Apply heatmap
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Resize so each cell is visible
        heatmap = cv2.resize(
            heatmap,
            (cols * cell_size, rows * cell_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # Draw grid lines
        for i in range(rows + 1):
            y = i * cell_size
            cv2.line(heatmap, (0, y), (cols * cell_size, y), (255, 255, 255), 1)

        for j in range(cols + 1):
            x = j * cell_size
            cv2.line(heatmap, (x, 0), (x, rows * cell_size), (255, 255, 255), 1)

        # Write values inside cells
        for i in range(rows):
            for j in range(cols):
                value = value_table[i, j]

                text = f"{value:.2f}"

                x = j * cell_size + 8
                y = i * cell_size + cell_size // 2

                cv2.putText(
                    heatmap,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Add iteration number at top
        header_height = 45
        header = np.zeros((header_height, heatmap.shape[1], 3), dtype=np.uint8)

        cv2.putText(
            header,
            f"Policy Evaluation - Iteration {iteration}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frame = np.vstack([header, heatmap])

        cv2.imshow("Value Table Updates", frame)

        key = cv2.waitKey(delay)

        # Press q to quit early
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
