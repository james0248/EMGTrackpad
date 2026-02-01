import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


def visualize_mouse_log(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: '{csv_file}' file not found.")
        return

    x = df["X"].values
    y = df["Y"].values
    time = df["Timestamp"].values

    time_relative = time - time[0]

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))

    norm = plt.Normalize(time_relative.min(), time_relative.max())
    lc = LineCollection(segments, cmap="plasma", norm=norm)

    lc.set_array(time_relative[:-1])
    lc.set_linewidth(2)

    line = ax.add_collection(lc)

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Time (seconds)")

    ax.set_xlim(x.min() - 50, x.max() + 50)
    ax.set_ylim(y.min() - 50, y.max() + 50)

    # Screen coordinates have (0,0) at top-left, so invert Y axis
    ax.invert_yaxis()

    ax.set_title(f"Mouse Trajectory: {csv_file}")
    ax.set_xlabel("Screen X")
    ax.set_ylabel("Screen Y")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.show()


filename = "mouse_log_50fps.csv"
visualize_mouse_log(filename)
