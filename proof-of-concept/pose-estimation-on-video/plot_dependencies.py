import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Reference: from Adrian Clark Computer Vision Lab 1 - CSEE - University of Essex
def plot_histogram (x, y, title, xlabel, colours=["blue", "green", "red"]):
    """
    Plot a histogram (bar-chart) of the data in `x` and `y` using
    Matplotlib.  The `y` array can be either a single-dimensional one
    (for the histogram of a monochrome image) or two-dimensional for a
    colour image, in which case the first dimension selects the colour
    band and the second the value in that colour band.  `title` is the
    title of the plot, shown along its top edge.

    Args:
        x (array): numpy array containing the values to plot along the
                   abscissa (x) axis
        y (array): numpy array of the same length as `x` containing the
                   values to plot along the ordinate (y) axis
        title (str): title to put along the top edge of the plot
        xlabel (str): title to put along the x-axis
        colours (list of strings): the colours to use when there is more
                                   than one plot on the axes
                                   (default: blue, green, red)
    """

    # Set up the plot.
    plt.figure ()
    plt.grid ()
    plt.xlim ([0, x[-1]])
    plt.xlabel ( xlabel )
    plt.ylabel ("frequency")
    plt.title (title)

    # Plot the data.
    if len (y.shape) == 1:
        plt.bar (x, y, color="cyan")
    else:
        nc, np = y.shape
        for c in range (0, nc):
            plt.bar (x, y[c], color=colours[c])

    # Show the result.
    plt.show()

def plot_activity_fall(csv_file=None, plot_title=None):
    df = pd.read_csv(csv_file)

    # Assign unique colors and heights for each action
    actions = df['action'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(actions)))
    action_color_map = {action: color for action, color in zip(actions, colors)}
    action_height_map = {action: i + 1 for i, action in enumerate(actions)}

    # Plot the actions as vertical bars
    fig, ax = plt.subplots(figsize=(15, 6))
    for _, row in df.iterrows():
        start, end = row['start_time'], row['end_time']
        action = row['action']
        is_fall = row['is_fall']
        color = action_color_map[action]
        height = action_height_map[action]
        if action != "None":
            ax.barh(1, end - start, left=start, color=color, edgecolor='black', height=height * 0.1, label=action)

    # Highlight falls with red lines
    for _, row in df[df['is_fall']].iterrows():
        start, end = row['start_time'], row['end_time']
        ax.plot([start, end], [1, 1], color='red', lw=2, label='Fall')

    # Customizing the plot
    ax.set_yticks([])
    ax.set_xlabel('Time', fontsize=14)
    ax.set_title(f'{plot_title}: Manual Label Visualization', fontsize=16)
    ax.legend(handles=[plt.Line2D([0], [0], color=color, lw=4, label=action)
                       for action, color in action_color_map.items() if action != "None"] +
                      [plt.Line2D([0], [0], color='red', lw=4, label='Fall')],
              loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    plt.show()