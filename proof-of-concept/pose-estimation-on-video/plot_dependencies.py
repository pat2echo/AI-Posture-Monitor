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

def plot_activity_fall(csv_file=None, plot_title=None, show_all_classes=False):
    # Load data
    df = pd.read_csv(csv_file)

    # Replace NaN or None with "Inactivity"
    df['action'] = df['action'].fillna("Inactivity")

    # Adjust action column and priorities based on the `show_all_classes` flag
    if show_all_classes:
        action_priority = {
            "Stand": 30,
            "Stand-Sit": 28,
            "Stand-Lie": 26,
            "Sit": 20,
            "Sit-Stand": 18,
            "Sit-Lie": 16,
            "Lie": 10,
            "Lie-Stand": 8,
            "Lie-Sit": 6,
            "Lie-Fall": 4,
            "Inactivity": 1
        }
    else:
        df['action'] = df['action'].str.split('-').str[0]
        action_priority = {
            "Stand": 30,
            "Sit": 20,
            "Lie": 10,
            "Inactivity": 1
        }

    # Assign y-values based on action priority
    df['y'] = df['action'].map(action_priority)

    # Prepare the step plot data
    times = []
    values = []
    fall_times = []
    for _, row in df.iterrows():
        start, end = row['start_time'], row['end_time']

        times.append(start)  # Start time
        values.append(row['y'])         # Current priority value
        times.append(end)   # End time
        values.append(row['y'])         # Maintain value until the end time
        fall_value = -1 if row['is_fall'] else 0
        fall_times.extend([(t, fall_value) for t in range(start, end + 1)])

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    activity_ax, bar_ax = ax

    # Step plot for actions
    activity_ax.step(times, values, where='post', label='Activity Classes', color='blue', linewidth=2)

    if fall_times:
        fall_df = pd.DataFrame(fall_times, columns=["time", "fall"])
        activity_ax.step(fall_df['time'], fall_df['fall'] * 10, where='post', color='red', lw=2,
                         label='Fall')  # Scale fall line for visibility

    # Customize the activity plot
    y_ticks = list(action_priority.values()) + [-1]
    y_labels = list(action_priority.keys()) + ["Fall"]
    activity_ax.set_yticks(y_ticks)
    activity_ax.set_yticklabels(y_labels)
    activity_ax.set_xlabel('Time', fontsize=14)
    activity_ax.set_title(f'{plot_title}: Activity and Fall Visualization', fontsize=16)
    activity_ax.axhline(-1, color='black', linestyle='--', linewidth=0.5)  # Baseline for falls
    activity_ax.grid(True, linestyle='--', alpha=0.6)
    activity_ax.legend(loc='upper left', fontsize=10)

    # Bar plot for percentage distribution of activity classes
    action_counts = df['action'].value_counts()
    total_count = action_counts.sum()
    action_percentages = (action_counts * 100) / total_count

    colors = plt.cm.tab20(np.linspace(0, 1, len(action_percentages)))
    action_color_map = {action: color for action, color in zip(action_percentages.index, colors)}

    bars = bar_ax.bar(action_percentages.index, action_percentages,
                      color=[action_color_map[action] for action in action_percentages.index],
                      edgecolor='black')

    # Add percentage values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        bar_ax.text(bar.get_x() + bar.get_width() / 2, yval + 1,  # 1 is an offset to position the text above the bar
                    f'{yval:.2f}%', ha='center', va='bottom', fontsize=12)

    bar_ax.set_ylabel('Percentage (%)', fontsize=14)
    bar_ax.set_xlabel('Activity Class', fontsize=14)
    bar_ax.grid(True, linestyle='--', alpha=0.6)

    bar_ax.legend(handles=[plt.Line2D([0], [0], color=color, lw=4, label=action)
                           for action, color in action_color_map.items()],
                  loc='upper right', fontsize=10, title='Activity Classes')

    # Show plots
    plt.tight_layout()
    plt.savefig(f"output/plot/{plot_title}.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory