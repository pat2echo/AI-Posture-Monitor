import numpy as np
import pandas as pd
#import matplotlib
import os

import matplotlib.backends.backend_agg as plt_backend_agg

#matplotlib.use('TkAgg')  # Use TkAgg for terminal
import matplotlib.pyplot as plt

class DynamicPlot:
    def __init__(self):
        # dynamic plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(4, 2), dpi=100, facecolor='white', alpha=0.5)
        self.plot_prediction = []
        self.plot_img_resized = None

    def close(self):
        plt.ioff()
        plt.close(self.fig)
        self.plot_prediction = []
        self.plot_img_resized = None

    def plot(self, prediction=None, frame_output=None, type='pose', offset=0):
        plot_title = 'Static Posture' if type == 'pose' else 'Fall Detection'

        self.plot_prediction.append(prediction)
        if len(self.plot_prediction) >= 600:
            self.plot_prediction.pop(0)
        if len(self.plot_prediction) >= 100:
            if self.plot_img_resized is None or offset % 10 == 0:
                ptimes, pvalues, pactions = self.plot_values_for_dynamic_trend(predictions=self.plot_prediction,
                                                                          type=type, offset=offset)
                self.ax.cla()
                y_ticks = list(pactions.values())
                y_labels = list(pactions.keys())
                self.ax.set_yticks(y_ticks)
                self.ax.set_yticklabels(y_labels)
                self.ax.step(ptimes, pvalues, where='post', label=plot_title, color='green', linewidth=2)

                # Convert plot to image
                canvas = plt_backend_agg.FigureCanvasAgg(self.fig)
                canvas.draw()
                plot_img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)

                # Resize and overlay on frame
                width, height = canvas.get_width_height()
                self.plot_img_resized = plot_img.reshape((height, width, 4))[:, :, :3]

            if self.plot_img_resized is not None:
                x_offset = 5
                #y_offset = frame_output.shape[0] - (5 + self.plot_img_resized.shape[0] )
                y_offset = 5
                frame_output[y_offset:y_offset + self.plot_img_resized.shape[0],
                x_offset:x_offset + self.plot_img_resized.shape[1]] = self.plot_img_resized

        return frame_output

    def plot_values_for_dynamic_trend(self, predictions=None, type='pose', offset=0):
        actions = {
            'pose':{
                "stand": 3,
                "sit": 2,
                "lie": 1,
                "Inactive": 0
            },
            'fall':{
                "Safe":1,
                "Fall":0,
            }
        }
        action_priority = actions[type]


        # Prepare the step plot data
        times = []
        values = []
        start = offset
        prev_pred = 0
        if type == 'pose':
            for pred_key in predictions:
                pred = action_priority[pred_key] if pred_key in action_priority else 0

                if prev_pred != pred:
                    times.append(start)  # Start time
                    values.append(prev_pred)  # Current priority value
                    times.append(start)  # Start time
                    values.append(pred)  # Current priority value
                elif start == offset:
                    times.append(start)  # Start time
                    values.append(pred)  # Current priority value

                prev_pred = pred

                start += 1
        else:
            for pred_key in predictions:
                pred = pred_key
                prev_pred = 0 if pred else 1
                times.append(start)
                values.append(prev_pred)  # Maintain value until the end time
                start += 1

        times.append(start)  # Start time
        values.append(prev_pred)  # Current priority value
        #print(predictions)
        #print(times, values, action_priority)
        return times, values, action_priority


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

def plot_activity_fall(csv_file=None, class_label='action', plot_title=None, show_all_classes=False):
    # Load data
    df = pd.read_csv(csv_file)

    # Replace NaN or None with "Inactivity"
    df['action'] = df[ class_label ].fillna("Inactivity")

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
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), gridspec_kw={'height_ratios': [3, 1]})
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
    create_directory("output/plot")
    plt.savefig(f"output/plot/{plot_title}.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory

def basic_line(csv_file=None):
    df = pd.read_csv(csv_file)

    #fill missing frames
    print(df.shape, df['frame'].max())
    #print( len(np.ones(49)))
    #new_df = pd.DataFrame({'frame': range(0, int(df['frame'].max())) })
    #df = pd.merge(new_df, df, on='frame', how='left').fillna(0)

    # Apply a Hamming window to the 'l' and 'r' columns
    #window = windows.hamming(len(df))
    #df['lshoulder'] = df['lshoulder']**3
    #df['lshoulder'] *= window
    #print(df['frame'][(df['frame'] > 600) & (df['frame'] < 650)])

    # Create the plot
    plt.figure(figsize=(10, 6))
    #plt.plot(df['frame'][(df['frame'] > 600) & (df['frame'] < 650)], (df['lshoulder'][(df['frame'] > 600) & (df['frame'] < 650)]), label='lshoulder')
    #plt.plot(df['frame'][(df['frame'] > 600) & (df['frame'] < 650)], list((np.ones(16)*0.55))+list(np.ones(49-16)*.5), label='initial_window')
    #plt.plot(df['frame'][(df['frame'] > 600) & (df['frame'] < 650)], list(np.ones(8)*.5)+list((np.ones(16)*0.58))+list(np.ones(49-(16+8))*.5), label='next_window')
    plt.plot(df['frame'], (df['lshoulder']), label='lshoulder')
    #plt.plot(df['frame'], np.abs(df['lshoulder']), label='abs_lshoulder')
    #plt.plot(df['frame'], df['lsa'], label='lsa')

    # Customize the plot
    plt.xlabel('Frame')
    plt.ylabel('Y displacement')
    plt.title('Line Plot of Downward Velocity of Left Shoulder Y Co-ordinate\nInitial Results')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_label_vs_prediction(csv_file=None, class_label='label', prediction_label='prediction', smooth_prediction_label='smooth_prediction', plot_title=None, show_all_classes=False):
    # Load data
    df = pd.read_csv(csv_file)
    return plot_label_vs_prediction_data(df=df, class_label=class_label, prediction_label=prediction_label, smooth_prediction_label=smooth_prediction_label,plot_title=plot_title, show_all_classes=show_all_classes)

def plot_label_vs_prediction_data(df=None, class_label='label', prediction_label='prediction', smooth_prediction_label='smooth_prediction', plot_title=None, show_all_classes=False, plot_size=(15, 5)):
    # Load data
    hasLabel = False if df[class_label].isna().all() else True
    multiplier = 0.6 if hasLabel else 1

    df['action'] = df[ class_label ].fillna("Inactivity")
    # Replace NaN or None with "Inactivity"
    df['pred'] = df[ prediction_label ].fillna("Inactivity")
    df['smooth'] = df[ smooth_prediction_label ].fillna("Inactivity")

    # Adjust action column and priorities based on the `show_all_classes` flag
    if show_all_classes:
        action_priority = {
            "stand": 30,
            "stand-Sit": 28,
            "stand-Lie": 26,
            "sit": 20,
            "sit-Stand": 18,
            "sit-Lie": 16,
            "lie": 10,
            "lie-Stand": 8,
            "lie-Sit": 6,
            "lie-Fall": 4,
            "Inactivity": 1
        }
    else:
        df['action1'] = df['action'].str.split('-').str[1]
        df['action'] = df['action'].str.split('-').str[0]
        df['pred'] = df['pred'].str.split('-').str[0]
        df['smooth'] = df['smooth'].str.split('-').str[0]
        action_priority = {
            "stand": 30,
            "sit": 20,
            "lie": 10,
            "Inactivity": 1
        }

    # Assign y-values based on action priority
    df['y1'] = df['action1'].map(action_priority)
    df['y'] = df['action'].map(action_priority)
    df['z'] = df['pred'].map(action_priority)
    df['s'] = df['smooth'].map(action_priority)
    #print( np.unique(df['y'], return_counts=True))

    # Prepare the step plot data
    times = []
    values = []
    values_z = []
    values_zb = []
    marker_times = []
    marker_values = []
    marker_times_s = []
    marker_values_s = []
    start = 0
    for _, row in df.iterrows():
        end = row['file_name']

        failed_prediction = row['z']
        pred_b = row['z']
        if row['z'] == row['y'] or row['z'] == row['y1']:
            pred_b = row['y']
            failed_prediction = 0

        pred = row['s']
        failed_sprediction = row['s']
        if hasLabel and row['s'] == row['y'] or row['s'] == row['y1']:
            pred = row['y']
            failed_sprediction = 0

        times.append(start)  # Start time
        values.append(row['y'])         # Current priority value
        values_z.append(pred*multiplier)         # Current priority value
        values_zb.append(pred_b*0.8)         # Current priority value
        times.append(end)   # End time
        values.append(row['y'])         # Maintain value until the end time
        values_z.append(pred*multiplier)         # Current priority value
        values_zb.append(pred*0.8)         # Current priority value

        # Record marker positions for false predictions
        if failed_prediction != 0 and row['z'] != row['s']:
            marker_times.append((start + end) / 2)  # Use the midpoint of start and end
            marker_values.append(row['z'])  # Corresponding prediction value

        # Record marker positions for false predictions
        if failed_sprediction != 0:
            marker_times_s.append((start + end) / 2)  # Use the midpoint of start and end
            marker_values_s.append(row['s']*.9)  # Corresponding prediction value

        start = end
        if start > 1500:
            pass

    # Plotting
    fig, ax = plt.subplots(figsize=plot_size)
    activity_ax = ax

    # Step plot for actions
    #activity_ax.step(times, values_zb, where='post', label='Prediction Before Smoothing', color='red', linewidth=1, linestyle='--')
    if hasLabel:
        activity_ax.step(times, values, where='post', label='Label', color='blue', linewidth=2)
        activity_ax.scatter(marker_times, marker_values, color='red', label='False Prediction Eliminated By Smoothing', zorder=5)
    activity_ax.step(times, values_z, where='post', label='Prediction', color='green', linewidth=2)
    #activity_ax.scatter(marker_times_s, marker_values_s, color='yellow', label='False S.Prediction', zorder=5)

    # Customize the activity plot
    y_ticks = list(action_priority.values())
    y_labels = list(action_priority.keys())
    activity_ax.set_yticks(y_ticks)
    activity_ax.set_yticklabels(y_labels)
    activity_ax.set_xlabel('Time', fontsize=14)
    activity_ax.set_title(f'{plot_title}: Activity Prediction', fontsize=16)
    activity_ax.axhline(-1, color='black', linestyle='--', linewidth=0.5)  # Baseline for falls
    activity_ax.grid(True, linestyle='--', alpha=0.6)
    activity_ax.legend(loc='lower left', fontsize=10)

    # Show plots
    plt.tight_layout()
    create_directory("output/plot")
    plt.savefig(f"output/plot/{plot_title}.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory

def plot_transitions(csv_file=None, class_label='label', prediction_label='smooth_prediction', plot_title=None):
    # Load data
    df = pd.read_csv(csv_file)
    # Replace NaN or None with "Inactivity"
    df['action'] = df[ class_label ].fillna("Inactivity")
    df['pred'] = df[ prediction_label ].fillna("Inactivity")

    states = {
        "State Change": 3,
        "Stationary": 2,
        "Present": 1,
        "Absent": 0
    }

    start = 0
    times = []
    values = []
    pred_values = []
    tran_values = []
    previous_activity = 'Absent'
    prev_state = None

    pred_previous_activity = 'Absent'
    pred_prev_state = None
    annotations = []

    marker_times = []
    marker_values = []
    vmarker_times = []
    vmarker_values = []

    p_pred = 0
    f_pred = 0

    tp_pred = 0
    tf_pred = 0

    result = {'label':[], 'predicted_label':[]}

    for _, row in df.iterrows():
        end = row['file_name']

        current_state = row['action']
        transition, current_activity = get_transition(prev_state=prev_state, current_state=current_state, previous_activity=previous_activity)

        t_value = states[current_activity]
        t_value = 1 if transition else 0
        result['label'].append(t_value)
        times.append(start)  # Start time
        values.append(t_value)
        times.append(end)  # End time
        values.append(t_value)  # Maintain value until the end time

        prev_state = current_state
        previous_activity = current_activity

        current_state = row['pred']
        transition, current_activity = get_transition(prev_state=pred_prev_state, current_state=current_state,
                                                      previous_activity=pred_previous_activity)

        pred_t_value = states[current_activity]
        pred_t_value = 1 if transition else 0
        result['predicted_label'].append(pred_t_value)
        pred_values.append(pred_t_value*-1)
        pred_values.append(pred_t_value*-1)  # Maintain value until the end time

        # tran_t_value = 1 if row['is_transition'] else 0
        # tran_values.append(tran_t_value*-0.5)
        # tran_values.append(tran_t_value*-0.5)
        #
        # if t_value == tran_t_value:
        #     tp_pred += 1
        # else:
        #     tf_pred += 1

        if pred_t_value != t_value:
            marker_times.append((start + end) / 2)  # Use the midpoint of start and end
            marker_values.append(0.5)  # Corresponding prediction value
            f_pred += 1
        else:
            p_pred += 1

        pred_prev_state = current_state
        pred_previous_activity = current_activity

        # if t_value and transition and previous_activity != current_activity:
        #     annotations.append((start, t_value, current_state))

        start = end

        ar = np.abs(row[['v_lshoulder','v_rshoulder','v_lhip','v_rhip']].values)
        if np.all(ar) != 0:
            #print('found')
            vmarker_times.append((start + end) / 2)  # Use the midpoint of start and end
            vmarker_values.append( np.max(np.abs(ar)) )  # Corresponding prediction value

        if start > 1600:
            break

    print(df["action"].value_counts())
    print('accuracy', p_pred/(p_pred+f_pred))
    #print('accuracy t', tp_pred/(tp_pred+tf_pred))
    #print(np.unique( np.array(values), return_counts=True))

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    activity_ax = ax

    # Step plot for actions
    activity_ax.step(times, values, where='post', label='Label', color='blue', linewidth=2)
    activity_ax.step(times, pred_values, where='post', label='Predicted State Change', color='green', linewidth=2)
    #activity_ax.step(times, tran_values, where='post', label='Transition', color='yellow', linewidth=1)

    #activity_ax.scatter(marker_times, marker_values, color='red', label='False Prediction of State Change', zorder=5)
    #activity_ax.scatter(vmarker_times, vmarker_values, color='purple', label='Velocity', zorder=5)
    #activity_ax.plot(df['file_name'][:1000], (df['bbox_aspect_ratio_of_pose'][:1000]), label='bbox_aspect_ratio_of_pose', color='orange')
    #activity_ax.plot(df['file_name'][:1000], (df['bbox_aspect_ratio'][:1000]), label='bbox_aspect_ratio', color='orange')
    # j = 0
    # y_offset = -0.6
    # for time, value, label in annotations:
    #     if j > 0 and j % 3 == 0:
    #         y_offset = -0.6
    #         j = 0
    #     j = j + 1
    #     y_offset += 0.2
    #
    #     activity_ax.annotate(
    #         label,  # The text to display
    #         xy=(time, value),  # Position of the annotation
    #         xytext=(time - 100, value + y_offset),  # Slightly offset to avoid overlap
    #         textcoords='data',
    #         fontsize=8,
    #         ha='left',  # Horizontally center the text
    #         arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)  # Optional arrow
    #     )
    # Customize the activity plot
    y_ticks = list(states.values())
    y_labels = list(states.keys())
    #activity_ax.set_yticks(y_ticks)
    #activity_ax.set_yticklabels(y_labels)

    activity_ax.set_xlabel('Time', fontsize=14)
    activity_ax.set_title(f'{plot_title}: Recognize Transition', fontsize=16)
    activity_ax.axhline(-1, color='black', linestyle='--', linewidth=0.5)  # Baseline for falls
    activity_ax.grid(True, linestyle='--', alpha=0.6)
    activity_ax.legend(loc='lower left', fontsize=10)

    # Show plots
    plt.tight_layout()
    create_directory("output/plot")
    plt.savefig(f"output/plot/{plot_title}.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory

    return pd.DataFrame(result)

def get_transition(prev_state, current_state, previous_activity):
    transition = False
    if '-' in current_state:
        # test for transition label
        transition = True
    elif prev_state is not None and current_state != prev_state and ('-' not in prev_state or '-' not in current_state):
        # test for transition label
        transition = True

    current_activity = previous_activity
    if transition:
        if prev_state == 'Inactivity':
            current_activity = 'Present'
        elif current_state == 'Inactivity':
            current_activity = 'Absent'
        else:
            # print(start, prev_state, current_state)
            if '-' in prev_state and '-' not in current_state:
                current_activity = 'Stationary'
            else:
                current_activity = 'State Change'

    return transition, current_activity

def plot_fall(csv_file=None, class_label='fall_watch', prediction_label='fall', plot_title=None):
    # Load data
    df = pd.read_csv(csv_file)
    return plot_fall_data(df, class_label=class_label, prediction_label=prediction_label, plot_title=plot_title)

def plot_fall_data(df=None, class_label='fall_watch', prediction_label='fall', plot_title=None, plot_size=(15, 5)):
    hasLabel = False if df[class_label].isna().all() else True
    multiplier = -1 if hasLabel else 1

    # Replace NaN or None with "Inactivity"
    df['action'] = df[ class_label ].fillna("Inactivity")
    df['pred'] = df[ prediction_label ].fillna("Inactivity")

    start = 0
    times = []
    values = []
    pred_values = []
    marker_times = []
    marker_values = []

    p_pred = 0
    f_pred = 0
    is_start_fall = False
    is_start_pred_fall = False

    is_start_non_fall = False
    is_start_pred_non_fall = False

    result = {'label':[], 'predicted_label':[]}
    result2 = {'label':[], 'predicted_label':[]}

    prev_label = None
    prev_pred = None

    for _, row in df.iterrows():
        end = row['file_name']

        label = row['action']
        t_value = 1 if label else 0
        result['label'].append(t_value)

        times.append(start)  # Start time
        times.append(end)  # End time
        if hasLabel:
            values.append(t_value)
            values.append(t_value)  # Maintain value until the end time

        pred = row['pred']
        pred_t_value = 1 if pred else 0
        result['predicted_label'].append(pred_t_value)
        pred_values.append(pred_t_value*multiplier)
        pred_values.append(pred_t_value*multiplier)  # Maintain value until the end time

        if label != prev_label:
            if prev_pred is not None and len(result2['label']) != len(result2['predicted_label']):
                result2['predicted_label'].append(prev_pred)
            result2['label'].append(label)

        if pred != prev_pred and len(result2['label']) != len(result2['predicted_label']):
            result2['predicted_label'].append(pred)
        prev_label = label
        prev_pred = pred

        start = end


    if len(result2['label']) != len(result2['predicted_label']):
        result2['predicted_label'].append(pred)
    #print(result2)
    #print(df["action"].value_counts())
    #print('accuracy t', tp_pred/(tp_pred+tf_pred))
    #print(np.unique( np.array(values), return_counts=True))

    # Plotting
    fig, ax = plt.subplots(figsize=plot_size)
    activity_ax = ax

    # Step plot for actions
    if hasLabel:
        activity_ax.step(times, values, where='post', label='Label', color='blue', linewidth=2)
    activity_ax.step(times, pred_values, where='post', label='Predicted Fall', color='green', linewidth=2)

    #activity_ax.scatter(marker_times, marker_values, color='red', label='False Prediction of State Change', zorder=5)
    #activity_ax.scatter(vmarker_times, vmarker_values, color='purple', label='Velocity', zorder=5)
    action_priority = {
        "fall": 1,
        "safe": 0
    }
    y_ticks = list(action_priority.values())
    y_labels = list(action_priority.keys())
    activity_ax.set_yticks(y_ticks)
    activity_ax.set_yticklabels(y_labels)

    # Customize the activity plot
    activity_ax.set_xlabel('Time', fontsize=14)
    activity_ax.set_title(f'{plot_title}: Fall Detection', fontsize=16)
    activity_ax.axhline(-1, color='black', linestyle='--', linewidth=0.5)  # Baseline for falls
    activity_ax.grid(True, linestyle='--', alpha=0.6)
    activity_ax.legend(loc='lower left', fontsize=10)

    # Show plots
    plt.tight_layout()
    create_directory("output/plot")
    plt.savefig(f"output/plot/{plot_title}.png", dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory

    return pd.DataFrame(result), pd.DataFrame(result2)

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
