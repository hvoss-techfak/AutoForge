import json
import numpy as np
import plotly.graph_objects as go
from scipy.stats import ttest_ind

# Load the JSON file with arbitrary keys
with open("out_dict.json", "r") as f:
    data = json.load(f)

# Calculate means, standard deviations, and store the arrays for each condition.
# We store as a tuple: (key, mean, std, data array)
stats = []
for key, values in data.items():
    arr = np.array(values)
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    stats.append((key, mean_val, std_val, arr))

# Sort the list of tuples based on the mean (ascending order)
stats.sort(key=lambda x: x[1])

# Extract sorted lists of keys, means, stds, and store the sorted arrays in a dictionary
sorted_keys = [item[0] for item in stats]
means = [item[1] for item in stats]
stds = [item[2] for item in stats]
data_sorted = {item[0]: item[3] for item in stats}
n_keys = len(sorted_keys)

# Create x-positions for the bars (one per key)
x_positions = list(range(n_keys))

# Create a Plotly bar chart with error bars
fig = go.Figure()
for i, key in enumerate(sorted_keys):
    fig.add_trace(
        go.Bar(
            name=key,
            x=[x_positions[i]],
            y=[means[i]],
            error_y=dict(type="data", array=[stds[i]], visible=True),
        )
    )

fig.update_layout(
    title="Comparison of Conditions with Mean ± STD (Sorted by Mean)",
    xaxis=dict(
        tickmode="array",
        tickvals=x_positions,
        ticktext=sorted_keys,
        tickangle=45,  # Rotate labels for better readability
    ),
    yaxis_title="Value",
    barmode="group",
)


# Function to generate significance stars based on the p-value.
def significance_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# To avoid overlapping significance annotations when bars share a common index,
# keep track of the highest annotation drawn for each bar.
annotation_levels = {i: 0 for i in range(n_keys)}
# Base margin added above the highest error bar to start the significance line.
base_margin = max(means) * 0.05  # Adjust as needed

# Loop over adjacent pairs in the sorted order and perform t-tests.
for i in range(n_keys - 1):
    group1 = data_sorted[sorted_keys[i]]
    group2 = data_sorted[sorted_keys[i + 1]]
    stat, p_value = ttest_ind(group1, group2)

    # Only annotate if the difference is significant (p < 0.05)
    if p_value < 0.05:
        # Determine the base y-level from the taller of the two error bars.
        base_y = max(means[i] + stds[i], means[i + 1] + stds[i + 1])
        # Ensure new annotations are drawn above any previous ones.
        current_level = max(annotation_levels[i], annotation_levels[i + 1])
        line_y = base_y + base_margin + current_level

        # Update annotation levels for these two bars.
        annotation_levels[i] = line_y
        annotation_levels[i + 1] = line_y

        # Add a horizontal line between the two bars.
        fig.add_shape(
            type="line",
            x0=x_positions[i],
            x1=x_positions[i + 1],
            y0=line_y,
            y1=line_y,
            line=dict(color="black", width=1),
            xref="x",
            yref="y",
        )
        # Add the significance stars annotation above the line.
        fig.add_annotation(
            x=(x_positions[i] + x_positions[i + 1]) / 2,
            y=line_y + base_margin * 0.2,
            text=significance_label(p_value),
            showarrow=False,
            font=dict(size=12),
        )
# save image
fig.write_image("out.png")
