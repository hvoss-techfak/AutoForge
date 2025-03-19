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
    #key is formatted as lr=float, we need to extract the float value and round it to 4 decimal values
    key = round(float(key.split("=")[1]), 5)
    stats.append((key, mean_val, std_val, arr))

# Sort the list of tuples based on the mean (ascending order)
#stats.sort(key=lambda x: x[1])

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

# First, compute all significant pairwise comparisons
significant_pairs = []  # will hold tuples of (p_value, i, j)
for i in range(n_keys):
    for j in range(i + 1, n_keys):
        group1 = data_sorted[sorted_keys[i]]
        group2 = data_sorted[sorted_keys[j]]
        stat, p_value = ttest_ind(group1, group2)
        if p_value < 0.05:
            significant_pairs.append((p_value, i, j))

# Sort pairs by p-value (most significant first)
significant_pairs.sort(key=lambda x: x[0])

# To ensure each bar is annotated only once, track used indices.
used_bars = set()
base_margin = max(means) * 0.05  # margin above the error bars

significant_pairs = []
for p_value, i, j in significant_pairs:
    # If either bar is already annotated, skip this pair.
    if i in used_bars or j in used_bars:
        continue

    # Compute the base y-level from the higher error bar of the two.
    base_y = max(means[i] + stds[i], means[j] + stds[j])
    line_y = base_y + base_margin

    # Mark these bars as used.
    used_bars.update([i, j])

    # Draw horizontal significance line between the two bars.
    fig.add_shape(
        type="line",
        x0=x_positions[i],
        x1=x_positions[j],
        y0=line_y,
        y1=line_y,
        line=dict(color="black", width=1),
        xref="x",
        yref="y",
    )
    # Place the significance annotation (stars) above the line.
    fig.add_annotation(
        x=(x_positions[i] + x_positions[j]) / 2,
        y=line_y + base_margin * 0.2,
        text=significance_label(p_value),
        showarrow=False,
        font=dict(size=12),
    )


fig.update_layout(
    autosize=False,
    width=1920,
    height=800,
)
# save image
fig.write_image("out.png")
