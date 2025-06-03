import json
import numpy as np
import plotly.graph_objects as go
from scipy.stats import ttest_ind

# Load the JSON file with arbitrary keys
with open("output_grid/out_dict_20250602_221648.json", "r") as f:
    data = json.load(f)

# Calculate means, standard deviations, and store the arrays for each condition.
# We store as a tuple: (key, mean, std, data array)
stats = {}
for run in data:
    param_value = run["param_value"]
    loss = run["loss"]
    stats.setdefault(param_value, []).append(loss)

# Sort the list of tuples based on the mean (ascending order)
#stats.sort(key=lambda x: x[1])

# Extract sorted lists of keys, means, stds, and store the sorted arrays in a dictionary
sorted_keys = sorted(list(stats.keys()))
means = [np.mean(stats[key]) for key in sorted_keys]
stds = [np.std(stats[key]) for key in sorted_keys]
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


fig.update_layout(
    autosize=False,
    width=1920,
    height=800,
)
# save image
fig.write_image("out.png")
