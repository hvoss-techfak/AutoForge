import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import optuna
import concurrent.futures
import multiprocessing
from tqdm import tqdm

from autoforge.Misc.color_fit_all import model_weighted_average


# ===== Helper Functions: Color Conversion =====
def hex_to_rgb(hex_str):
    """Convert a hex color (e.g. "#d9e0e9") to an RGB numpy array (floats 0-255)."""
    hex_str = hex_str.lstrip("#")
    return np.array([int(hex_str[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def rgb_to_hex(rgb):
    """Convert an RGB numpy array (floats 0-255) to a hex color string."""
    return "#" + "".join(f"{int(round(x)):02X}" for x in rgb)


# ---- sRGB <-> Linear conversion functions ----
def srgb_to_linear(rgb):
    """Convert an sRGB color (0-255) to linear space (0-1)."""
    rgb = rgb / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear


def linear_to_srgb(linear):
    """Convert a linear color (0-1) to sRGB (0-255)."""
    srgb = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * (linear ** (1 / 2.4)) - 0.055
    )
    srgb = np.clip(srgb, 0, 1)
    return srgb * 255


# ===== Loss Calculation =====
def total_loss_for_model(model_func, params, df, layer_thickness, blending_mode="sRGB"):
    """
    Compute the combined loss and also per-layer loss.
    For each layer, the absolute error is computed between the predicted color
    and the measured color.
    Returns:
        total_loss: overall loss averaged over all rows and layers
        per_layer_loss: a NumPy array (length=num_layers) of average losses per layer
        params: the used parameters (for reference)
    """
    num_layers = 16
    total_loss = 0.0
    per_layer_loss = np.zeros(num_layers)
    num_rows = len(df)

    for idx, row in df.iterrows():
        T = float(row["Transmission Distance"]) * 0.1
        bg = hex_to_rgb(row["Background Material"])
        fg = hex_to_rgb(row["Layer Material"])
        for layer in range(1, num_layers + 1):
            try:
                t = layer * layer_thickness
                manual_offsets = np.array(params[:3])
                model_params = params[3:]
                alpha_model = model_func(model_params, t, T)
                if np.isscalar(alpha_model):
                    alpha_model = np.array([alpha_model, alpha_model, alpha_model])
                alpha = manual_offsets + alpha_model
                alpha = np.clip(alpha, 0.0, 1.0)
                pred = bg + alpha * (fg - bg)
                meas = hex_to_rgb(row[f"Layer {layer}"])
                error = np.sum(np.abs(pred - meas))
            except Exception:
                error = 1e10
            per_layer_loss[layer - 1] += error
            total_loss += error
    total_loss /= num_rows * num_layers
    per_layer_loss /= num_rows
    return total_loss, per_layer_loss, params


# ===== Worker Function: Run a Single Batch of Trials in an Independent Study =====
def run_worker_for_model_batch(
    model_func,
    param_ranges,
    param_names,
    df,
    layer_thickness,
    batch_trials,
    blending_mode,
    initial_trials,
):
    """
    Run an independent study for a fixed number of trials (a batch).
    If initial_trials (a list of dicts with 'params' and 'value') are provided,
    add them as seed trials.
    """
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )

    # Add each of the initial trials.
    if initial_trials is not None:
        for trial_data in initial_trials:
            trial = optuna.trial.create_trial(
                params=trial_data["params"],
                distributions={
                    name: optuna.distributions.UniformDistribution(*param_ranges[i])
                    for i, name in enumerate(param_names)
                },
                value=trial_data["value"],
            )
            study.add_trial(trial)

    def objective(trial):
        params = []
        for i, name in enumerate(param_names):
            low, high = param_ranges[i]
            params.append(trial.suggest_float(name, low, high))
        loss, _, _ = total_loss_for_model(
            model_func, params, df, layer_thickness, blending_mode
        )
        return loss

    study.optimize(objective, n_trials=batch_trials, n_jobs=1)

    # Serialize all completed trials from this study.
    trial_records = [
        {"params": t.params, "value": t.value}
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    return trial_records


# ===== Batch Runner: Execute One Batch Round Across All Workers =====
def run_batch_for_model(
    model_func,
    param_ranges,
    param_names,
    df,
    layer_thickness,
    batch_trials,
    blending_mode,
    initial_trials,
):
    cpu_count = multiprocessing.cpu_count()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [
            executor.submit(
                run_worker_for_model_batch,
                model_func,
                param_ranges,
                param_names,
                df,
                layer_thickness,
                batch_trials,
                blending_mode,
                initial_trials,
            )
            for _ in range(cpu_count)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as e:
                print("Worker failed:", e)
    return results


# ===== Merge Batch Results =====
def merge_batch_results(all_trial_records):
    """
    Given a list of trial records (each a dict with keys 'params' and 'value'),
    return the best trial (lowest loss) and also the full list.
    """
    best_trial = min(all_trial_records, key=lambda x: x["value"])
    return best_trial, all_trial_records


# ===== Helper: Deduplicate Trials =====
def deduplicate_trials(trials):
    """
    Remove duplicate trial records based on their parameters.
    Trials with identical parameter sets (ignoring order) are merged,
    keeping the one with the lower loss.
    """
    unique = {}
    for trial in trials:
        # Create a key from sorted (param, value) pairs
        key = tuple(sorted(trial["params"].items()))
        if key in unique:
            if trial["value"] < unique[key]["value"]:
                unique[key] = trial
        else:
            unique[key] = trial
    return list(unique.values())


# ===== Merge Batch Results =====
def merge_batch_results(all_trial_records):
    """
    Given a list of trial records (each a dict with keys 'params' and 'value'),
    return the best trial (lowest loss) and also the full list.
    """
    best_trial = min(all_trial_records, key=lambda x: x["value"])
    return best_trial, all_trial_records


# ===== Search Function with Periodic Merging (All Trials Seeded and Duplicates Removed) =====
def run_search_for_model_with_batches(
    model_name,
    model_data,
    df,
    layer_thickness,
    total_trials,
    batch_trials,
    blending_mode,
):
    """
    Run the optimization in rounds (batches). In each round, every CPU core runs an independent
    study for `batch_trials` trials. At the end of the round, all trial records are merged and duplicates
    are removed. The resulting set is used to seed the next round.
    """
    model_func = model_data["func"]
    # Make copies of the parameter names and ranges so we can modify them.
    param_ranges = model_data["ranges"].copy()
    param_names = model_data["param_names"].copy()

    # Insert three manual offset parameters (for R, G, B) at the beginning.
    for label in ["manual offset B", "manual offset G", "manual offset R"]:
        param_names.insert(0, label)
        param_ranges.insert(0, (-1.0, 1.0))

    cpu_count = multiprocessing.cpu_count()
    total_batches = total_trials // (batch_trials * cpu_count)

    # This list will accumulate all trial records to seed new batches.
    cumulative_trials = []

    best_trial = None
    for batch_idx in range(total_batches):
        print(
            f"Starting batch {batch_idx + 1}/{total_batches} with {len(cumulative_trials)} seed trials."
        )
        batch_trial_records = run_batch_for_model(
            model_func,
            param_ranges,
            param_names,
            df,
            layer_thickness,
            batch_trials,
            blending_mode,
            cumulative_trials if cumulative_trials else None,
        )
        # Merge with existing cumulative trials and deduplicate.
        if cumulative_trials:
            cumulative_trials.extend(batch_trial_records)
            cumulative_trials = deduplicate_trials(cumulative_trials)
        else:
            cumulative_trials = batch_trial_records

        best_trial, _ = merge_batch_results(cumulative_trials)
        print(f"Batch {batch_idx + 1}: Best loss = {best_trial['value']:.2f}")

    print(
        f"\nFinal best result for {model_name}: Loss = {best_trial['value']:.2f} with params {best_trial['params']}"
    )
    return {
        "model": model_name,
        "blending_mode": blending_mode,
        "best_params": best_trial["params"],
        "best_loss": best_trial["value"],
        "param_names": param_names,
        "func": model_func,
    }


# ===== Main Function =====
def main():
    df = pd.read_csv("printed_colors.csv")
    layer_thickness = 0.04  # mm per layer
    blending_mode = "sRGB"  # using sRGB blending
    total_trials = 30000  # total trials per model
    batch_trials = 200  # trials per worker per batch round

    extra_models = {
        "Model 22: Weighted Average": {
            "func": model_weighted_average,
            "ranges": [(0.0, 1.0), (0.0, 1.0)],
            "param_names": ["p0", "p1"],
        },
    }

    results = []
    tbar = tqdm(total=len(extra_models))
    for model_name, model_data in extra_models.items():
        result = run_search_for_model_with_batches(
            model_name,
            model_data,
            df,
            layer_thickness,
            total_trials,
            batch_trials,
            blending_mode,
        )
        results.append(result)
        tbar.update(1)
    tbar.close()

    results.sort(key=lambda x: x["best_loss"])
    print(
        "\n=== Phase 1: Results (Batched Independent Optimization with Full Trial Merging) ==="
    )

    # --- Plot Sample Comparison for the Best Model (all CSV rows) ---
    best_result = results[0]
    func = best_result["func"]
    best_params = best_result["best_params"]

    total_loss, per_layer_loss, _ = total_loss_for_model(
        func, best_params, df, layer_thickness, blending_mode
    )
    layers = np.arange(1, 17)
    plt.figure(figsize=(8, 4))
    plt.bar(layers, per_layer_loss, color="skyblue")
    plt.xlabel("Layer Number")
    plt.ylabel("Average Absolute Error")
    plt.title("Average Error per Layer")
    plt.xticks(layers)
    plt.tight_layout()
    plt.show()

    # --- Plot Measured vs. Predicted for Each CSV Row ---
    n_csv = len(df)
    num_layers = 16
    fig, axes = plt.subplots(nrows=2 * n_csv, ncols=1, figsize=(8, 3 * n_csv))
    fig.suptitle(
        "Best Model: All CSV Rows\nMeasured (top row) vs. Predicted (bottom row)",
        fontsize=16,
    )
    for idx in range(n_csv):
        row_data = df.iloc[idx]
        T = float(row_data["Transmission Distance"])
        bg = hex_to_rgb(row_data["Background Material"])
        fg = hex_to_rgb(row_data["Layer Material"])
        measured = [
            hex_to_rgb(row_data[f"Layer {layer}"]) for layer in range(1, num_layers + 1)
        ]
        predicted = []
        for layer in range(1, num_layers + 1):
            t = layer * layer_thickness
            alpha = func(best_params, t, T)
            if np.isscalar(alpha):
                alpha = max(0.0, min(1.0, alpha))
            else:
                alpha = np.clip(alpha, 0.0, 1.0)
            pred = bg + alpha * (fg - bg)
            predicted.append(pred)
        ax_meas = axes[2 * idx]
        for j in range(num_layers):
            rect = Rectangle((j, 0), 1, 1, color=np.clip(measured[j] / 255, 0, 1))
            ax_meas.add_patch(rect)
        ax_meas.set_xlim(0, num_layers)
        ax_meas.set_ylim(0, 1)
        ax_meas.set_xticks([])
        ax_meas.set_yticks([])
        ax_meas.set_ylabel(f"Row {idx + 1}\nMeasured", fontsize=10)
        ax_pred = axes[2 * idx + 1]
        for j in range(num_layers):
            rect = Rectangle((j, 0), 1, 1, color=np.clip(predicted[j] / 255, 0, 1))
            ax_pred.add_patch(rect)
        ax_pred.set_xlim(0, num_layers)
        ax_pred.set_ylim(0, 1)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.set_ylabel(f"Row {idx + 1}\nPredicted", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
