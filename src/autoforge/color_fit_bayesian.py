import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import optuna
import multiprocessing

# ===== Helper Functions: Color Conversion =====
def hex_to_rgb(hex_str):
    """Convert a hex color (e.g. "#d9e0e9") to an RGB numpy array (floats 0-255)."""
    hex_str = hex_str.lstrip('#')
    return np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=float)

def rgb_to_hex(rgb):
    """Convert an RGB numpy array (floats 0-255) to a hex color string."""
    return '#' + ''.join(f'{int(round(x)):02X}' for x in rgb)

# ---- sRGB <-> Linear conversion functions ----
def srgb_to_linear(rgb):
    """Convert an sRGB color (0-255) to linear space (0-1)."""
    rgb = rgb / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

def linear_to_srgb(linear):
    """Convert a linear color (0-1) to sRGB (0-255)."""
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * (linear ** (1/2.4)) - 0.055)
    srgb = np.clip(srgb, 0, 1)
    return srgb * 255

def model7(params, t, T):
    A, k = params
    if k < 1e-6:
        return 0.0
    return A * math.log(1 + k * (t / T)) / math.log(1 + k)

def new_model4(params, t, T):
    A, k, B = params
    return A * math.log(1 + k*(t/T)) + B * (t/T)

# ===== Dictionary of Extra Candidate Models =====
extra_models = {
    "New Model 4: Log-Linear": {
        "func": new_model4,
        "ranges": [(0.0, 1.0), (0.0, 100.0), (0.0, 1.0)],
        "param_names": ["A", "k", "B"]
    },
}

# ===== Loss Calculation =====
def total_loss_for_model(model_func, params, df, layer_thickness, blending_mode="sRGB"):
    loss = 0.0
    for idx, row in df.iterrows():
        T = float(row['Transmission Distance']) * 0.1
        bg = hex_to_rgb(row['Background Material'])
        fg = hex_to_rgb(row['Layer Material'])
        for layer in range(1, 17):
            t = layer * layer_thickness
            alpha = model_func(params, t, T)
            # Ensure alpha is within [0, 1]
            if np.isscalar(alpha):
                alpha = max(0.0, min(1.0, alpha))
            else:
                alpha = np.clip(alpha, 0.0, 1.0)
            pred = bg + alpha * (fg - bg)
            meas = hex_to_rgb(row[f"Layer {layer}"])
            loss += np.sum((pred - meas) ** 2)
    loss /= len(df) * 16
    return loss, params

# ===== Optuna Bayesian Optimization using Process-Based Parallelism =====
def optuna_search_model_process(model_func, model_name, param_ranges, param_names, df,
                                layer_thickness, total_trials, blending_mode="sRGB"):
    print(f"Starting Optuna Bayesian Optimization (process-based) for {model_name} ({blending_mode}) ...")

    # Define the objective function for Optuna
    def objective(trial):
        params = []
        for i, name in enumerate(param_names):
            low, high = param_ranges[i]
            params.append(trial.suggest_float(name, low, high))
        loss, _ = total_loss_for_model(model_func, params, df, layer_thickness, blending_mode)
        return loss

    # Create a study with a shared SQLite storage so multiple processes can add trials.
    study = optuna.create_study(
        direction="minimize",
        study_name=model_name,
        storage="sqlite:///{}.db".format(model_name.replace(" ", "_")),
        load_if_exists=True,
    )

    # Determine the number of worker processes and how many trials each will run.
    cpu_count = multiprocessing.cpu_count()
    trials_per_worker = total_trials // cpu_count
    print(f"Launching {cpu_count} processes, each with {trials_per_worker} trials.")

    # Function for each process to run its share of trials.
    def run_worker():
        study.optimize(objective, n_trials=trials_per_worker, n_jobs=1)

    # Launch the worker processes.
    processes = []
    for _ in range(cpu_count):
        p = multiprocessing.Process(target=run_worker)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    best_params_dict = study.best_params
    best_params = [best_params_dict[name] for name in param_names]
    best_loss = study.best_value
    print(f"Best Loss={best_loss:.2f} for {model_name} with params {best_params}")
    return best_params, best_loss, study

# ===== Worker Function for Model Search =====
def run_search_for_model(task):
    model_name, model_data, df, layer_thickness, total_trials, blending_mode = task
    func = model_data["func"]
    param_ranges = model_data["ranges"]
    param_names = model_data["param_names"]
    best_params, best_loss, study = optuna_search_model_process(
        func, model_name, param_ranges, param_names,
        df, layer_thickness, total_trials, blending_mode
    )
    return {
        "model": model_name,
        "blending_mode": blending_mode,
        "best_params": best_params,
        "best_loss": best_loss,
        "param_names": param_names,
        "func": func,
        "study": study
    }

# ===== Main Function =====
def main():
    # Load CSV (adjust path as needed)
    df = pd.read_csv("printed_colors.csv")
    layer_thickness = 0.04  # mm per layer
    blending_mode = "sRGB"  # using sRGB blending

    # --- Phase 1: Process-Based Optuna Bayesian Optimization search for all models ---
    total_trials = 5000  # Total trials per model (will be divided among processes)

    results = []
    for model_name, model_data in extra_models.items():
        result = run_search_for_model((model_name, model_data, df, layer_thickness, total_trials, blending_mode))
        results.append(result)

    results.sort(key=lambda x: x["best_loss"])
    print("\n=== Phase 1: Results (Optuna Process-Based Optimization) ===")
    for res in results:
        param_str = ", ".join(f"{name}={val:.4f}" for name, val in zip(res["param_names"], res["best_params"]))
        print(f"{res['model']} ({res['blending_mode']}): Loss = {res['best_loss']:.2f} with {param_str}")

    # --- Plot Sample Comparison for the Best Model (all CSV rows) ---
    best_result = results[0]
    func = best_result["func"]
    best_params = best_result["best_params"]

    # Optional: plot the model curve for new_model4 for a range of t values.
    y = [new_model4(best_params, t/10, 1) for t in range(1, 11)]
    x = list(range(1, 11))
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.title(f"{best_result['model']} Curve")
    plt.xlabel("t/10")
    plt.ylabel("Model Output")
    plt.show()

    n_csv = len(df)
    # Create a figure with 2 rows per CSV row (measured and predicted) for the best model.
    n_rows = 2 * n_csv
    n_cols = 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8, 3 * n_csv))
    fig.suptitle("Best Model: All CSV Rows\nMeasured (top row) vs. Predicted (bottom row)", fontsize=16)

    for idx in range(n_csv):
        row_data = df.iloc[idx]
        T = float(row_data['Transmission Distance'])
        bg = hex_to_rgb(row_data['Background Material'])
        fg = hex_to_rgb(row_data['Layer Material'])
        measured = [hex_to_rgb(row_data[f"Layer {layer}"]) for layer in range(1, 17)]
        predicted = []
        for layer in range(1, 17):
            t = layer * layer_thickness
            alpha = func(best_params, t, T)
            if np.isscalar(alpha):
                alpha = max(0.0, min(1.0, alpha))
            else:
                alpha = np.clip(alpha, 0.0, 1.0)
            pred = bg + alpha * (fg - bg)
            predicted.append(pred)

        # Plot measured colors
        ax_meas = axes[2*idx]
        for j in range(16):
            rect = Rectangle((j, 0), 1, 1, color=np.clip(measured[j] / 255, 0, 1))
            ax_meas.add_patch(rect)
        ax_meas.set_xlim(0, 16)
        ax_meas.set_ylim(0, 1)
        ax_meas.set_xticks([])
        ax_meas.set_yticks([])
        ax_meas.set_ylabel(f"Row {idx+1}\nMeasured", fontsize=10)

        # Plot predicted colors
        ax_pred = axes[2*idx + 1]
        for j in range(16):
            rect = Rectangle((j, 0), 1, 1, color=np.clip(predicted[j] / 255, 0, 1))
            ax_pred.add_patch(rect)
        ax_pred.set_xlim(0, 16)
        ax_pred.set_ylim(0, 1)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_pred.set_ylabel(f"Row {idx+1}\nPredicted", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()
