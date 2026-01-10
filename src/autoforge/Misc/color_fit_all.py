import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import TorchOptimizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

"""
Opacity SINDy Estimator (scalar opacity per layer)
-------------------------------------------------
- Extracts a single scalar opacity o_l per layer via RGB least squares using
  the Porter–Duff over compositing model.
- Fits a PySINDy continuous-time model with control inputs u = [TD_I, layer_height].
- Simulates opacity sequences and composites predicted colors; plots measured vs predicted.

Expected CSV columns:
- "Transmission Distance" (TD_I)
- "Background Material" (hex color)
- "Layer Material" (hex color)
- "Layer 1" ... "Layer N" (hex color per layer composite measurement)
"""

# ---------------------
# Utilities
# ---------------------


def hex_to_rgb(hex_str: str) -> np.ndarray:
    s = hex_str.lstrip("#")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return np.array([r, g, b], dtype=np.float32)


# sRGB (0-1) <-> linear RGB (0-1)


def _srgb01_to_linear01(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    return np.where(
        c <= 0.04045,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4,
    )


def _linear01_to_srgb01(c_lin: np.ndarray) -> np.ndarray:
    c_lin = np.asarray(c_lin, dtype=np.float32)
    return np.where(
        c_lin <= 0.0031308,
        12.92 * c_lin,
        1.055 * (np.clip(c_lin, 0.0, None) ** (1.0 / 2.4)) - 0.055,
    )


def hex_to_linear_rgb01(hex_str: str) -> np.ndarray:
    rgb255 = hex_to_rgb(hex_str)
    srgb01 = np.clip(rgb255 / 255.0, 0.0, 1.0)
    return np.clip(_srgb01_to_linear01(srgb01), 0.0, 1.0)


def linear_rgb01_to_srgb255(rgb_lin01: np.ndarray) -> np.ndarray:
    srgb01 = _linear01_to_srgb01(np.clip(rgb_lin01, 0.0, 1.0))
    return np.clip(srgb01 * 255.0, 0.0, 255.0)


# Forward compositing from scalar opacity series (in linear RGB domain)
# Returns list of per-layer composite sRGB (float32 in [0,255]) and final color.


def composite_from_opacity_series(
    bg_lin01: np.ndarray,
    fg_lin01: np.ndarray,
    o_series: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    comp_lin = np.zeros(3, dtype=np.float32)
    remaining = 1.0
    per_layer_colors_srgb255: List[np.ndarray] = []
    for o in o_series:
        o = float(np.clip(o, 0.0, 1.0))
        comp_lin = comp_lin + (remaining * o) * fg_lin01
        remaining = remaining * (1.0 - o)
        color_lin = comp_lin + remaining * bg_lin01
        color_srgb255 = linear_rgb01_to_srgb255(color_lin)
        per_layer_colors_srgb255.append(color_srgb255)
    final_color = (
        per_layer_colors_srgb255[-1]
        if per_layer_colors_srgb255
        else linear_rgb01_to_srgb255(bg_lin01)
    )
    return per_layer_colors_srgb255, final_color


def extract_scalar_opacity_series(
    df: pd.DataFrame,
    num_layers: int,
    layer_height: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Invert compositing to recover a scalar opacity per layer using RGB least squares.
    Operates in linear RGB domain for physical correctness.

    Returns:
      - o_list: list of (num_layers,) arrays per sample
      - t_list: list of (num_layers,) time arrays per sample (t = idx * layer_height)
    """
    o_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []

    eps = 1e-8
    for _, row in df.iterrows():
        bg_lin = hex_to_linear_rgb01(str(row["Background Material"]))
        fg_lin = hex_to_linear_rgb01(str(row["Layer Material"]))

        comp_lin = np.zeros(3, dtype=np.float32)
        remaining = 1.0

        o_series = np.zeros(num_layers, dtype=np.float32)
        for layer in range(1, num_layers + 1):
            meas_lin = hex_to_linear_rgb01(str(row[f"Layer {layer}"]))
            # d = meas - (comp + remaining*bg) = remaining * o * (fg - bg)
            d = meas_lin - (comp_lin + remaining * bg_lin)
            v = remaining * (fg_lin - bg_lin)
            denom = float(np.dot(v, v))
            if denom < eps:
                o = 0.0
            else:
                o = float(np.dot(d, v) / denom)
            o = float(np.clip(o, 0.0, 1.0))
            o_series[layer - 1] = o

            # Update state in linear
            comp_lin = comp_lin + (remaining * o) * fg_lin
            remaining = remaining * (1.0 - o)

        t = np.arange(1, num_layers + 1, dtype=np.float32) * layer_height
        o_list.append(o_series)
        t_list.append(t)

    return o_list, t_list


def build_control_list(
    df: pd.DataFrame, num_layers: int, layer_height: float
) -> List[np.ndarray]:
    """
    Build control input list U per trajectory with two controls per layer:
      u = [TD_I, layer_height]
    Returns a list of arrays shape (num_layers, 2).
    """
    U_list: List[np.ndarray] = []
    for _, row in df.iterrows():
        TD_I = float(row.at["Transmission Distance"])  # scalar
        U = np.stack(
            [
                np.full(num_layers, TD_I, dtype=np.float32),
                np.full(num_layers, float(layer_height), dtype=np.float32),
            ],
            axis=1,
        )
        U_list.append(U)
    return U_list


# ---------------------
# SINDy training (scalar state with controls)
# ---------------------


def fit_sindy_with_control(
    trajectories: List[np.ndarray],
    t_list: List[np.ndarray],
    U_list_scaled: List[np.ndarray],
) -> ps.SINDy:
    """
    Fit continuous-time SINDy-with-control for a scalar state using multiple trajectories.
    Controls are already scaled.
    """
    feature_library = PolynomialLibrary(
        degree=3, include_interaction=True, include_bias=True
    )

    model = ps.SINDy(
        feature_library=feature_library,
        differentiation_method=ps.FiniteDifference(),
        optimizer=TorchOptimizer(),
    )

    x_list = [traj.reshape(-1, 1) for traj in trajectories]
    model.fit(x=x_list, t=t_list, u=U_list_scaled)
    return model


# ---------------------
# Public API
# ---------------------


def run_sindy_opacity(
    csv_path: str = "printed_colors.csv",
    num_layers: int = 16,
    layer_height: float = 0.04,
    plot_examples: bool = True,
    n_plot_samples: int = 6,
    random_seed: int | None = 123,
) -> Dict[str, object]:
    """
    Load CSV, invert compositing to get per-layer scalar opacities, fit SINDy-with-control
    model, print equation, and return the model, scaler, and other artifacts.
    """
    df = pd.read_csv(csv_path)

    # Extract scalar opacity trajectories (in linear space)
    o_list, t_list = extract_scalar_opacity_series(df, num_layers, layer_height)

    # Build controls and scale them
    U_list = build_control_list(df, num_layers, layer_height)
    scaler = StandardScaler()
    U_stacked = np.vstack(U_list)
    scaler.fit(U_stacked)
    U_list_scaled = [scaler.transform(U) for U in U_list]

    # Fit SINDy model (scalar state)
    sindy_model = fit_sindy_with_control(o_list, t_list, U_list_scaled)

    print("\nDiscovered opacity dynamics (with controls TD_I & layer_height):")
    sindy_model.print()

    # Smoke simulate first trajectory
    try:
        sim_t = t_list[0]
        sim_u = U_list_scaled[0]
        x0 = np.array([o_list[0][0]], dtype=np.float32)
        x_sim = np.array(sindy_model.simulate(x0, sim_t, u=sim_u)).reshape(-1)
        x_sim = np.clip(x_sim, 0.0, 1.0)
        print("\nSample simulated opacity (first trajectory):", x_sim[:8])
    except Exception as e:
        print("Simulation failed:", e)

    # Evaluate simulation error across all trajectories (opacity + final RGB)
    metrics = evaluate_simulation_and_color_error(
        sindy_model, df, o_list, t_list, U_list, U_list_scaled
    )
    print("\nOpacity and color simulation error (averaged across trajectories):")
    print(
        f"Opacity: MAE={metrics['opacity']['MAE']:.4f}, RMSE={metrics['opacity']['RMSE']:.4f}"
    )
    print(
        f"Final RGB: MAE={metrics['rgb']['MAE']:.4f}, RMSE={metrics['rgb']['RMSE']:.4f}"
    )

    artifacts: Dict[str, object] = {
        "model": sindy_model,
        "scaler": scaler,
        "controls": ["Transmission Distance", "Layer Height"],
        "num_layers": num_layers,
        "layer_height": layer_height,
    }

    if plot_examples:
        try:
            plot_random_color_comparison(
                sindy_model,
                scaler,
                df,
                o_list,
                t_list,
                U_list,
                num_layers=num_layers,
                n_samples=n_plot_samples,
                seed=random_seed,
            )
        except Exception as e:
            print("Plotting failed:", e)

    return artifacts


def evaluate_simulation_and_color_error(
    model: ps.SINDy,
    df: pd.DataFrame,
    o_list: List[np.ndarray],
    t_list: List[np.ndarray],
    U_list: List[np.ndarray],
    U_list_scaled: List[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Simulate each trajectory and compute errors vs extracted opacity and final RGB.
    Returns a dict with MAE and RMSE for opacity (scalar) and RGB (final color).
    """
    mae_o_all, rmse_o_all = [], []
    mae_rgb_all, rmse_rgb_all = [], []

    for i, o_series in enumerate(o_list):
        t = t_list[i]
        u_scaled = U_list_scaled[i]
        x0 = np.array([o_series[0]], dtype=np.float32)
        try:
            sim = np.array(model.simulate(x0, t, u=u_scaled)).reshape(-1)
        except Exception:
            continue
        sim = np.clip(sim, 0.0, 1.0)
        n = min(len(sim), len(o_series))
        err = sim[:n] - o_series[:n]
        mae_o_all.append(float(np.mean(np.abs(err))))
        rmse_o_all.append(float(np.sqrt(np.mean(err**2))))

        # Compare final RGB (work in linear domain for compositing)
        row = df.iloc[i]
        bg_lin = hex_to_linear_rgb01(str(row["Background Material"]))
        fg_lin = hex_to_linear_rgb01(str(row["Layer Material"]))

        # Measured final color (sRGB for comparison)
        try:
            meas_hex = str(row[f"Layer {len(o_series)}"])
            meas_rgb = hex_to_rgb(meas_hex)
        except Exception:
            last_layer_col = [c for c in df.columns if c.startswith("Layer ")][-1]
            meas_rgb = hex_to_rgb(str(row[last_layer_col]))

        _, pred_rgb_srgb = composite_from_opacity_series(bg_lin, fg_lin, sim[:n])
        # Color error in 0-1 space averaged across channels
        e_rgb = (pred_rgb_srgb - meas_rgb) / 255.0
        mae_rgb_all.append(float(np.mean(np.abs(e_rgb))))
        rmse_rgb_all.append(float(np.sqrt(np.mean(e_rgb**2))))
        print(
            f"Sample {i}: Average RGB error (MAE) = {np.mean(mae_rgb_all):.4f}, Average Opacity error (MAE) = {np.mean(mae_o_all):.4f}"
        )

    return {
        "opacity": {
            "MAE": float(np.mean(mae_o_all)) if mae_o_all else float("nan"),
            "RMSE": float(np.mean(rmse_o_all)) if rmse_o_all else float("nan"),
        },
        "rgb": {
            "MAE": float(np.mean(mae_rgb_all)) if mae_rgb_all else float("nan"),
            "RMSE": float(np.mean(rmse_rgb_all)) if rmse_rgb_all else float("nan"),
        },
    }


# ---------------------
# Visualization: measured vs predicted composite colors
# ---------------------


def _rgb_to_patch(rgb: np.ndarray, size: int = 40) -> np.ndarray:
    rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
    arr = np.ones((size, size, 3), dtype=np.float32)
    arr[..., 0] = rgb01[0]
    arr[..., 1] = rgb01[1]
    arr[..., 2] = rgb01[2]
    return arr


def plot_random_color_comparison(
    model: ps.SINDy,
    scaler: StandardScaler,
    df: pd.DataFrame,
    o_list: List[np.ndarray],
    t_list: List[np.ndarray],
    U_list: List[np.ndarray],
    num_layers: int,
    n_samples: int = 6,
    seed: int | None = None,
    save_path: str | None = None,
):
    """
    Pick 5-10 random rows from df, simulate opacity with the discovered
    ODE and composite predicted colors. Plot measured final color vs predicted
    final color side by side per sample with MAE in the title.
    """
    rng = np.random.default_rng(seed)
    N = len(df)
    n = max(1, min(n_samples, N))
    idxs = rng.choice(N, size=n, replace=False)

    rows = [df.iloc[int(i)] for i in idxs]

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(6, 2.2 * n))
    if n == 1:
        axes = np.array([axes])

    for row_idx, (ax_left, ax_right) in enumerate(axes):
        i = int(idxs[row_idx])
        row = rows[row_idx]
        bg_lin = hex_to_linear_rgb01(str(row["Background Material"]))
        fg_lin = hex_to_linear_rgb01(str(row["Layer Material"]))

        # Measured final composite color from CSV (Layer num_layers, sRGB space)
        try:
            meas_hex = str(row[f"Layer {num_layers}"])
            meas_rgb = hex_to_rgb(meas_hex)
        except Exception:
            last_layer_col = [c for c in df.columns if c.startswith("Layer ")][-1]
            meas_rgb = hex_to_rgb(str(row[last_layer_col]))

        # Controls
        U = U_list[i]
        U_scaled = scaler.transform(U)

        # Simulate predicted opacity sequence
        t = t_list[i]
        x0 = np.array([o_list[i][0]], dtype=np.float32)
        try:
            sim = np.array(model.simulate(x0, t, u=U_scaled)).reshape(-1)
        except Exception:
            sim = np.zeros_like(t)
        sim = np.clip(sim, 0.0, 1.0)

        # Forward composite predicted final color (linear -> sRGB conversion inside)
        _, pred_rgb = composite_from_opacity_series(
            bg_lin01=bg_lin, fg_lin01=fg_lin, o_series=sim
        )

        # Build small color patches
        left_img = _rgb_to_patch(meas_rgb)
        right_img = _rgb_to_patch(pred_rgb)

        ax_left.imshow(left_img)
        ax_left.set_title("Measured")
        ax_left.axis("off")

        ax_right.imshow(right_img)
        err = np.mean(np.abs((pred_rgb - meas_rgb) / 255.0))
        ax_right.set_title(f"Predicted (MAE={err:.3f})")
        ax_right.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Run with default CSV name if available; show comparison.
    try:
        run_sindy_opacity(plot_examples=True, n_plot_samples=12)
    except FileNotFoundError:
        print(
            "CSV not found for SINDy opacity run. Please provide a valid csv_path or run within the project that contains printed_colors.csv."
        )
