import math
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import optuna
import concurrent.futures
import multiprocessing

from tqdm import tqdm

# optuna.logging.set_verbosity(optuna.logging.WARNING)


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


# ===== Candidate Model Functions =====
# ===== Candidate Model Functions (Models 1-50) =====
# Model 1: Linear
def model_linear(params, t, T):
    return params[0] * (t / T)


# Model 2: Quadratic
def model_quadratic(params, t, T):
    return params[0] * (t / T) ** 2


# Model 3: Cubic
def model_cubic(params, t, T):
    return params[0] * (t / T) ** 3


# Model 4: Power function
def model_power(params, t, T):
    return (t / T) ** params[0]


# Model 5: Exponential
def model_exponential(params, t, T):
    return 1 - math.exp(-params[0] * (t / T))


# Model 6: Logarithmic
def model_logarithmic(params, t, T):
    return (
        params[0] * math.log(1 + params[1] * (t / T)) / math.log(1 + params[1])
        if params[1] != 0
        else 0
    )


# Model 7: Sigmoid
def model_sigmoid(params, t, T):
    return 1 / (1 + math.exp(-params[0] * ((t / T) - params[1])))


# Model 8: Logistic
def model_logistic(params, t, T):
    return params[0] / (1 + math.exp(-params[1] * ((t / T) - params[2])))


# Model 9: Tanh
def model_tanh(params, t, T):
    return 0.5 * (math.tanh(params[0] * ((t / T) - params[1])) + 1)


# Model 10: Arctan
def model_arctan(params, t, T):
    return (2 / math.pi) * math.atan(params[0] * (t / T))


# Model 11: Inverse
def model_inverse(params, t, T):
    return params[0] * (t / T) / (params[1] + (t / T))


# Model 12: Square Root
def model_sqrt(params, t, T):
    return params[0] * math.sqrt(t / T)


# Model 13: New Model 4: Log-Linear
def new_model4(params, t, T):
    c, A, k, B = params
    return (A * math.log(1 + k * (t / T)) + B * (t / T)) / c


# Model 14: Piecewise Linear
def model_piecewise_linear(params, t, T):
    ratio = t / T
    if ratio < params[1]:
        return params[0] * ratio
    else:
        return params[0] * params[1] + params[2] * (ratio - params[1])


# Model 15: Piecewise Exponential
def model_piecewise_exponential(params, t, T):
    ratio = t / T
    if ratio < params[0]:
        return params[1] * ratio
    else:
        return params[1] * params[0] + params[2] * (ratio - params[0])


# Model 16: Quadratic then Saturate
def model_quadratic_saturate(params, t, T):
    val = params[0] * (t / T) ** 2 + params[1] * (t / T)
    return min(1, val)


# Model 17: Cubic then Saturate
def model_cubic_saturate(params, t, T):
    val = params[0] * (t / T) ** 3 + params[1] * (t / T) ** 2 + params[2] * (t / T)
    return min(1, val)


# Model 18: Combined Exponential and Linear
def model_exp_linear(params, t, T):
    return params[0] * (1 - math.exp(-params[1] * (t / T))) + params[2] * (t / T)


# Model 19: Inverse Logistic
def model_inverse_logistic(params, t, T):
    return params[0] / (1 + math.exp(params[1] * ((t / T) - params[2])))


# Model 20: Gaussian CDF
def model_gaussian_cdf(params, t, T):
    return (
        params[0]
        * 0.5
        * (1 + math.erf(((t / T) - params[1]) / (params[2] * math.sqrt(2))))
    )


# Model 21: Sigmoid Variation 2
def model_sigmoid_var2(params, t, T):
    return params[0] / (1 + math.exp(-params[1] * ((t / T) - params[2]))) + params[
        3
    ] * (t / T)


# Model 22: Weighted Average
def model_weighted_average(params, t, T):
    denom = params[0] + params[1] if (params[0] + params[1]) != 0 else 1e-6
    return (params[0] * (t / T) + params[1] * math.sqrt(t / T)) / denom


# Model 23: Logarithmic Variation
def model_logarithmic_var(params, t, T):
    return (
        params[0]
        * math.log(1 + params[1] * (t / T) + params[2] * (t / T) ** 2)
        / math.log(1 + params[1] + params[2])
    )


# Model 24: Sine-based
def model_sine(params, t, T):
    return params[0] * math.sin(params[1] * (t / T) * math.pi / 2)


# Model 25: Cosine-based
def model_cosine(params, t, T):
    return params[0] * (1 - math.cos(params[1] * (t / T) * math.pi / 2))


# Model 26: Combined Sine and Linear
def model_sine_linear(params, t, T):
    return params[0] * math.sin(params[1] * (t / T) * math.pi / 2) + params[2] * (t / T)


# Model 27: Combined Cosine and Linear
def model_cosine_linear(params, t, T):
    return params[0] * (1 - math.cos(params[1] * (t / T) * math.pi / 2)) + params[2] * (
        t / T
    )


# Model 28: Polynomial Degree 2
def model_poly2(params, t, T):
    return params[0] * (t / T) ** 2 + (1 - params[0]) * (t / T)


# Model 29: Polynomial Degree 3
def model_poly3(params, t, T):
    return (
        params[0] * (t / T) ** 3
        + params[1] * (t / T) ** 2
        + (1 - params[0] - params[1]) * (t / T)
    )


# Model 30: Exponential with Offset
def model_exp_offset(params, t, T):
    ratio = t / T
    if ratio > params[2]:
        return params[0] * (1 - math.exp(-params[1] * (ratio - params[2])))
    else:
        return 0


# Model 31: Logarithmic with Offset
def model_log_offset(params, t, T):
    ratio = t / T
    if ratio > params[2]:
        return (
            params[0]
            * math.log(1 + params[1] * (ratio - params[2]))
            / math.log(1 + params[1])
        )
    else:
        return 0


# Model 32: Piecewise Constant
def model_piecewise_constant(params, t, T):
    return params[0] if (t / T) < params[1] else params[2]


# Model 33: Linear then Saturate
def model_linear_saturate(params, t, T):
    return min(params[0] * (t / T), params[1])


# Model 34: Quadratic then Saturate Variant
def model_quad_saturate(params, t, T):
    return min(params[0] * (t / T) ** 2 + params[1] * (t / T), params[2])


# Model 35: Cubic then Saturate Variant
def model_cubic_saturate_var(params, t, T):
    return min(
        params[0] * (t / T) ** 3 + params[1] * (t / T) ** 2 + params[2] * (t / T),
        params[3],
    )


# Model 36: Harmonic Series
def model_harmonic(params, t, T):
    return params[0] * (1 - 1 / (1 + (t / T)))


# Model 37: Inverse Proportion
def model_inv_prop(params, t, T):
    return params[0] * (t / T) / (1 + params[1] * (t / T))


# Model 38: Damped Growth
def model_damped_growth(params, t, T):
    return params[0] * (t / T) * math.exp(-params[1] * (t / T))


# Model 39: Gompertz Function (S-shaped growth)
def model_gompertz(params, t, T):
    return params[0] * math.exp(-params[1] * math.exp(-params[2] * (t / T)))


# Model 40: Log-Logistic
def model_log_logistic(params, t, T):
    return params[0] / (1 + ((t / T) / params[1]) ** params[2])


# Model 41: Bass Diffusion
def model_bass(params, t, T):
    return params[0] * (
        (1 - math.exp(-params[1] * (t / T)))
        / (1 + params[2] * math.exp(-params[1] * (t / T)))
    )


# Model 42: Weibull CDF
def model_weibull(params, t, T):
    return params[0] * (1 - math.exp(-(((t / T) / params[1]) ** params[2])))


# Model 43: Gamma CDF Approximation
def model_gamma(params, t, T):
    return params[0] * (t / T) ** params[1] * math.exp(-params[2] * (t / T))


# Model 44: Custom Polynomial Degree 4
def model_poly4(params, t, T):
    return (
        params[0] * (t / T) ** 4
        + params[1] * (t / T) ** 3
        + params[2] * (t / T) ** 2
        + (1 - params[0] - params[1] - params[2]) * (t / T)
    )


# Model 45: Sinusoidal with Offset
def model_sine_offset(params, t, T):
    return params[0] * math.sin(params[1] * (t / T) * math.pi) + params[2] * (t / T)


# Model 46: Cosinusoidal with Offset
def model_cosine_offset(params, t, T):
    return params[0] * (1 - math.cos(params[1] * (t / T) * math.pi)) + params[2] * (
        t / T
    )


# Model 47: Double Sigmoid
def model_double_sigmoid(params, t, T):
    return (
        1
        / (1 + math.exp(-params[0] * ((t / T) - params[1])))
        * 1
        / (1 + math.exp(params[2] * ((t / T) - params[3])))
    )


# Model 48: Exponential Decay
def model_exp_decay(params, t, T):
    return params[0] * math.exp(-params[1] * (1 - (t / T)))


# Model 49: Quadratic Decay
def model_quad_decay(params, t, T):
    return params[0] * (1 - (t / T) ** 2)


# Model 50: Cubic Decay
def model_cubic_decay(params, t, T):
    return params[0] * (1 - (t / T) ** 3)


# ===== Additional Candidate Model Functions (Models 51-100) =====
# Model 51: Sine Squared
def model_sine_squared(params, t, T):
    return params[0] * (math.sin(params[1] * (t / T) * math.pi)) ** 2


# Model 52: Cosine Squared
def model_cosine_squared(params, t, T):
    return params[0] * (1 - math.cos(params[1] * (t / T) * math.pi)) ** 2


# Model 53: Polynomial-Exponential Combination
def model_poly_exp(params, t, T):
    return params[0] * (t / T) ** 2 * math.exp(-params[1] * (t / T))


# Model 54: Logarithmic plus Linear
def model_log_linear(params, t, T):
    return params[0] * math.log(1 + params[1] * (t / T)) + params[2] * (t / T)


# Model 55: Inverse Squared
def model_inverse_squared(params, t, T):
    return params[0] * (t / T) / (params[1] + (t / T) ** 2)


# Model 56: Piecewise Quadratic
def model_piecewise_quadratic(params, t, T):
    ratio = t / T
    if ratio < params[0]:
        return params[1] * (ratio) ** 2
    else:
        return params[1] * (params[0]) ** 2 + params[2] * (ratio - params[0]) ** 2


# Model 57: Sine with Phase Shift
def model_sine_phase(params, t, T):
    return params[0] * math.sin(params[1] * (t / T) * math.pi + params[2])


# Model 58: Cosine with Phase Shift
def model_cosine_phase(params, t, T):
    return params[0] * (1 - math.cos(params[1] * (t / T) * math.pi + params[2]))


# Model 59: Exponential Saturation
def model_exp_saturation(params, t, T):
    ratio = t / T
    return (
        (math.exp(params[0] * ratio) - 1) / (math.exp(params[0]) - 1)
        if params[0] != 0
        else ratio
    )


# Model 60: Power Function with Linear Offset
def model_power_linear(params, t, T):
    return (t / T) ** params[0] + params[1] * (t / T)


# Model 61: Logistic plus Quadratic
def model_logistic_quadratic(params, t, T):
    return (
        params[0] / (1 + math.exp(-params[1] * ((t / T) - params[2])))
        + params[3] * (t / T) ** 2
    )


# Model 62: Tanh plus Linear
def model_tanh_linear(params, t, T):
    return 0.5 * (math.tanh(params[0] * ((t / T) - params[1])) + 1) + params[2] * (
        t / T
    )


# Model 63: Reciprocal Linear
def model_reciprocal_linear(params, t, T):
    return 1 / (params[0] + params[1] * (1 - t / T))


# Model 64: Reciprocal Quadratic
def model_reciprocal_quadratic(params, t, T):
    return 1 / (params[0] + params[1] * (1 - t / T) ** 2)


# Model 65: Logarithm with Saturation
def model_log_saturate(params, t, T):
    return min(1, params[0] * math.log(1 + params[1] * (t / T)))


# Model 66: Arctan with Offset
def model_arctan_offset(params, t, T):
    return (2 / math.pi) * math.atan(params[0] * (t / T) + params[1])


# Model 67: Hyperbolic Secant
def model_sech(params, t, T):
    return params[0] / math.cosh(params[1] * (t / T))


# Model 68: Damped Sine
def model_damped_sine(params, t, T):
    return params[0] * math.sin(params[1] * (t / T)) * math.exp(-params[2] * (t / T))


# Model 69: Sum of Sine and Cosine
def model_sine_cosine(params, t, T):
    return params[0] * math.sin(params[1] * (t / T)) + params[2] * math.cos(
        params[3] * (t / T)
    )


# Model 70: Cubic Polynomial with Bias
def model_cubic_bias(params, t, T):
    return (
        params[0] * (t / T) ** 3
        + params[1] * (t / T) ** 2
        + params[2] * (t / T)
        + params[3]
    )


# Model 71: Quartic Polynomial
def model_quartic(params, t, T):
    return (
        params[0] * (t / T) ** 4
        + params[1] * (t / T) ** 3
        + params[2] * (t / T) ** 2
        + params[3] * (t / T)
        + params[4]
    )


# Model 72: Damped Sinusoid plus Linear
def model_damped_sinusoid_linear(params, t, T):
    return params[0] * math.sin(params[1] * (t / T)) * math.exp(
        -params[2] * (t / T)
    ) + params[3] * (t / T)


# Model 73: Log-Modified Exponential
def model_log_mod_exp(params, t, T):
    return (1 - math.exp(-params[0] * (t / T))) * math.log(1 + params[1] * (t / T))


# Model 74: Sum of Two Exponentials
def model_two_exp(params, t, T):
    return params[0] * (1 - math.exp(-params[1] * (t / T))) + params[2] * (
        1 - math.exp(-params[3] * (t / T))
    )


# Model 75: Weighted Power Average
def model_weighted_power(params, t, T):
    num = params[0] * (t / T) ** params[1] + params[2] * (t / T) ** params[3]
    denom = params[0] + params[2] if (params[0] + params[2]) != 0 else 1e-6
    return num / denom


# Model 76: Modified Logistic (Shifted Down)
def model_modified_logistic(params, t, T):
    return 1 / (1 + math.exp(-params[0] * ((t / T) - params[1]))) - 0.5


# Model 77: Damped Tanh
def model_damped_tanh(params, t, T):
    return 0.5 * (math.tanh(params[0] * (t / T)) + 1) * math.exp(-params[1] * (t / T))


# Model 78: Shifted Logistic with Linear Decay
def model_shifted_logistic(params, t, T):
    return params[0] / (1 + math.exp(-params[1] * ((t / T) - params[2]))) + params[
        3
    ] * (1 - t / T)


# Model 79: Polynomial Blend
def model_poly_blend(params, t, T):
    ratio = t / T
    return (ratio ** params[0]) / (ratio ** params[0] + (1 - ratio) ** params[1])


# Model 80: Generalized Mean Blend
def model_generalized_mean(params, t, T):
    ratio = t / T
    num = params[0] * (ratio) ** params[1] + params[2] * (1 - ratio) ** params[3]
    denom = params[0] + params[2] if (params[0] + params[2]) != 0 else 1e-6
    return (num / denom) ** (1 / params[4])


# Model 81: Oscillatory Decay
def model_oscillatory_decay(params, t, T):
    return (
        params[0]
        * (t / T)
        * (1 + params[1] * math.cos(params[2] * (t / T)))
        * math.exp(-params[3] * (t / T))
    )


# Model 82: Hyperbolic Function
def model_hyperbolic(params, t, T):
    return params[0] * (t / T) / (params[1] + (t / T))


# Model 83: Polynomial plus Logarithm
def model_poly_log(params, t, T):
    return params[0] * (t / T) + params[1] * math.log(1 + (t / T))


# Model 84: Square plus Square Root
def model_square_sqrt(params, t, T):
    return params[0] * (t / T) ** 2 + params[1] * math.sqrt(t / T)


# Model 85: Weighted Sum of Exponentials
def model_weighted_exp(params, t, T):
    return params[0] * math.exp(-params[1] * (1 - (t / T))) + params[2] * math.exp(
        -params[3] * (t / T)
    )


# Model 86: Rational Function
def model_rational(params, t, T):
    ratio = t / T
    denom = params[2] * ratio + params[3]
    return (params[0] * ratio + params[1]) / (denom if denom != 0 else 1e-6)


# Model 87: Quadratic plus Log Correction
def model_quad_log(params, t, T):
    return params[0] * (t / T) ** 2 + params[1] * math.log(1 + params[2] * (t / T))


# Model 88: Piecewise Combination
def model_piecewise_combo(params, t, T):
    ratio = t / T
    if ratio < params[0]:
        return params[1] * (ratio) ** 2
    else:
        return params[2] * (ratio) + params[3]


# Model 89: Cubic with Saturation
def model_cubic_sat(params, t, T):
    return min(
        1,
        params[0] * (t / T) ** 3
        + params[1] * (t / T) ** 2
        + params[2] * (t / T)
        + params[3],
    )


# Model 90: Sinusoid plus Polynomial
def model_sinusoid_poly(params, t, T):
    return params[0] * (t / T) + params[1] * math.sin(params[2] * (t / T) * math.pi)


# Model 91: Sum of Two Logistic Functions
def model_sum_logistics(params, t, T):
    return params[0] / (1 + math.exp(-params[1] * ((t / T) - params[2]))) + params[
        3
    ] / (1 + math.exp(-params[4] * ((t / T) - params[5])))


# Model 92: Tanh plus Linear (Variant)
def model_tanh_linear2(params, t, T):
    return 0.5 * (math.tanh(params[0] * ((t / T) - params[1])) + 1) + params[2] * (
        t / T
    )


# Model 93: Arctan Blend
def model_arctan_blend(params, t, T):
    return (2 / math.pi) * math.atan(params[0] * (t / T) + params[1] * (t / T) ** 2)


# Model 94: Exponential Blend
def model_exponential_blend(params, t, T):
    return 1 - math.exp(-params[0] * (t / T) ** params[1])


# Model 95: Sine Blend Squared
def model_sine_blend_squared(params, t, T):
    return (math.sin(params[0] * (t / T) * math.pi)) ** params[1]


# Model 96: Cosine Blend Squared
def model_cosine_blend_squared(params, t, T):
    return 1 - (math.cos(params[0] * (t / T) * math.pi)) ** params[1]


# Model 97: Oscillatory Logistic
def model_oscillatory_logistic(params, t, T):
    return (
        params[0]
        / (1 + math.exp(-params[1] * ((t / T) - params[2])))
        * (1 + params[3] * math.sin(params[4] * (t / T)))
    )


# Model 98: Damped Harmonic
def model_damped_harmonic(params, t, T):
    return (
        params[0]
        * (t / T)
        * math.exp(-params[1] * (t / T))
        * math.cos(params[2] * (t / T))
    )


# Model 99: Sigmoid Squared
def model_sigmoid_squared(params, t, T):
    return (1 / (1 + math.exp(-params[0] * ((t / T) - params[1])))) ** 2


# Model 100: Composite Function
def model_composite(params, t, T):
    return (
        params[0] * (1 - math.exp(-params[1] * (t / T)))
        + params[2] * (t / T) ** params[3]
    )


# Model 101: Channel Linear
def model_channel_linear(params, t, T):
    ratio = t / T
    return np.array([params[0] * ratio, params[1] * ratio, params[2] * ratio])


# Model 102: Channel Quadratic
def model_channel_quadratic(params, t, T):
    ratio = t / T
    return np.array([params[0] * ratio**2, params[1] * ratio**2, params[2] * ratio**2])


# Model 103: Channel Cubic
def model_channel_cubic(params, t, T):
    ratio = t / T
    return np.array([params[0] * ratio**3, params[1] * ratio**3, params[2] * ratio**3])


# Model 104: Channel Exponential
def model_channel_exponential(params, t, T):
    ratio = t / T
    return np.array(
        [
            1 - math.exp(-params[0] * ratio),
            1 - math.exp(-params[1] * ratio),
            1 - math.exp(-params[2] * ratio),
        ]
    )


# Model 105: Channel Logarithmic (6 parameters: two per channel)
def model_channel_logarithmic(params, t, T):
    ratio = t / T
    val_r = (
        params[0] * math.log(1 + params[1] * ratio) / math.log(1 + params[1])
        if params[1] != 0
        else 0
    )
    val_g = (
        params[2] * math.log(1 + params[3] * ratio) / math.log(1 + params[3])
        if params[3] != 0
        else 0
    )
    val_b = (
        params[4] * math.log(1 + params[5] * ratio) / math.log(1 + params[5])
        if params[5] != 0
        else 0
    )
    return np.array([val_r, val_g, val_b])


# Model 106: Channel Sigmoid (6 parameters)
def model_channel_sigmoid(params, t, T):
    ratio = t / T
    val_r = 1 / (1 + math.exp(-params[0] * (ratio - params[1])))
    val_g = 1 / (1 + math.exp(-params[2] * (ratio - params[3])))
    val_b = 1 / (1 + math.exp(-params[4] * (ratio - params[5])))
    return np.array([val_r, val_g, val_b])


# Model 107: Channel Logistic (9 parameters)
def model_channel_logistic(params, t, T):
    ratio = t / T
    val_r = params[0] / (1 + math.exp(-params[1] * (ratio - params[2])))
    val_g = params[3] / (1 + math.exp(-params[4] * (ratio - params[5])))
    val_b = params[6] / (1 + math.exp(-params[7] * (ratio - params[8])))
    return np.array([val_r, val_g, val_b])


# Model 108: Channel Tanh (6 parameters)
def model_channel_tanh(params, t, T):
    ratio = t / T
    val_r = 0.5 * (math.tanh(params[0] * (ratio - params[1])) + 1)
    val_g = 0.5 * (math.tanh(params[2] * (ratio - params[3])) + 1)
    val_b = 0.5 * (math.tanh(params[4] * (ratio - params[5])) + 1)
    return np.array([val_r, val_g, val_b])


# Model 109: Channel Arctan (3 parameters)
def model_channel_arctan(params, t, T):
    ratio = t / T
    return np.array(
        [
            (2 / math.pi) * math.atan(params[0] * ratio),
            (2 / math.pi) * math.atan(params[1] * ratio),
            (2 / math.pi) * math.atan(params[2] * ratio),
        ]
    )


# Model 110: Channel Inverse (6 parameters)
def model_channel_inverse(params, t, T):
    ratio = t / T
    val_r = params[0] * ratio / (params[1] + ratio)
    val_g = params[2] * ratio / (params[3] + ratio)
    val_b = params[4] * ratio / (params[5] + ratio)
    return np.array([val_r, val_g, val_b])


# Model 111: Channel Square Root (3 parameters)
def model_channel_sqrt(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * math.sqrt(ratio),
            params[1] * math.sqrt(ratio),
            params[2] * math.sqrt(ratio),
        ]
    )


# Model 112: Channel Piecewise Linear (9 parameters)
def model_channel_piecewise_linear(params, t, T):
    ratio = t / T

    def piecewise(a, b, c):
        return a * ratio if ratio < b else a * b + c * (ratio - b)

    return np.array(
        [
            piecewise(params[0], params[1], params[2]),
            piecewise(params[3], params[4], params[5]),
            piecewise(params[6], params[7], params[8]),
        ]
    )


# Model 113: Channel Combined Exponential + Linear (9 parameters)
def model_channel_exp_linear(params, t, T):
    ratio = t / T

    def func(a, b, c):
        return a * (1 - math.exp(-b * ratio)) + c * ratio

    return np.array(
        [
            func(params[0], params[1], params[2]),
            func(params[3], params[4], params[5]),
            func(params[6], params[7], params[8]),
        ]
    )


# Model 114: Channel Sine-based (6 parameters)
def model_channel_sine(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * math.sin(params[1] * ratio * math.pi / 2),
            params[2] * math.sin(params[3] * ratio * math.pi / 2),
            params[4] * math.sin(params[5] * ratio * math.pi / 2),
        ]
    )


# Model 115: Channel Cosine-based (6 parameters)
def model_channel_cosine(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * (1 - math.cos(params[1] * ratio * math.pi / 2)),
            params[2] * (1 - math.cos(params[3] * ratio * math.pi / 2)),
            params[4] * (1 - math.cos(params[5] * ratio * math.pi / 2)),
        ]
    )


# Model 116: Channel Polynomial Degree 2 (6 parameters)
def model_channel_poly2(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * ratio**2 + params[1] * ratio,
            params[2] * ratio**2 + params[3] * ratio,
            params[4] * ratio**2 + params[5] * ratio,
        ]
    )


# Model 117: Channel Polynomial Degree 3 (9 parameters)
def model_channel_poly3(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * ratio**3 + params[1] * ratio**2 + params[2] * ratio,
            params[3] * ratio**3 + params[4] * ratio**2 + params[5] * ratio,
            params[6] * ratio**3 + params[7] * ratio**2 + params[8] * ratio,
        ]
    )


# Model 118: Channel Sine + Linear (9 parameters)
def model_channel_sine_linear(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * math.sin(params[1] * ratio * math.pi / 2) + params[2] * ratio,
            params[3] * math.sin(params[4] * ratio * math.pi / 2) + params[5] * ratio,
            params[6] * math.sin(params[7] * ratio * math.pi / 2) + params[8] * ratio,
        ]
    )


# Model 119: Channel Cosine + Linear (9 parameters)
def model_channel_cosine_linear(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * (1 - math.cos(params[1] * ratio * math.pi / 2))
            + params[2] * ratio,
            params[3] * (1 - math.cos(params[4] * ratio * math.pi / 2))
            + params[5] * ratio,
            params[6] * (1 - math.cos(params[7] * ratio * math.pi / 2))
            + params[8] * ratio,
        ]
    )


# Model 120: Channel Weighted Average (6 parameters)
def model_channel_weighted_average(params, t, T):
    ratio = t / T

    def channel_func(a, b):
        denom = a + b if (a + b) != 0 else 1e-6
        return (a * ratio + b * math.sqrt(ratio)) / denom

    return np.array(
        [
            channel_func(params[0], params[1]),
            channel_func(params[2], params[3]),
            channel_func(params[4], params[5]),
        ]
    )


# Model 121: Channel Logarithmic Variation (9 parameters)
def model_channel_log_var(params, t, T):
    ratio = t / T

    def channel_func(a, b, c):
        denom = math.log(1 + b + c) if (b + c) != 0 else 1
        return a * math.log(1 + b * ratio + c * ratio**2) / denom

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 122: Channel Sinusoidal with Offset (9 parameters)
def model_channel_sine_offset(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * math.sin(params[1] * ratio * math.pi) + params[2] * ratio,
            params[3] * math.sin(params[4] * ratio * math.pi) + params[5] * ratio,
            params[6] * math.sin(params[7] * ratio * math.pi) + params[8] * ratio,
        ]
    )


# Model 123: Channel Cosinusoidal with Offset (9 parameters)
def model_channel_cosine_offset(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * (1 - math.cos(params[1] * ratio * math.pi)) + params[2] * ratio,
            params[3] * (1 - math.cos(params[4] * ratio * math.pi)) + params[5] * ratio,
            params[6] * (1 - math.cos(params[7] * ratio * math.pi)) + params[8] * ratio,
        ]
    )


# Model 124: Channel Double Sigmoid (12 parameters)
def model_channel_double_sigmoid(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d):
        return (
            1 / (1 + math.exp(-a * (ratio - b))) * 1 / (1 + math.exp(c * (ratio - d)))
        )

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3]),
            channel_func(params[4], params[5], params[6], params[7]),
            channel_func(params[8], params[9], params[10], params[11]),
        ]
    )


# Model 125: Channel Exponential Decay (6 parameters)
def model_channel_exp_decay(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * math.exp(-params[1] * (1 - ratio)),
            params[2] * math.exp(-params[3] * (1 - ratio)),
            params[4] * math.exp(-params[5] * (1 - ratio)),
        ]
    )


# Model 126: Channel Quadratic Decay (3 parameters)
def model_channel_quad_decay(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * (1 - ratio**2),
            params[1] * (1 - ratio**2),
            params[2] * (1 - ratio**2),
        ]
    )


# Model 127: Channel Cubic Decay (3 parameters)
def model_channel_cubic_decay(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * (1 - ratio**3),
            params[1] * (1 - ratio**3),
            params[2] * (1 - ratio**3),
        ]
    )


# Model 128: Channel Sine Squared (6 parameters)
def model_channel_sine_squared(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * (math.sin(params[1] * ratio * math.pi / 2)) ** 2,
            params[2] * (math.sin(params[3] * ratio * math.pi / 2)) ** 2,
            params[4] * (math.sin(params[5] * ratio * math.pi / 2)) ** 2,
        ]
    )


# Model 129: Channel Cosine Squared (6 parameters)
def model_channel_cosine_squared(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * (1 - (math.cos(params[1] * ratio * math.pi / 2)) ** 2),
            params[2] * (1 - (math.cos(params[3] * ratio * math.pi / 2)) ** 2),
            params[4] * (1 - (math.cos(params[5] * ratio * math.pi / 2)) ** 2),
        ]
    )


# Model 130: Channel Oscillatory Logistic (15 parameters)
def model_channel_oscillatory_logistic(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d, e):
        return a / (1 + math.exp(-b * (ratio - c))) * (1 + d * math.sin(e * ratio))

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3], params[4]),
            channel_func(params[5], params[6], params[7], params[8], params[9]),
            channel_func(params[10], params[11], params[12], params[13], params[14]),
        ]
    )


# Model 131: Channel Damped Harmonic (9 parameters)
def model_channel_damped_harmonic(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0]
            * ratio
            * math.exp(-params[1] * ratio)
            * math.cos(params[2] * ratio),
            params[3]
            * ratio
            * math.exp(-params[4] * ratio)
            * math.cos(params[5] * ratio),
            params[6]
            * ratio
            * math.exp(-params[7] * ratio)
            * math.cos(params[8] * ratio),
        ]
    )


# Model 132: Channel Sigmoid Squared (6 parameters)
def model_channel_sigmoid_squared(params, t, T):
    ratio = t / T
    val_r = (1 / (1 + math.exp(-params[0] * (ratio - params[1])))) ** 2
    val_g = (1 / (1 + math.exp(-params[2] * (ratio - params[3])))) ** 2
    val_b = (1 / (1 + math.exp(-params[4] * (ratio - params[5])))) ** 2
    return np.array([val_r, val_g, val_b])


# Model 133: Channel Composite Function (12 parameters)
def model_channel_composite(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d):
        return a * (1 - math.exp(-b * ratio)) + c * (ratio**d)

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3]),
            channel_func(params[4], params[5], params[6], params[7]),
            channel_func(params[8], params[9], params[10], params[11]),
        ]
    )


# Model 134: Channel Piecewise Constant (9 parameters)
def model_channel_piecewise_constant(params, t, T):
    ratio = t / T

    def channel_func(thresh, val1, val2):
        return val1 if ratio < thresh else val2

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 135: Channel Linear then Saturate (6 parameters)
def model_channel_linear_saturate(params, t, T):
    ratio = t / T
    return np.array(
        [
            min(params[0] * ratio, params[1]),
            min(params[2] * ratio, params[3]),
            min(params[4] * ratio, params[5]),
        ]
    )


# Model 136: Channel Quadratic then Saturate (9 parameters)
def model_channel_quadratic_saturate(params, t, T):
    ratio = t / T

    def channel_func(a, b, c):
        return min(a * ratio**2 + b * ratio, c)

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 137: Channel Cubic then Saturate (12 parameters)
def model_channel_cubic_saturate(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d):
        return min(a * ratio**3 + b * ratio**2 + c * ratio, d)

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3]),
            channel_func(params[4], params[5], params[6], params[7]),
            channel_func(params[8], params[9], params[10], params[11]),
        ]
    )


# Model 138: Channel Damped Growth (9 parameters)
def model_channel_damped_growth(params, t, T):
    ratio = t / T

    def channel_func(a, b, c):
        return a * ratio * math.exp(-b * ratio) + c * ratio

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 139: Channel Gompertz (9 parameters)
def model_channel_gompertz(params, t, T):
    ratio = t / T

    def channel_func(a, b, c):
        return a * math.exp(-b * math.exp(-c * ratio))

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 140: Channel Log-Logistic (9 parameters)
def model_channel_log_logistic(params, t, T):
    ratio = t / T

    def channel_func(a, b, c):
        return a / (1 + (ratio / b) ** c) if b != 0 else 0

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 141: Channel Bass Diffusion (9 parameters)
def model_channel_bass(params, t, T):
    ratio = t / T

    def channel_func(a, b, c):
        return a * ((1 - math.exp(-b * ratio)) / (1 + c * math.exp(-b * ratio)))

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 142: Channel Weibull CDF (9 parameters)
def model_channel_weibull(params, t, T):
    ratio = t / T

    def channel_func(a, b, c):
        return a * (1 - math.exp(-((ratio / b) ** c))) if b != 0 else 0

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 143: Channel Gamma CDF Approximation (9 parameters)
def model_channel_gamma(params, t, T):
    ratio = t / T

    def channel_func(a, b, c):
        return a * (ratio**b) * math.exp(-c * ratio)

    return np.array(
        [
            channel_func(params[0], params[1], params[2]),
            channel_func(params[3], params[4], params[5]),
            channel_func(params[6], params[7], params[8]),
        ]
    )


# Model 144: Channel Polynomial Degree 4 (12 parameters)
def model_channel_poly4(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d):
        return a * ratio**4 + b * ratio**3 + c * ratio**2 + d * ratio

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3]),
            channel_func(params[4], params[5], params[6], params[7]),
            channel_func(params[8], params[9], params[10], params[11]),
        ]
    )


# Model 145: Channel Sine with Phase Shift (9 parameters)
def model_channel_sine_phase(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * math.sin(params[1] * ratio * math.pi + params[2]),
            params[3] * math.sin(params[4] * ratio * math.pi + params[5]),
            params[6] * math.sin(params[7] * ratio * math.pi + params[8]),
        ]
    )


# Model 146: Channel Cosinusoid with Phase Shift (9 parameters)
def model_channel_cosine_phase(params, t, T):
    ratio = t / T
    return np.array(
        [
            params[0] * (1 - math.cos(params[1] * ratio * math.pi + params[2])),
            params[3] * (1 - math.cos(params[4] * ratio * math.pi + params[5])),
            params[6] * (1 - math.cos(params[7] * ratio * math.pi + params[8])),
        ]
    )


# Model 147: Channel Sum of Two Exponentials (12 parameters)
def model_channel_two_exp(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d):
        return a * (1 - math.exp(-b * ratio)) + c * (1 - math.exp(-d * ratio))

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3]),
            channel_func(params[4], params[5], params[6], params[7]),
            channel_func(params[8], params[9], params[10], params[11]),
        ]
    )


# Model 148: Channel Weighted Sum of Exponentials (12 parameters)
def model_channel_weighted_exp(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d):
        return a * math.exp(-b * (1 - ratio)) + c * math.exp(-d * ratio)

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3]),
            channel_func(params[4], params[5], params[6], params[7]),
            channel_func(params[8], params[9], params[10], params[11]),
        ]
    )


# Model 149: Channel Rational Function (12 parameters)
def model_channel_rational(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d):
        denom = c * ratio + d if (c * ratio + d) != 0 else 1e-6
        return (a * ratio + b) / denom

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3]),
            channel_func(params[4], params[5], params[6], params[7]),
            channel_func(params[8], params[9], params[10], params[11]),
        ]
    )


# Model 150: Channel Composite Function Variant (12 parameters)
def model_channel_composite_variant(params, t, T):
    ratio = t / T

    def channel_func(a, b, c, d):
        return a * (1 - math.exp(-b * ratio)) + c * (ratio**d)

    return np.array(
        [
            channel_func(params[0], params[1], params[2], params[3]),
            channel_func(params[4], params[5], params[6], params[7]),
            channel_func(params[8], params[9], params[10], params[11]),
        ]
    )


# ===== Dictionary of 100 Candidate Models =====
# (Combining the original 50 and the additional 50)
extra_models = {
    "Model 1: Linear": {
        "func": model_linear,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 2: Quadratic": {
        "func": model_quadratic,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 3: Cubic": {
        "func": model_cubic,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 4: Power": {
        "func": model_power,
        "ranges": [(0.1, 5.0)],
        "param_names": ["p0"],
    },
    "Model 5: Exponential": {
        "func": model_exponential,
        "ranges": [(0.1, 10.0)],
        "param_names": ["p0"],
    },
    "Model 6: Logarithmic": {
        "func": model_logarithmic,
        "ranges": [(0.0, 1.0), (0.001, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 7: Sigmoid": {
        "func": model_sigmoid,
        "ranges": [(1.0, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 8: Logistic": {
        "func": model_logistic,
        "ranges": [(0.0, 1.0), (1.0, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 9: Tanh": {
        "func": model_tanh,
        "ranges": [(1.0, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 10: Arctan": {
        "func": model_arctan,
        "ranges": [(0.1, 10.0)],
        "param_names": ["p0"],
    },
    "Model 11: Inverse": {
        "func": model_inverse,
        "ranges": [(0.0, 1.0), (0.001, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 12: Sqrt": {
        "func": model_sqrt,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 13: New Model 4: Log-Linear": {
        "func": new_model4,
        "ranges": [(0.001, 2.0), (0.0, 1.0), (0.0, 100.0), (0.0, 1.0)],
        "param_names": ["c", "A", "k", "B"],
    },
    "Model 14: Piecewise Linear": {
        "func": model_piecewise_linear,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 15: Piecewise Exponential": {
        "func": model_piecewise_exponential,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 16: Quadratic Saturate": {
        "func": model_quadratic_saturate,
        "ranges": [(0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 17: Cubic Saturate": {
        "func": model_cubic_saturate,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 18: Exp + Linear": {
        "func": model_exp_linear,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 19: Inverse Logistic": {
        "func": model_inverse_logistic,
        "ranges": [(0.0, 1.0), (1.0, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 20: Gaussian CDF": {
        "func": model_gaussian_cdf,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.1, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 21: Sigmoid Var2": {
        "func": model_sigmoid_var2,
        "ranges": [(0.0, 1.0), (1.0, 10.0), (0.0, 1.0), (-1.0, 1.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 22: Weighted Average": {
        "func": model_weighted_average,
        "ranges": [(0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 23: Logarithmic Var": {
        "func": model_logarithmic_var,
        "ranges": [(0.0, 1.0), (0.001, 10.0), (0.0, 10.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 24: Sine": {
        "func": model_sine,
        "ranges": [(0.0, 1.0), (0.5, 2.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 25: Cosine": {
        "func": model_cosine,
        "ranges": [(0.0, 1.0), (0.5, 2.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 26: Sine + Linear": {
        "func": model_sine_linear,
        "ranges": [(0.0, 1.0), (0.5, 2.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 27: Cosine + Linear": {
        "func": model_cosine_linear,
        "ranges": [(0.0, 1.0), (0.5, 2.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 28: Poly Degree 2": {
        "func": model_poly2,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 29: Poly Degree 3": {
        "func": model_poly3,
        "ranges": [(0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 30: Exp with Offset": {
        "func": model_exp_offset,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 31: Log with Offset": {
        "func": model_log_offset,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 32: Piecewise Constant": {
        "func": model_piecewise_constant,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 33: Linear Saturate": {
        "func": model_linear_saturate,
        "ranges": [(0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 34: Quadratic Saturate Var": {
        "func": model_quad_saturate,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 35: Cubic Saturate Var": {
        "func": model_cubic_saturate_var,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 36: Harmonic": {
        "func": model_harmonic,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 37: Inverse Proportion": {
        "func": model_inv_prop,
        "ranges": [(0.0, 1.0), (0.0, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 38: Damped Growth": {
        "func": model_damped_growth,
        "ranges": [(0.0, 1.0), (0.0, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 39: Gompertz": {
        "func": model_gompertz,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.1, 10.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 40: Log-Logistic": {
        "func": model_log_logistic,
        "ranges": [(0.0, 1.0), (0.1, 1.0), (0.1, 10.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 41: Bass Diffusion": {
        "func": model_bass,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 10.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 42: Weibull CDF": {
        "func": model_weibull,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.1, 10.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 43: Gamma CDF": {
        "func": model_gamma,
        "ranges": [(0.0, 1.0), (0.1, 5.0), (0.1, 5.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 44: Poly Degree 4": {
        "func": model_poly4,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 45: Sine with Offset": {
        "func": model_sine_offset,
        "ranges": [(0.0, 1.0), (0.5, 2.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 46: Cosine with Offset": {
        "func": model_cosine_offset,
        "ranges": [(0.0, 1.0), (0.5, 2.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 47: Double Sigmoid": {
        "func": model_double_sigmoid,
        "ranges": [(1.0, 10.0), (0.0, 1.0), (1.0, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 48: Exponential Decay": {
        "func": model_exp_decay,
        "ranges": [(0.0, 1.0), (0.1, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 49: Quadratic Decay": {
        "func": model_quad_decay,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 50: Cubic Decay": {
        "func": model_cubic_decay,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 51: Sine Squared": {
        "func": model_sine_squared,
        "ranges": [(0.0, 1.0), (0.5, 2.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 52: Cosine Squared": {
        "func": model_cosine_squared,
        "ranges": [(0.0, 1.0), (0.5, 2.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 53: Poly-Exponential": {
        "func": model_poly_exp,
        "ranges": [(0.0, 1.0), (0.0, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 54: Log + Linear": {
        "func": model_log_linear,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 55: Inverse Squared": {
        "func": model_inverse_squared,
        "ranges": [(0.0, 1.0), (0.001, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 56: Piecewise Quadratic": {
        "func": model_piecewise_quadratic,
        "ranges": [(0.1, 0.9), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 57: Sine with Phase": {
        "func": model_sine_phase,
        "ranges": [(0.0, 1.0), (0.5, 2.0), (-3.14, 3.14)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 58: Cosine with Phase": {
        "func": model_cosine_phase,
        "ranges": [(0.0, 1.0), (0.5, 2.0), (-3.14, 3.14)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 59: Exponential Saturation": {
        "func": model_exp_saturation,
        "ranges": [(0.1, 10.0)],
        "param_names": ["p0"],
    },
    "Model 60: Power + Linear": {
        "func": model_power_linear,
        "ranges": [(0.1, 5.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 61: Logistic + Quadratic": {
        "func": model_logistic_quadratic,
        "ranges": [(0.0, 1.0), (1.0, 10.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 62: Tanh + Linear": {
        "func": model_tanh_linear,
        "ranges": [(1.0, 10.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 63: Reciprocal Linear": {
        "func": model_reciprocal_linear,
        "ranges": [(0.001, 1.0), (0.0, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 64: Reciprocal Quadratic": {
        "func": model_reciprocal_quadratic,
        "ranges": [(0.001, 1.0), (0.0, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 65: Log with Saturation": {
        "func": model_log_saturate,
        "ranges": [(0.0, 1.0), (0.1, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 66: Arctan with Offset": {
        "func": model_arctan_offset,
        "ranges": [(0.1, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 67: Hyperbolic Secant": {
        "func": model_sech,
        "ranges": [(0.0, 1.0), (0.1, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 68: Damped Sine": {
        "func": model_damped_sine,
        "ranges": [(0.0, 1.0), (0.5, 5.0), (0.0, 5.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 69: Sine + Cosine": {
        "func": model_sine_cosine,
        "ranges": [(0.0, 1.0), (0.5, 5.0), (0.0, 1.0), (0.5, 5.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 70: Cubic with Bias": {
        "func": model_cubic_bias,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-0.5, 0.5)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 71: Quartic Polynomial": {
        "func": model_quartic,
        "ranges": [(0.0, 1.0)] * 5,
        "param_names": ["p0", "p1", "p2", "p3", "p4"],
    },
    "Model 72: Damped Sinusoid + Linear": {
        "func": model_damped_sinusoid_linear,
        "ranges": [(0.0, 1.0), (0.5, 5.0), (0.0, 5.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 73: Log-Modified Exponential": {
        "func": model_log_mod_exp,
        "ranges": [(0.1, 10.0), (0.1, 10.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 74: Sum of Two Exponentials": {
        "func": model_two_exp,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 1.0), (0.1, 10.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 75: Weighted Power Average": {
        "func": model_weighted_power,
        "ranges": [(0.0, 1.0), (0.1, 5.0), (0.0, 1.0), (0.1, 5.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 76: Modified Logistic": {
        "func": model_modified_logistic,
        "ranges": [(1.0, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 77: Damped Tanh": {
        "func": model_damped_tanh,
        "ranges": [(1.0, 10.0), (0.0, 5.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 78: Shifted Logistic with Linear Decay": {
        "func": model_shifted_logistic,
        "ranges": [(0.0, 1.0), (1.0, 10.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 79: Polynomial Blend": {
        "func": model_poly_blend,
        "ranges": [(0.1, 5.0), (0.1, 5.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 80: Generalized Mean": {
        "func": model_generalized_mean,
        "ranges": [(0.0, 1.0), (0.1, 5.0), (0.0, 1.0), (0.1, 5.0), (0.1, 5.0)],
        "param_names": ["p0", "p1", "p2", "p3", "p4"],
    },
    "Model 81: Oscillatory Decay": {
        "func": model_oscillatory_decay,
        "ranges": [(0.0, 1.0), (-0.5, 0.5), (0.5, 5.0), (0.0, 5.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 82: Hyperbolic": {
        "func": model_hyperbolic,
        "ranges": [(0.0, 1.0), (0.001, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 83: Poly + Log": {
        "func": model_poly_log,
        "ranges": [(0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 84: Square plus Sqrt": {
        "func": model_square_sqrt,
        "ranges": [(0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 85: Weighted Sum Exp": {
        "func": model_weighted_exp,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 1.0), (0.1, 10.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 86: Rational Function": {
        "func": model_rational,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.1, 2.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 87: Quadratic + Log": {
        "func": model_quad_log,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.1, 10.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 88: Piecewise Combo": {
        "func": model_piecewise_combo,
        "ranges": [(0.1, 0.9), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 89: Cubic Saturation": {
        "func": model_cubic_sat,
        "ranges": [(0.0, 1.0)] * 4,
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 90: Sinusoid + Poly": {
        "func": model_sinusoid_poly,
        "ranges": [(0.0, 1.0), (0.0, 1.0), (0.5, 2.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 91: Sum of Two Logistics": {
        "func": model_sum_logistics,
        "ranges": [
            (0.0, 1.0),
            (1.0, 10.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (1.0, 10.0),
            (0.0, 1.0),
        ],
        "param_names": ["p0", "p1", "p2", "p3", "p4", "p5"],
    },
    "Model 92: Tanh + Linear 2": {
        "func": model_tanh_linear2,
        "ranges": [(1.0, 10.0), (0.0, 1.0), (0.0, 1.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 93: Arctan Blend": {
        "func": model_arctan_blend,
        "ranges": [(0.1, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 94: Exponential Blend": {
        "func": model_exponential_blend,
        "ranges": [(0.1, 10.0), (0.1, 5.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 95: Sine Blend Squared": {
        "func": model_sine_blend_squared,
        "ranges": [(0.5, 2.0), (1.0, 3.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 96: Cosine Blend Squared": {
        "func": model_cosine_blend_squared,
        "ranges": [(0.5, 2.0), (1.0, 3.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 97: Oscillatory Logistic": {
        "func": model_oscillatory_logistic,
        "ranges": [(0.0, 1.0), (1.0, 10.0), (0.0, 1.0), (-0.5, 0.5), (0.5, 5.0)],
        "param_names": ["p0", "p1", "p2", "p3", "p4"],
    },
    "Model 98: Damped Harmonic": {
        "func": model_damped_harmonic,
        "ranges": [(0.0, 1.0), (0.0, 5.0), (0.5, 5.0)],
        "param_names": ["p0", "p1", "p2"],
    },
    "Model 99: Sigmoid Squared": {
        "func": model_sigmoid_squared,
        "ranges": [(1.0, 10.0), (0.0, 1.0)],
        "param_names": ["p0", "p1"],
    },
    "Model 100: Composite Function": {
        "func": model_composite,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 1.0), (0.1, 5.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
    "Model 100: Composite Function": {
        "func": model_composite,
        "ranges": [(0.0, 1.0), (0.1, 10.0), (0.0, 1.0), (0.1, 5.0)],
        "param_names": ["p0", "p1", "p2", "p3"],
    },
}


# ===== Loss Calculation =====
# ===== Loss Calculation =====
def total_loss_for_model(model_func, params, df, layer_thickness, blending_mode="sRGB"):
    """
    Compute the combined loss and also per-layer loss using cumulative blending.
    For each row, we start with the background color and then update the composite
    color iteratively:
         composite = composite + alpha * (fg - composite)
    where alpha = manual_offsets + model_func(remaining_params, t, T)
    is clamped to [0,1].
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
        composite = np.array(bg, dtype=float)  # initialize composite as background

        for layer in range(1, num_layers + 1):
            try:
                t = layer * layer_thickness
                # The first three parameters are the manual offsets for R, G, B.
                manual_offsets = np.array(params[:3])
                # The remaining parameters are passed to model_func.
                model_params = params[3:]
                alpha_model = model_func(model_params, t, T)
                # If the model returns a scalar, replicate it to all channels.
                if np.isscalar(alpha_model):
                    alpha_model = np.array([alpha_model, alpha_model, alpha_model])
                # Combine the manual offsets with the model output.
                alpha = manual_offsets + alpha_model
                # Clamp each channel independently to [0,1]
                alpha = np.clip(alpha, 0.0, 1.0)
                # Cumulative blending: update composite color
                composite = composite + alpha * (fg - composite)
                meas = hex_to_rgb(row[f"Layer {layer}"])
                # L2 loss
                error = np.sum(np.abs(composite - meas))
            except Exception:
                error = 1e10
            per_layer_loss[layer - 1] += error
            total_loss += error

    total_loss /= num_rows * num_layers
    per_layer_loss /= num_rows
    return total_loss, per_layer_loss, params


# ===== Worker Function: Run a Single Independent Study =====
def run_worker_for_model(
    model_func, param_ranges, param_names, df, layer_thickness, trials, blending_mode
):
    def objective(trial):
        params = []
        for i, name in enumerate(param_names):
            low, high = param_ranges[i]
            params.append(trial.suggest_float(name, low, high))
        loss, _, _ = total_loss_for_model(
            model_func, params, df, layer_thickness, blending_mode
        )
        return loss

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=trials, n_jobs=1)
    best_params = [study.best_params[name] for name in param_names]
    best_loss = study.best_value
    return best_params, best_loss


# ===== Function to Search for the Best Model Parameters using All Cores =====
def run_search_for_model(
    model_name, model_data, df, layer_thickness, total_trials, blending_mode
):
    func = model_data["func"]
    param_ranges = model_data["ranges"]
    param_names = model_data["param_names"]

    # Insert three manual offset parameters (for R, G, B) at the beginning.
    for label in ["manual offset B", "manual offset G", "manual offset R"]:
        param_names.insert(0, label)
        param_ranges.insert(0, (-1.0, 1.0))

    cpu_count = multiprocessing.cpu_count()
    trials_per_worker = max(1, total_trials // cpu_count)
    print(
        f"Running {cpu_count} workers, each with {trials_per_worker} trials for {model_name}."
    )
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [
            executor.submit(
                run_worker_for_model,
                func,
                param_ranges,
                param_names,
                df,
                layer_thickness,
                trials_per_worker,
                blending_mode,
            )
            for _ in range(cpu_count)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception:
                pass

    best_params, best_loss = min(results, key=lambda x: x[1])
    print(f"Best Loss={best_loss:.2f} for {model_name} with params {best_params}")
    return {
        "model": model_name,
        "blending_mode": blending_mode,
        "best_params": best_params,
        "best_loss": best_loss,
        "param_names": param_names,
        "func": func,
    }


def total_loss_for_model(model_func, params, df, layer_thickness, blending_mode="sRGB"):
    """
    Compute the combined loss using an exponential accumulation model.

    Instead of standard iterative alpha blending,
    we accumulate an effective per-layer opacity and convert it to a cumulative effect via:
         effective_opacity = 1 - exp(-k * accumulated_alpha)
    where accumulated_alpha is the sum of (model_func output) from each layer.

    The predicted composite is then:
         composite = bg + effective_opacity * (fg - bg)

    Returns:
        total_loss: overall loss averaged over all rows and layers
        per_layer_loss: a NumPy array (length=num_layers) of average losses per layer
        params: the used parameters (for reference)
    """
    num_layers = 16
    total_loss = 0.0
    per_layer_loss = np.zeros(num_layers)
    num_rows = len(df)

    # k controls how fast the effective opacity saturates (adjust if needed)
    k = 2.0

    for idx, row in df.iterrows():
        T = float(row["Transmission Distance"]) * 0.1
        bg = np.array(hex_to_rgb(row["Background Material"]), dtype=float)
        fg = np.array(hex_to_rgb(row["Layer Material"]), dtype=float)
        accumulated_alpha = 0.0  # starting accumulated opacity

        for layer in range(1, num_layers + 1):
            try:
                t = layer * layer_thickness
                # The first three parameters are the manual offsets for R, G, B.
                manual_offsets = np.array(params[:3])
                # The remaining parameters are passed to model_func.
                model_params = params[3:]
                alpha_model = model_func(model_params, t, T)
                # If the model returns a scalar, replicate it to all channels.
                if np.isscalar(alpha_model):
                    alpha_model = np.array([alpha_model, alpha_model, alpha_model])
                # Combine the manual offsets with the model output.
                alpha = manual_offsets + alpha_model
                # Clamp each channel independently to [0,1]
                alpha = np.clip(alpha, 0.0, 1.0)
                # Compute predicted color for each channel:
                pred = bg + alpha * (fg - bg)
                meas = hex_to_rgb(row[f"Layer {layer}"])
                # l1 loss
                error = np.sum(np.abs(pred - meas))
            except Exception:
                traceback.print_exc()
                error = 1e10
            per_layer_loss[layer - 1] += error
            total_loss += error
    total_loss /= num_rows * num_layers
    per_layer_loss /= num_rows
    return total_loss, per_layer_loss, params


# ===== Worker Function: Run a Single Independent Study (Sequentially) =====
def run_worker_for_model(
    model_func, param_ranges, param_names, df, layer_thickness, trials, blending_mode
):
    def objective(trial):
        params = []
        for i, name in enumerate(param_names):
            low, high = param_ranges[i]
            params.append(trial.suggest_float(name, low, high))
        loss, _, _ = total_loss_for_model(
            model_func, params, df, layer_thickness, blending_mode
        )
        return loss

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(
        objective, n_trials=trials, n_jobs=1
    )  # Run trials sequentially in this worker
    best_params = [study.best_params[name] for name in param_names]
    best_loss = study.best_value
    return best_params, best_loss


# ===== Function to Search for the Best Model Parameters (No Internal Parallelism) =====
def run_search_for_model(
    model_name, model_data, df, layer_thickness, total_trials, blending_mode
):
    func = model_data["func"]
    param_ranges = model_data["ranges"]
    param_names = model_data["param_names"]

    # If you previously injected manual offset parameters, comment that out.
    for label in ["manual offset B", "manual offset G", "manual offset R"]:
        param_names.insert(0, label)
        param_ranges.insert(0, (-1.0, 1.0))

    # Run the worker sequentially for all trials for this model configuration.
    best_params, best_loss = run_worker_for_model(
        func,
        param_ranges,
        param_names,
        df,
        layer_thickness,
        total_trials,
        blending_mode,
    )
    print(f"Best Loss={best_loss:.2f} for {model_name} with params {best_params}")
    return {
        "model": model_name,
        "blending_mode": blending_mode,
        "best_params": best_params,
        "best_loss": best_loss,
        "param_names": param_names,
        "func": func,
    }


# ===== Main Function =====
def main():
    df = pd.read_csv("printed_colors.csv")
    layer_thickness = 0.04  # mm per layer
    blending_mode = "sRGB"  # using sRGB blending
    total_trials = 1000  # Total trials per model configuration

    extra_models = {
        "Model 60: Power + Linear": {
            "func": model_power_linear,
            "ranges": [(0.1, 5.0), (0.0, 1.0)],
            "param_names": ["p0", "p1"],
        },
        "Model 4: Power": {
            "func": model_power,
            "ranges": [(0.1, 5.0)],
            "param_names": ["p0"],
        },
        "Model 22: Weighted Average": {
            "func": model_weighted_average,
            "ranges": [(0.0, 1.0), (0.0, 1.0)],
            "param_names": ["p0", "p1"],
        },
        "Model 120: Channel Weighted Average": {
            "func": model_channel_weighted_average,
            "ranges": [(0.0, 1.0), (0.1, 5.0)] * 3,
            "param_names": ["p0", "p1", "p2", "p3", "p4", "p5"],
        },
        "Model 80: Generalized Mean": {
            "func": model_generalized_mean,
            "ranges": [(0.0, 1.0), (0.1, 5.0), (0.0, 1.0), (0.1, 5.0), (0.1, 5.0)],
            "param_names": ["p0", "p1", "p2", "p3", "p4"],
        },
        "Model 84: Square plus Sqrt": {
            "func": model_square_sqrt,
            "ranges": [(0.0, 1.0), (0.0, 1.0)],
            "param_names": ["p0", "p1"],
        },
        "Model 75: Weighted Power Average": {
            "func": model_weighted_power,
            "ranges": [(0.0, 1.0), (0.1, 5.0), (0.0, 1.0), (0.1, 5.0)],
            "param_names": ["p0", "p1", "p2", "p3"],
        },
    }

    # Use a ProcessPoolExecutor to run each model configuration concurrently.
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(extra_models)
    ) as executor:
        futures = []
        for model_name, model_data in extra_models.items():
            futures.append(
                executor.submit(
                    run_search_for_model,
                    model_name,
                    model_data,
                    df,
                    layer_thickness,
                    total_trials,
                    blending_mode,
                )
            )
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                results.append(future.result())
                # print result
                print(f"Result: {results[-1]}")
            except Exception:
                traceback.print_exc()

    results.sort(key=lambda x: x["best_loss"])
    print("\n=== Phase 1: Results (Independent Model Optimization) ===")
    for res in results:
        param_str = ", ".join(
            f"{name}={val:.4f}"
            for name, val in zip(res["param_names"], res["best_params"])
        )
        print(
            f"{res['model']} ({res['blending_mode']}): Loss = {res['best_loss']:.2f} with {param_str}"
        )

    # --- Plot Sample Comparison for the Best Model (all CSV rows) ---
    best_result = results[0]
    func = best_result["func"]
    best_params = best_result["best_params"]

    # Compute per-layer average error using the best parameters
    total_loss, per_layer_loss, _ = total_loss_for_model(
        func, best_params, df, layer_thickness, blending_mode
    )
    layers = np.arange(1, 17)

    # Plot average error per layer as a bar chart
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
