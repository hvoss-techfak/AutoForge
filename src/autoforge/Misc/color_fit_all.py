import random
import traceback
from typing import Tuple

import torch
import torch.jit
import torch.optim as optim
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from tqdm import tqdm

import torch
import torch.jit

##############################
# Candidate Model Functions  #
##############################


# Model 1: Linear
@torch.jit.script
def model_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T)


# Model 2: Quadratic
@torch.jit.script
def model_quadratic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) ** 2


# Model 3: Cubic
@torch.jit.script
def model_cubic(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return params[0] * (t / T) ** 3


# Model 4: Power function
@torch.jit.script
def model_power(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return (t / T) ** params[0]


# Model 5: Exponential
@torch.jit.script
def model_exponential(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 1 - torch.exp(-params[0] * (t / T))


# Model 6: Logarithmic
@torch.jit.script
def model_logarithmic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    if params[1] != 0:
        return params[0] * torch.log(1 + params[1] * (t / T)) / torch.log(1 + params[1])
    else:
        return torch.tensor(0.0)


# Model 7: Sigmoid
@torch.jit.script
def model_sigmoid(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 1 / (1 + torch.exp(-params[0] * ((t / T) - params[1])))


# Model 8: Logistic
@torch.jit.script
def model_logistic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] / (1 + torch.exp(-params[1] * ((t / T) - params[2])))


# Model 9: Tanh
@torch.jit.script
def model_tanh(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.tanh(params[0] * ((t / T) - params[1])) + 1)


# Model 10: Arctan
@torch.jit.script
def model_arctan(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (2 / 3.141592653589793) * torch.atan(params[0] * (t / T))


# Model 11: Inverse
@torch.jit.script
def model_inverse(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) / (params[1] + (t / T))


# Model 12: Square Root
@torch.jit.script
def model_sqrt(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return params[0] * torch.sqrt(t / T)


# Model 13: New Model 4: Log-Linear
@torch.jit.script
def new_model4(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    # params order: [c, A, k, B]
    return params[0] * torch.log(1 + params[1] * (t / T)) + params[2] * (t / T)


@torch.jit.script
def new_model4c(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    # params order: [c, A, k, B]
    return params[0] * torch.log(1 + params[1] * (t / T)) + params[2] * (t / T)


# Model 14: Piecewise Linear
@torch.jit.script
def model_piecewise_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    if ratio < params[1]:
        return params[0] * ratio
    else:
        return params[0] * params[1] + params[2] * (ratio - params[1])


# Model 15: Piecewise Exponential
@torch.jit.script
def model_piecewise_exponential(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    if ratio < params[0]:
        return params[1] * ratio
    else:
        return params[1] * params[0] + params[2] * (ratio - params[0])


# Model 16: Quadratic then Saturate
@torch.jit.script
def model_quadratic_saturate(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    val = params[0] * (t / T) ** 2 + params[1] * (t / T)
    return torch.clamp(val, max=1.0)


# Model 17: Cubic then Saturate
@torch.jit.script
def model_cubic_saturate(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    val = params[0] * (t / T) ** 3 + params[1] * (t / T) ** 2 + params[2] * (t / T)
    return torch.clamp(val, max=1.0)


# Model 18: Combined Exponential and Linear
@torch.jit.script
def model_exp_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (1 - torch.exp(-params[1] * (t / T))) + params[2] * (t / T)


# Model 19: Inverse Logistic
@torch.jit.script
def model_inverse_logistic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] / (1 + torch.exp(params[1] * ((t / T) - params[2])))


# Model 20: Gaussian CDF
@torch.jit.script
def model_gaussian_cdf(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0]
        * 0.5
        * (
            1
            + torch.erf(
                ((t / T) - params[1]) / (params[2] * torch.sqrt(torch.tensor(2.0)))
            )
        )
    )


# Model 21: Sigmoid Variation 2
@torch.jit.script
def model_sigmoid_var2(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] / (1 + torch.exp(-params[1] * ((t / T) - params[2]))) + params[
        3
    ] * (t / T)


# Model 22: Weighted Average
@torch.jit.script
def model_weighted_average(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    s = params[0] + params[1]
    denom = s if s != 0 else torch.tensor(1e-6)
    return (params[0] * (t / T) + params[1] * torch.sqrt(t / T)) / denom


# Model 23: Logarithmic Variation
@torch.jit.script
def model_logarithmic_var(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0]
        * torch.log(1 + params[1] * (t / T) + params[2] * (t / T) ** 2)
        / torch.log(1 + params[1] + params[2])
    )


# Model 24: Sine-based
@torch.jit.script
def model_sine(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return params[0] * torch.sin(params[1] * (t / T) * 3.141592653589793 / 2)


# Model 25: Cosine-based
@torch.jit.script
def model_cosine(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (1 - torch.cos(params[1] * (t / T) * 3.141592653589793 / 2))


# Model 26: Combined Sine and Linear
@torch.jit.script
def model_sine_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.sin(params[1] * (t / T) * 3.141592653589793 / 2) + params[
        2
    ] * (t / T)


# Model 27: Combined Cosine and Linear
@torch.jit.script
def model_cosine_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (
        1 - torch.cos(params[1] * (t / T) * 3.141592653589793 / 2)
    ) + params[2] * (t / T)


# Model 28: Polynomial Degree 2
@torch.jit.script
def model_poly2(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return params[0] * (t / T) ** 2 + (1 - params[0]) * (t / T)


# Model 29: Polynomial Degree 3
@torch.jit.script
def model_poly3(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return (
        params[0] * (t / T) ** 3
        + params[1] * (t / T) ** 2
        + (1 - params[0] - params[1]) * (t / T)
    )


# Model 30: Exponential with Offset
@torch.jit.script
def model_exp_offset(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    if ratio > params[2]:
        return params[0] * (1 - torch.exp(-params[1] * (ratio - params[2])))
    else:
        return torch.tensor(0.0)


# Model 31: Logarithmic with Offset
@torch.jit.script
def model_log_offset(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    if ratio > params[2]:
        return (
            params[0]
            * torch.log(1 + params[1] * (ratio - params[2]))
            / torch.log(1 + params[1])
        )
    else:
        return torch.tensor(0.0)


# Model 32: Piecewise Constant
@torch.jit.script
def model_piecewise_constant(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    if (t / T) < params[1]:
        return params[0]
    else:
        return params[2]


# Model 33: Linear then Saturate
@torch.jit.script
def model_linear_saturate(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return torch.clamp(params[0] * (t / T), max=params[1])


# Model 34: Quadratic then Saturate Variant
@torch.jit.script
def model_quad_saturate(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    val = params[0] * (t / T) ** 2 + params[1] * (t / T)
    return torch.clamp(val, max=params[2])


# Model 35: Cubic then Saturate Variant
@torch.jit.script
def model_cubic_saturate_var(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    val = params[0] * (t / T) ** 3 + params[1] * (t / T) ** 2 + params[2] * (t / T)
    return torch.clamp(val, max=params[3])


# Model 36: Harmonic Series
@torch.jit.script
def model_harmonic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (1 - 1 / (1 + (t / T)))


# Model 37: Inverse Proportion
@torch.jit.script
def model_inv_prop(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) / (1 + params[1] * (t / T))


# Model 38: Damped Growth
@torch.jit.script
def model_damped_growth(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) * torch.exp(-params[1] * (t / T))


# Model 39: Gompertz Function
@torch.jit.script
def model_gompertz(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.exp(-params[1] * torch.exp(-params[2] * (t / T)))


# Model 40: Log-Logistic
@torch.jit.script
def model_log_logistic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] / (1 + ((t / T) / params[1]) ** params[2])


# Model 41: Bass Diffusion
@torch.jit.script
def model_bass(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return params[0] * (
        (1 - torch.exp(-params[1] * (t / T)))
        / (1 + params[2] * torch.exp(-params[1] * (t / T)))
    )


# Model 42: Weibull CDF
@torch.jit.script
def model_weibull(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (1 - torch.exp(-(((t / T) / params[1]) ** params[2])))


# Model 43: Gamma CDF Approximation
@torch.jit.script
def model_gamma(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return params[0] * (t / T) ** params[1] * torch.exp(-params[2] * (t / T))


# Model 44: Custom Polynomial Degree 4
@torch.jit.script
def model_poly4(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return (
        params[0] * (t / T) ** 4
        + params[1] * (t / T) ** 3
        + params[2] * (t / T) ** 2
        + (1 - params[0] - params[1] - params[2]) * (t / T)
    )


# Model 45: Sinusoidal with Offset
@torch.jit.script
def model_sine_offset(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.sin(params[1] * (t / T) * 3.141592653589793) + params[
        2
    ] * (t / T)


# Model 46: Cosinusoidal with Offset
@torch.jit.script
def model_cosine_offset(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (
        1 - torch.cos(params[1] * (t / T) * 3.141592653589793)
    ) + params[2] * (t / T)


# Model 47: Double Sigmoid
@torch.jit.script
def model_double_sigmoid(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (1 / (1 + torch.exp(-params[0] * ((t / T) - params[1])))) * (
        1 / (1 + torch.exp(params[2] * ((t / T) - params[3])))
    )


# Model 48: Exponential Decay
@torch.jit.script
def model_exp_decay(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.exp(-params[1] * (1 - (t / T)))


# Model 49: Quadratic Decay
@torch.jit.script
def model_quad_decay(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (1 - (t / T) ** 2)


# Model 50: Cubic Decay
@torch.jit.script
def model_cubic_decay(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (1 - (t / T) ** 3)


# Model 51: Sine Squared
@torch.jit.script
def model_sine_squared(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (torch.sin(params[1] * (t / T) * 3.141592653589793)) ** 2


# Model 52: Cosine Squared
@torch.jit.script
def model_cosine_squared(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (1 - torch.cos(params[1] * (t / T) * 3.141592653589793)) ** 2


# Model 53: Polynomial-Exponential Combination
@torch.jit.script
def model_poly_exp(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) ** 2 * torch.exp(-params[1] * (t / T))


# Model 54: Logarithmic plus Linear
@torch.jit.script
def model_log_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.log(1 + params[1] * (t / T)) + params[2] * (t / T)


# Model 55: Inverse Squared
@torch.jit.script
def model_inverse_squared(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) / (params[1] + (t / T) ** 2)


# Model 56: Piecewise Quadratic
@torch.jit.script
def model_piecewise_quadratic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    if ratio < params[0]:
        return params[1] * (ratio) ** 2
    else:
        return params[1] * (params[0]) ** 2 + params[2] * (ratio - params[0]) ** 2


# Model 57: Sine with Phase Shift
@torch.jit.script
def model_sine_phase(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.sin(params[1] * (t / T) * 3.141592653589793 + params[2])


# Model 58: Cosine with Phase Shift
@torch.jit.script
def model_cosine_phase(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (
        1 - torch.cos(params[1] * (t / T) * 3.141592653589793 + params[2])
    )


# Model 59: Exponential Saturation
@torch.jit.script
def model_exp_saturation(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    if params[0] != 0:
        return (torch.exp(params[0] * ratio) - 1) / (torch.exp(params[0]) - 1)
    else:
        return ratio


# Model 60: Power Function with Linear Offset
@torch.jit.script
def model_power_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (t / T) ** params[0] + params[1] * (t / T)


# Model 61: Logistic plus Quadratic
@torch.jit.script
def model_logistic_quadratic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0] / (1 + torch.exp(-params[1] * ((t / T) - params[2])))
        + params[3] * (t / T) ** 2
    )


# Model 62: Tanh plus Linear
@torch.jit.script
def model_tanh_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (torch.tanh(params[0] * ((t / T) - params[1])) + 1) + params[2] * (
        t / T
    )


# Model 63: Reciprocal Linear
@torch.jit.script
def model_reciprocal_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 1 / (params[0] + params[1] * (1 - t / T))


# Model 64: Reciprocal Quadratic
@torch.jit.script
def model_reciprocal_quadratic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 1 / (params[0] + params[1] * (1 - t / T) ** 2)


# Model 65: Logarithm with Saturation
@torch.jit.script
def model_log_saturate(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    val = params[0] * torch.log(1 + params[1] * (t / T))
    return torch.clamp(val, max=1.0)


# Model 66: Arctan with Offset
@torch.jit.script
def model_arctan_offset(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (2 / 3.141592653589793) * torch.atan(params[0] * (t / T) + params[1])


# Model 67: Hyperbolic Secant
@torch.jit.script
def model_sech(params: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    return params[0] / torch.cosh(params[1] * (t / T))


# Model 68: Damped Sine
@torch.jit.script
def model_damped_sine(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.sin(params[1] * (t / T)) * torch.exp(-params[2] * (t / T))


# Model 69: Sum of Sine and Cosine
@torch.jit.script
def model_sine_cosine(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.sin(params[1] * (t / T)) + params[2] * torch.cos(
        params[3] * (t / T)
    )


# Model 70: Cubic Polynomial with Bias
@torch.jit.script
def model_cubic_bias(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0] * (t / T) ** 3
        + params[1] * (t / T) ** 2
        + params[2] * (t / T)
        + params[3]
    )


# Model 71: Quartic Polynomial
@torch.jit.script
def model_quartic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0] * (t / T) ** 4
        + params[1] * (t / T) ** 3
        + params[2] * (t / T) ** 2
        + params[3] * (t / T)
        + params[4]
    )


# Model 72: Damped Sinusoid plus Linear
@torch.jit.script
def model_damped_sinusoid_linear(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.sin(params[1] * (t / T)) * torch.exp(
        -params[2] * (t / T)
    ) + params[3] * (t / T)


# Model 73: Log-Modified Exponential
@torch.jit.script
def model_log_mod_exp(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (1 - torch.exp(-params[0] * (t / T))) * torch.log(1 + params[1] * (t / T))


# Model 74: Sum of Two Exponentials
@torch.jit.script
def model_two_exp(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (1 - torch.exp(-params[1] * (t / T))) + params[2] * (
        1 - torch.exp(-params[3] * (t / T))
    )


# Model 75: Weighted Power Average
@torch.jit.script
def model_weighted_power(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    num = params[0] * (t / T) ** params[1] + params[2] * (t / T) ** params[3]
    s = params[0] + params[2]
    denom = s if s != 0 else torch.tensor(1e-6)
    return num / denom


# Model 76: Modified Logistic (Shifted Down)
@torch.jit.script
def model_modified_logistic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 1 / (1 + torch.exp(-params[0] * ((t / T) - params[1]))) - 0.5


# Model 77: Damped Tanh
@torch.jit.script
def model_damped_tanh(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (torch.tanh(params[0] * (t / T)) + 1) * torch.exp(-params[1] * (t / T))


# Model 78: Shifted Logistic with Linear Decay
@torch.jit.script
def model_shifted_logistic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] / (1 + torch.exp(-params[1] * ((t / T) - params[2]))) + params[
        3
    ] * (1 - t / T)


# Model 79: Polynomial Blend
@torch.jit.script
def model_poly_blend(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    return (ratio ** params[0]) / ((ratio ** params[0]) + ((1 - ratio) ** params[1]))


# Model 80: Generalized Mean Blend
@torch.jit.script
def model_generalized_mean(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    num = params[0] * (ratio) ** params[1] + params[2] * ((1 - ratio) ** params[3])
    s = params[0] + params[2]
    denom = s if s != 0 else torch.tensor(1e-6)
    return (num / denom) ** (1 / params[4])


# Model 81: Oscillatory Decay
@torch.jit.script
def model_oscillatory_decay(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0]
        * (t / T)
        * (1 + params[1] * torch.cos(params[2] * (t / T)))
        * torch.exp(-params[3] * (t / T))
    )


# Model 82: Hyperbolic Function
@torch.jit.script
def model_hyperbolic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) / (params[1] + (t / T))


# Model 83: Polynomial plus Logarithm
@torch.jit.script
def model_poly_log(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) + params[1] * torch.log(1 + (t / T))


# Model 84: Square plus Square Root
@torch.jit.script
def model_square_sqrt(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) ** 2 + params[1] * torch.sqrt(t / T)


# Model 85: Weighted Sum of Exponentials
@torch.jit.script
def model_weighted_exp(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * torch.exp(-params[1] * (1 - (t / T))) + params[2] * torch.exp(
        -params[3] * (t / T)
    )


# Model 86: Rational Function
@torch.jit.script
def model_rational(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    denom = params[2] * ratio + params[3]
    if denom == 0:
        denom = torch.tensor(1e-6)
    return (params[0] * ratio + params[1]) / denom


# Model 87: Quadratic plus Log Correction
@torch.jit.script
def model_quad_log(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) ** 2 + params[1] * torch.log(1 + params[2] * (t / T))


# Model 88: Piecewise Combination
@torch.jit.script
def model_piecewise_combo(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    ratio = t / T
    if ratio < params[0]:
        return params[1] * (ratio) ** 2
    else:
        return params[2] * ratio + params[3]


# Model 89: Cubic with Saturation
@torch.jit.script
def model_cubic_sat(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    val = (
        params[0] * (t / T) ** 3
        + params[1] * (t / T) ** 2
        + params[2] * (t / T)
        + params[3]
    )
    return torch.clamp(val, max=1.0)


# Model 90: Sinusoid plus Polynomial
@torch.jit.script
def model_sinusoid_poly(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return params[0] * (t / T) + params[1] * torch.sin(
        params[2] * (t / T) * 3.141592653589793
    )


# Model 91: Sum of Two Logistic Functions
@torch.jit.script
def model_sum_logistics(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (params[0] / (1 + torch.exp(-params[1] * ((t / T) - params[2])))) + (
        params[3] / (1 + torch.exp(-params[4] * ((t / T) - params[5])))
    )


# Model 92: Tanh plus Linear (Variant)
@torch.jit.script
def model_tanh_linear2(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (torch.tanh(params[0] * ((t / T) - params[1])) + 1) + params[2] * (
        t / T
    )


# Model 93: Arctan Blend
@torch.jit.script
def model_arctan_blend(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (2 / 3.141592653589793) * torch.atan(
        params[0] * (t / T) + params[1] * (t / T) ** 2
    )


# Model 94: Exponential Blend
@torch.jit.script
def model_exponential_blend(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 1 - torch.exp(-params[0] * (t / T) ** params[1])


# Model 95: Sine Blend Squared
@torch.jit.script
def model_sine_blend_squared(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (torch.sin(params[0] * (t / T) * 3.141592653589793)) ** params[1]


# Model 96: Cosine Blend Squared
@torch.jit.script
def model_cosine_blend_squared(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return 1 - (torch.cos(params[0] * (t / T) * 3.141592653589793)) ** params[1]


# Model 97: Oscillatory Logistic
@torch.jit.script
def model_oscillatory_logistic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0]
        / (1 + torch.exp(-params[1] * ((t / T) - params[2])))
        * (1 + params[3] * torch.sin(params[4] * (t / T)))
    )


# Model 98: Damped Harmonic
@torch.jit.script
def model_damped_harmonic(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0]
        * (t / T)
        * torch.exp(-params[1] * (t / T))
        * torch.cos(params[2] * (t / T))
    )


# Model 99: Sigmoid Squared
@torch.jit.script
def model_sigmoid_squared(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (1 / (1 + torch.exp(-params[0] * ((t / T) - params[1])))) ** 2


# Model 100: Composite Function
@torch.jit.script
def model_composite(
    params: torch.Tensor, t: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    return (
        params[0] * (1 - torch.exp(-params[1] * (t / T)))
        + params[2] * (t / T) ** params[3]
    )


#############################################
# Dictionary of Candidate Models (1-100)    #
#############################################

extra_models = {
    "Model 1: Linear": {
        "func": model_linear,
        "ranges": [(0.0, 1.0)],
        "param_names": ["p0"],
    },
    "Model 13: New Model 4: Log-Linear": {
        "func": new_model4,
        "ranges": [(0.0, 1.0), (0.0, 100.0), (0.0, 1.0)],
        "param_names": ["A", "k", "B"],
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
        "ranges": [(0.0, 1.0)] * 3,
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
}

###############################################
# PyTorch Loss Function for a Candidate Model #
###############################################from typing import Tuple
########################################
# Vectorized Loss Module (TorchScript)
########################################


class VectorizedLossModule(torch.jit.ScriptModule):
    def __init__(self, model_func):
        super(VectorizedLossModule, self).__init__()
        # model_func must be TorchScript compiled.
        self.model_func = model_func

    @torch.jit.script_method
    def forward(
        self,
        params: torch.Tensor,
        transmissions: torch.Tensor,
        backgrounds: torch.Tensor,
        foregrounds: torch.Tensor,
        measured: torch.Tensor,
        layer_thickness: float,
        num_layers: int,
    ) -> torch.Tensor:
        """
        Compute the average L1 loss over all samples and layers using sequential compositing.
        The candidate model's output (clamped to [0,1]) is directly used as the opacity (opac)
        for that layer, in a manner matching the production code.
        """
        N = transmissions.size(0)
        # Start with the background color as the initial composite. # [N, 3]
        # "Remaining" fraction (i.e. the unprinted part) starts at 1 for each sample.
        total_loss = torch.tensor(0.0, dtype=backgrounds.dtype)

        t_tensor = torch.full((N,), 0.04, dtype=transmissions.dtype)
        offset = params[0]
        model_params = params[1:]

        comp = torch.zeros(N, 3, dtype=torch.float32)
        remaining = torch.ones(N, dtype=torch.float32)
        background = backgrounds

        for i in range(num_layers):
            TD_i = transmissions
            color_i = foregrounds
            opac = offset + self.model_func(model_params, t_tensor, TD_i)
            opac = torch.clamp(opac, 0.0, 1.0)
            comp = comp + ((remaining * opac).unsqueeze(-1) * color_i)
            remaining = remaining * (1 - opac)

            meas = measured[:, i, :]
            comp1 = comp + remaining.unsqueeze(-1) * background
            comp1 = comp1.reshape(N, 3)
            # Accumulate per-sample L1 error.
            error = torch.sum(torch.abs(comp1 - meas))
            total_loss = total_loss + torch.sum(error)
        total_loss = total_loss / (N * num_layers)
        return total_loss


########################################
# Data Preprocessing
########################################


def hex_to_rgb_torch(hex_str: str) -> torch.Tensor:
    """
    Convert hex string (e.g. "#d9e0e9") to a tensor of shape (3,) with values 0-255.
    """
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return torch.tensor([r, g, b], dtype=torch.float32)


def preprocess_data(
    df: pd.DataFrame, num_layers: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert the CSV DataFrame into PyTorch tensors.
    Returns:
      - transmissions: [N] tensor
      - backgrounds: [N, 3] tensor
      - foregrounds: [N, 3] tensor
      - measured: [N, num_layers, 3] tensor
    """
    N = len(df)
    transmissions = torch.empty(N, dtype=torch.float32)
    backgrounds = torch.empty((N, 3), dtype=torch.float32)
    foregrounds = torch.empty((N, 3), dtype=torch.float32)
    measured = torch.empty((N, num_layers, 3), dtype=torch.float32)
    for i, row in df.iterrows():
        transmissions[i] = float(row["Transmission Distance"])
        backgrounds[i] = hex_to_rgb_torch(row["Background Material"])
        foregrounds[i] = hex_to_rgb_torch(row["Layer Material"])
        for layer in range(1, num_layers + 1):
            measured[i, layer - 1] = hex_to_rgb_torch(row[f"Layer {layer}"])
    return transmissions, backgrounds, foregrounds, measured


########################################
# Optimization Routine for One Model
########################################


def optimize_model(
    model_name: str,
    model_data: dict,
    transmissions: torch.Tensor,
    backgrounds: torch.Tensor,
    foregrounds: torch.Tensor,
    measured: torch.Tensor,
    layer_thickness: float,
    num_layers: int,
    num_iterations: int = 10000,
    lr: float = 0.01,
):
    """
    Optimize the parameters for a given candidate model using gradient descent.
    Three manual offset parameters (for B, G, R) are prepended to the model parameters.
    """
    # Create the parameter names list.
    param_names = [
        "manual offset B",
        "manual offset G",
        "manual offset R",
    ] + model_data["param_names"]
    # Initialize manual offsets to 0.0 and model parameters to midpoints of their ranges.
    init_params = [0.0]
    for rng in model_data["ranges"]:
        low, high = rng
        init_params.append(
            random.randrange(round(low * 10000), round(high * 10000)) / 10000
        )
    params = torch.tensor(init_params, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([params], lr=lr)
    loss_module = VectorizedLossModule(model_data["func"])
    best_loss = float("inf")
    best_params = None
    losses = []
    no_improvement_count = 0
    it = 0
    tbar = tqdm()
    while True:
        try:
            optimizer.zero_grad()
            loss = loss_module(
                params,
                transmissions,
                backgrounds,
                foregrounds,
                measured,
                layer_thickness,
                num_layers,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = params.detach().clone()
                no_improvement_count = 0
                tbar.set_description(f"Loss: {best_loss:.4f}")
            else:
                no_improvement_count += 1
            if no_improvement_count >= 10000:
                break
            # break if loss decrease is less than 1e-6 in last 100 steps
            if (
                len(losses) > 1000
                and np.mean(losses[-1000:-100]) - np.min(losses[-100:]) < 1e-3
            ):
                break
            it += 1
            tbar.update(1)
        except Exception:
            traceback.print_exc()
            print(f"Error in iteration {it}")
            break
    tbar.close()
    if best_params is not None:
        print(
            f"Final loss for {model_name}: {best_loss:.4f}, Params: {best_params.numpy()}"
        )
    return best_loss, best_params, losses, param_names


def predict_colors(model_func, best_params, T, bg, fg, layer_thickness, num_layers):
    """
    For a single sample, compute the predicted composite colors for each layer,
    using the same sequential blending logic as the forward function.

    - The composite is initialized as zeros.
    - The candidate model's output (after clamping) is used as the opacity.
    - A constant t_val (0.04) is used, matching the forward function.

    Returns:
      A numpy array of shape (num_layers, 3) with values in 0-255.
    """
    # Initialize composite as zeros (no printed contribution yet)
    composite = np.zeros_like(bg)  # shape (3,)
    remaining = 1.0
    predicted = []

    # Use the same constant t_val as in the forward function.
    t_val = torch.tensor(0.04, dtype=torch.float32)
    T_tensor = torch.tensor(T, dtype=torch.float32)

    offset = best_params[0]
    best_params = best_params[1:]

    for i in range(num_layers):
        # Get candidate model output and interpret as opacity for this layer.
        opac_tensor = offset + model_func(best_params, t_val, T_tensor)
        opac = np.clip(opac_tensor.detach().cpu().numpy(), 0.0, 1.0)
        # Update composite: add contribution from this layer.
        composite = composite + (remaining * opac) * fg
        # Update remaining unprinted fraction.
        remaining = remaining * (1 - opac)
        # Form full composite with background.
        comp_with_bg = composite + remaining * bg
        predicted.append(comp_with_bg.copy())
    return np.array(predicted)


def rgb_to_hex(rgb_array):
    """Convert an array of three values (0-255) to hex string."""
    return "#{:02X}{:02X}{:02X}".format(
        int(rgb_array[0]), int(rgb_array[1]), int(rgb_array[2])
    )


########################################
# Plot Measured vs Predicted using Plotly
########################################


def plot_measured_vs_predicted(
    best_results, df, num_layers, layer_thickness, sample_size=10
):
    """
    best_results: list of best result dictionaries (for the 3 best models)
    For sample_size random rows from the CSV, creates a grid with 1 row per sample and one column per model.
    Within each subplot, the upper half (y from 0.5 to 1) shows the measured colors and
    the lower half (y from 0 to 0.5) shows the predicted colors.
    """
    # Randomly select sample_size rows from df.
    sample_df = df.sample(sample_size, random_state=42)
    num_models = len(best_results)

    # Create subplots: one row per sample, one column per model.
    fig = make_subplots(
        rows=sample_size,
        cols=num_models,
        horizontal_spacing=0.05,
        vertical_spacing=0.02,
        subplot_titles=[res["model"] for res in best_results],
    )

    for sample_idx, (_, row) in enumerate(sample_df.iterrows()):
        T = float(row["Transmission Distance"])
        bg = hex_to_rgb_torch(row["Background Material"]).numpy()
        fg = hex_to_rgb_torch(row["Layer Material"]).numpy()
        measured_colors = [
            hex_to_rgb_torch(row[f"Layer {layer}"]).numpy()
            for layer in range(1, num_layers + 1)
        ]

        for model_idx, res in enumerate(best_results):
            # Compute predicted colors for this sample using the candidate model.
            pred_colors = predict_colors(
                res["func"], res["best_params"], T, bg, fg, layer_thickness, num_layers
            )

            for layer in range(num_layers):
                x0 = layer
                x1 = layer + 1

                # Add measured color rectangle in the upper half of the cell.
                color_meas = rgb_to_hex(measured_colors[layer])
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=0.5,
                    y1=1,
                    fillcolor=color_meas,
                    line=dict(width=0),
                    row=sample_idx + 1,
                    col=model_idx + 1,
                )

                # Add predicted color rectangle in the lower half of the cell.
                color_pred = rgb_to_hex(pred_colors[layer])
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=0.5,
                    fillcolor=color_pred,
                    line=dict(width=0),
                    row=sample_idx + 1,
                    col=model_idx + 1,
                )

            # Update axes for this cell.
            fig.update_xaxes(
                range=[0, num_layers],
                row=sample_idx + 1,
                col=model_idx + 1,
                showticklabels=False,
            )
            fig.update_yaxes(
                range=[0, 1],
                row=sample_idx + 1,
                col=model_idx + 1,
                showticklabels=False,
            )

    # Increase overall figure height so that each row is larger.
    fig.update_layout(
        height=sample_size * 150,
        width=num_models * 400,
        title_text="Measured (Top) vs. Predicted (Bottom) Colors for 10 Random Samples",
    )
    fig.show()


########################################
# Main Function: Run All Models & Plot
########################################


def main():
    # Load CSV data (adjust the filename as needed)
    df = pd.read_csv("printed_colors.csv")
    num_layers = 16
    layer_thickness = 0.04  # mm per layer
    transmissions, backgrounds, foregrounds, measured = preprocess_data(df, num_layers)

    # extra_models = {
    #     # "Model 43: Gamma CDF": {
    #     #     "func": model_gamma,
    #     #     "ranges": [(0.0, 1.0), (0.1, 5.0), (0.1, 5.0)],
    #     #     "param_names": ["p0", "p1", "p2"],
    #     # },
    #     "Model 13: New Model 4: Log-Linear": {
    #         "func": new_model4,
    #         "ranges": [(0.0, 1.0), (0.0, 200.0), (0.0, 1.0)],
    #         "param_names": ["A", "k", "B"],
    #     },
    # }
    extra_models = {
        # "Model 28: Poly Degree 2": {
        #     "func": model_poly2,
        #     "ranges": [(0.0, 1.0)],
        #     "param_names": ["p0"],
        # },
        "Model 29: Poly Degree 3": {
            "func": model_poly3,
            "ranges": [(0.0, 1.0), (0.0, 1.0)],
            "param_names": ["p0", "p1"],
        },
    }

    results = []
    # Loop over each candidate model in the dictionary.
    for model_name, model_data in extra_models.items():
        print(f"\n--- Optimizing {model_name} ---")
        best_loss, best_params, losses, param_names = optimize_model(
            model_name,
            model_data,
            transmissions,
            backgrounds,
            foregrounds,
            measured,
            layer_thickness,
            num_layers,
            num_iterations=100000,
            lr=0.01,
        )
        results.append(
            {
                "model": model_name,
                "best_loss": best_loss,
                "best_params": best_params,
                "param_names": param_names,
                "losses": losses,
                "func": model_data["func"],
            }
        )

    # Sort results by loss (lowest loss first)
    results.sort(key=lambda x: x["best_loss"])
    print("\n=== Best Models ===")
    for res in results:
        if res["best_params"] is not None:
            print(
                f"{res['model']}: Loss = {res['best_loss']:.4f}, Params = {res['best_params'].numpy()}"
            )

        # # Plot loss curves for each model (using matplotlib)
        # plt.figure(figsize=(10, 6))
        # for res in results:
        #     plt.plot(res["losses"], label=res["model"])
        # plt.xlabel("Iteration")
        # plt.ylabel("Loss")
        # plt.title("Loss Curves for Candidate Models")
        # plt.legend()
        # plt.show()

    # Now plot measured vs predicted for 10 random CSV rows using Plotly,
    # but only for the top 3 best models.
    best_three = results[:3]
    plot_measured_vs_predicted(best_three, df, 16, layer_thickness, sample_size=50)


if __name__ == "__main__":
    main()
