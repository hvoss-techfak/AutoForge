

def sample_gumbel(shape, key, eps=1e-20):
    """
    Sample from a Gumbel distribution.

    Args:
        shape (tuple): The shape of the output array.
        key (jax.random.PRNGKey): The random key for JAX operations.
        eps (float, optional): A small value to avoid numerical issues. Defaults to 1e-20.

    Returns:
        jnp.ndarray: Samples from the Gumbel distribution.
    """
    U = random.uniform(key, shape=shape, minval=0.0, maxval=1.0)
    return -np.log(-np.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, key):
    """
    Sample from the Gumbel-Softmax distribution.

    Args:
        logits (jnp.ndarray): The input logits for the Gumbel-Softmax.
        temperature (float): The temperature parameter for the Gumbel-Softmax.
        key (jax.random.PRNGKey): The random key for JAX operations.

    Returns:
        jnp.ndarray: The Gumbel-Softmax samples.
    """
    g = sample_gumbel(logits.shape, key)
    return jax.nn.softmax((logits + g) / temperature)


def gumbel_softmax(logits, temperature, key, hard=False):
    """
    Compute the Gumbel-Softmax.

    Args:
        logits (jnp.ndarray): The input logits for the Gumbel-Softmax.
        temperature (float): The temperature parameter for the Gumbel-Softmax.
        key (jax.random.PRNGKey): The random key for JAX operations.
        hard (bool, optional): If True, return hard one-hot encoded samples. Defaults to False.

    Returns:
        jnp.ndarray: The Gumbel-Softmax samples.
    """
    y = gumbel_softmax_sample(logits, temperature, key)
    if hard:
        y_hard = jax.nn.one_hot(jnp.argmax(y, axis=-1), logits.shape[-1])
        y = y_hard + jax.lax.stop_gradient(y - y_hard)
    return y


def srgb_to_linear_lab(rgb):
    """
    Convert sRGB (range [0,1]) to linear RGB.

    Args:
        rgb (jnp.ndarray): A JAX array of shape (..., 3) representing the sRGB values.

    Returns:
        jnp.ndarray: A JAX array of shape (..., 3) representing the linear RGB values.
    """
    return jnp.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def linear_to_xyz(rgb_linear):
    """
    Convert linear RGB to XYZ using sRGB D65.

    Args:
        rgb_linear (jnp.ndarray): A JAX array of shape (..., 3) representing the linear RGB values.

    Returns:
        jnp.ndarray: A JAX array of shape (..., 3) representing the XYZ values.
    """
    R = rgb_linear[..., 0]
    G = rgb_linear[..., 1]
    B = rgb_linear[..., 2]
    X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
    Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
    Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B
    return jnp.stack([X, Y, Z], axis=-1)


def xyz_to_lab(xyz):
    """
    Convert XYZ to CIELAB. Assumes D65 reference white.

    Args:
        xyz (jnp.ndarray): A JAX array of shape (..., 3) representing the XYZ values.

    Returns:
        jnp.ndarray: A JAX array of shape (..., 3) representing the CIELAB values.
    """
    # Reference white for D65:
    xyz_ref = jnp.array([0.95047, 1.0, 1.08883])
    xyz = xyz / xyz_ref
    delta = 6/29
    f = jnp.where(xyz > delta**3, xyz ** (1/3), (xyz / (3 * delta**2)) + (4/29))
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return jnp.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb):
    """
    Convert an sRGB image (values in [0,1]) to CIELAB.

    Args:
        rgb (jnp.ndarray): A JAX array of shape (..., 3) representing the sRGB values.

    Returns:
        jnp.ndarray: A JAX array of shape (..., 3) representing the CIELAB values.
    """
    rgb_linear = srgb_to_linear_lab(rgb)
    xyz = linear_to_xyz(rgb_linear)
    lab = xyz_to_lab(xyz)
    return lab


def adaptive_round(x, tau, high_tau=1.0, low_tau=0.01, temp=0.1):
    """
    Compute a soft (adaptive) rounding of x.

    When tau is high (>= high_tau) returns x (i.e. no rounding).
    When tau is low (<= low_tau) returns round(x).
    In between, linearly interpolates between x and round(x).

    Args:
        x (jnp.ndarray): The input array to be rounded.
        tau (float): The temperature parameter controlling the degree of rounding.
        high_tau (float, optional): The high threshold for tau. Defaults to 1.0.
        low_tau (float, optional): The low threshold for tau. Defaults to 0.01.
        temp (float, optional): A temperature parameter for interpolation. Defaults to 0.1.

    Returns:
        jnp.ndarray: The adaptively rounded array.
    """
    beta = jnp.clip((high_tau - tau) / (high_tau - low_tau), 0.0, 1.0)
    return (1 - beta) * x + beta * jnp.round(x)