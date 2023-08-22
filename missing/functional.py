import tensorflow as tf
import numpy as np

from scipy.special import log_softmax, softmax, expit, logit
from scipy.optimize import minimize

from sklearn.metrics import log_loss


__all__ = [
    "sparsity_normalize",
    "delta_t",
]


def sparsity_normalize(x, mask, constant=1., axis=-2):
    denorm = tf.math.count_nonzero(mask, axis=axis, keepdims=True, dtype=x.dtype)
    denorm = tf.where(tf.equal(denorm, 0.), 1., denorm)
    x_norm = constant * x / denorm
    return x_norm


def delta_t(times, measurements, measurement_indicators):
    """Add a delta t tensor which contains time since previous measurement.

    Args:
        times: The times of the measurements (tp,)
        measurements: The measured values (tp, measure_dim)
        measurement_indicators: Indicators if the variables was measured or not (tp, measure_dim)

    Returns:
        delta t tensor of shape (tp, measure_dim)
    """

    scattered_times = times * tf.cast(measurement_indicators, tf.float32)
    dt_array = tf.TensorArray(tf.float32, tf.shape(measurement_indicators)[0])
    # First observation has dt = 0
    first_dt = tf.zeros(tf.shape(measurement_indicators)[1:])
    dt_array = dt_array.write(0, first_dt)

    def compute_dt_timestep(i, last_dt, dt_array):
        last_dt = tf.where(
            measurement_indicators[i-1],
            tf.fill(tf.shape(last_dt), tf.squeeze(times[i] - times[i-1])),
            times[i] - times[i-1] + last_dt
        )
        dt_array = dt_array.write(i, last_dt)
        return i+1, last_dt, dt_array

    n_observations = tf.shape(scattered_times)[0]
    _, last_dt, dt_array = tf.while_loop(
        lambda i, a, b: i < n_observations,
        compute_dt_timestep,
        loop_vars=[tf.constant(1), first_dt, dt_array],
    )
    dt_tensor = dt_array.stack()
    dt_tensor.set_shape(measurements.get_shape())
    return dt_tensor


def calibrate_prediction(confidences, true_labels, target_confidences):
    logits = logit(confidences.astype(np.float64))

    opt_temp = minimize(
        fun=lambda t: log_loss(true_labels, expit(logits / t)),
        x0=1.0, method="nelder-mead", options={"xtol": 1e-3}
    ).x[0]

    calibrated_confidences = expit(logit(target_confidences) / opt_temp)

    return calibrated_confidences, opt_temp
