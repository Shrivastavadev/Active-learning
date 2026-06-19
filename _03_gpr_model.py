"""
Module 3 — Heteroscedastic GP Model
=====================================
Reusable functions for building, training, and predicting with
heteroscedastic SVGP models. Imported by Module 4.

No __main__ block — this file is a library, not a script.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — all needed, this file is self-contained
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from pathlib import Path
from scipy.spatial.distance import pdist

# Suppress TensorFlow and GPflow verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

INPUT_COLS  = ['Au [at.%]', 'Ir [at.%]']
OUTPUT_COLS = ['log_k0', 'alpha']


# ─────────────────────────────────────────────────────────────────────────────
# INPUT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

class Normaliser:
    """
    Normalises GP inputs to zero mean and unit variance.

    Fit on seed set only — simulates real active learning scenario
    where we don't know the full data distribution upfront.

    Usage:
        normaliser = Normaliser()
        X_train_norm = normaliser.fit_transform(X_train)
        X_test_norm  = normaliser.transform(X_test)
    """
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray) -> 'Normaliser':
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8   # epsilon prevents division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# D_MAX COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_d_max(seed_df: pd.DataFrame, pool_df: pd.DataFrame) -> float:
    """
    Compute the maximum Euclidean distance in composition space
    across all points (seed + pool combined).

    Used to normalise within-library movement costs to [0, 1]
    so they always stay below the switch-library penalty (1 + gamma).

    Parameters
    ----------
    seed_df : seed set DataFrame
    pool_df : pool set DataFrame

    Returns
    -------
    d_max : float
    """
    all_points = pd.concat([seed_df, pool_df])[INPUT_COLS].values
    d_max = pdist(all_points).max()
    print(f"  d_max (max composition distance): {d_max:.4f} at.% units")
    return float(d_max)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_heteroscedastic_svgp(
    X_init    : np.ndarray,
    n_inducing: int = 20,
) -> gpflow.models.SVGP:
    """
    Build a heteroscedastic SVGP with two latent GPs:
        GP1 (signal kernel) → models mean f(x)
        GP2 (noise kernel)  → models log noise variance log σ²(x)

    Likelihood: y ~ Normal(GP1(x), exp(GP2(x)))

    Parameters
    ----------
    X_init     : training inputs, used to initialise inducing point locations
    n_inducing : number of inducing points (capped at len(X_init))

    Returns
    -------
    Untrained gpflow.models.SVGP
    """
    # Two independent Matern32 kernels — one per latent GP
    signal_kernel = gpflow.kernels.Matern32(
        active_dims=[0, 1],
        lengthscales=[1.0, 1.0],
    )
    noise_kernel = gpflow.kernels.Matern32(
        active_dims=[0, 1],
        lengthscales=[1.0, 1.0],
    )
    kernel = gpflow.kernels.SeparateIndependent([signal_kernel, noise_kernel])

    # Heteroscedastic likelihood — combines both latent GPs
    # GP1 output → mean, GP2 output → log scale (exponentiated to ensure > 0)
    likelihood = gpflow.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,
        scale_transform=tfp.bijectors.Exp(),
    )

    # Initialise inducing points evenly spread across training data
    n_inducing = min(n_inducing, len(X_init))
    indices    = np.linspace(0, len(X_init) - 1, n_inducing, dtype=int)
    Z_init     = X_init[indices].copy()

    # Each latent GP gets its own inducing points
    inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables([
        gpflow.inducing_variables.InducingPoints(Z_init.copy()),
        gpflow.inducing_variables.InducingPoints(Z_init.copy()),
    ])

    model = gpflow.models.SVGP(
        kernel            = kernel,
        likelihood        = likelihood,
        inducing_variable = inducing_variable,
        num_latent_gps    = likelihood.latent_dim,  # = 2
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model    : gpflow.models.SVGP,
    X_train  : np.ndarray,
    Y_train  : np.ndarray,
    max_iter : int  = 500,
    verbose  : bool = False,
) -> dict:
    """
    Train SVGP using Scipy L-BFGS-B optimiser.
    Maximises the Evidence Lower BOund (ELBO).

    Parameters
    ----------
    model    : built (untrained) SVGP
    X_train  : normalised inputs,  shape (n, 2)
    Y_train  : outputs,            shape (n, 1)
    max_iter : max optimisation iterations
    verbose  : print ELBO before and after

    Returns
    -------
    dict with elbo_initial and elbo_final
    """
    data = (
        tf.constant(X_train, dtype=tf.float64),
        tf.constant(Y_train, dtype=tf.float64),
    )

    elbo_initial = model.elbo(data).numpy()

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss_closure(data),
        model.trainable_variables,
        options={'maxiter': max_iter, 'disp': False},
    )

    elbo_final = model.elbo(data).numpy()

    if verbose:
        print(f"     ELBO: {elbo_initial:.4f} → {elbo_final:.4f}  "
              f"(Δ={elbo_final - elbo_initial:.4f})")

    return {'elbo_initial': elbo_initial, 'elbo_final': elbo_final}


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model : gpflow.models.SVGP,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return posterior predictive mean and variance for test points.

    Variance includes both:
        - epistemic uncertainty  (GP uncertainty about the function)
        - aleatoric uncertainty  (learned noise from GP2)

    Parameters
    ----------
    model  : trained SVGP
    X_test : normalised test inputs, shape (n, 2)

    Returns
    -------
    mu  : predicted means,     shape (n,)
    var : predicted variances, shape (n,)
    """
    X_tf    = tf.constant(X_test, dtype=tf.float64)
    mu, var = model.predict_y(X_tf)
    return mu.numpy().flatten(), var.numpy().flatten()


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_predict(
    X_train    : np.ndarray,
    Y_train    : np.ndarray,
    X_test     : np.ndarray,
    output_name: str,
    n_inducing : int  = 20,
    max_iter   : int  = 500,
    verbose    : bool = True,
) -> tuple[gpflow.models.SVGP, np.ndarray, np.ndarray]:
    """
    Build, train, and predict for a single output in one call.

    Parameters
    ----------
    X_train     : training inputs,  shape (n_train, 2)
    Y_train     : training outputs, shape (n_train, 1)
    X_test      : test inputs,      shape (n_test,  2)
    output_name : label for print output e.g. 'log_k0'
    n_inducing  : number of inducing points
    max_iter    : optimisation iterations
    verbose     : print progress

    Returns
    -------
    model : trained SVGP
    mu    : predicted means,     shape (n_test,)
    var   : predicted variances, shape (n_test,)
    """
    if verbose:
        print(f"\n  ── {output_name} ──────────────────────────────────")
        print(f"     Training points : {len(X_train)}")
        print(f"     Inducing points : {min(n_inducing, len(X_train))}")

    model   = build_heteroscedastic_svgp(X_train, n_inducing=n_inducing)
    history = train_model(model, X_train, Y_train, max_iter=max_iter, verbose=verbose)
    mu, var = predict(model, X_test)

    return model, mu, var