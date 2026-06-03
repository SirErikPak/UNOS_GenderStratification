import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.box import ROUNDED
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np


# -------------------------------------------------------------------------------------------
# : Sample from a Fitted Univariate GMM Using NumPy's Generator API
# -------------------------------------------------------------------------------------------
def gmm_sample(model, n, random_state=None):
    """
    Draw n samples from a fitted univariate GaussianMixture model using NumPy's
    modern Generator API.

    Parameters
    ----------
    model : GaussianMixture
        A fitted sklearn GaussianMixture model (must be univariate).
    n : int
        Number of samples to generate.
    random_state : int, np.random.Generator, or None
        Seed or Generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of sampled values drawn from the mixture distribution.
    """

    # Initialize RNG (supports int seed, Generator, or None)
    rng = np.random.default_rng(random_state)

    # Choose mixture components according to learned weights
    comp_idx = rng.choice(model.n_components, size=n, p=model.weights_)

    # Extract means and variances (ravel = fast view, not a copy)
    means = model.means_.ravel()
    variances = model.covariances_.ravel()

    # Vectorized sampling: each sample uses the mean/std of its assigned component
    return rng.normal(
        loc=means[comp_idx],
        scale=np.sqrt(variances[comp_idx])
    )


# -------------------------------------------------------------------------------------------
# 1: Optimize GMM Components with BIC and KL Divergence, Rich Table Summary
# -------------------------------------------------------------------------------------------
# def optimize_gmm_components(
#     data: pd.DataFrame,
#     max_components: int = 10,
#     init_params: str ='kmeans',
#     n_init: int =10,
#     random_state: int =None,
#     n_samples: int = 10_000,
#     bins: int = 30
# ):
#     """
#     Sweeps through 1..max_components to evaluate Gaussian Mixture Models (GMM)
#     using BIC and KL Divergence. Produces a modern Rich table summary.

#     Returns
#     -------
#     results_df : pandas.DataFrame
#         Table of n_components, BIC, and KL divergence.
#     best_model : sklearn.mixture.GaussianMixture
#         Model with the lowest KL divergence.
#     """
#     # Initialize Rich console for output
#     console = Console()

#     # Convert Series to NumPy
#     if isinstance(data, pd.Series):
#         data = data.dropna().values

#     # Ensure 2D shape for sklearn
#     X = data.reshape(-1, 1) if data.ndim == 1 else data

#     # Storage
#     results = []
#     best_kl = float("inf")
#     best_model = None

#     # Header panel
#     console.print(
#         Panel(
#             "[bold white]GMM Component Optimization[/bold white]\n"
#             f"[grey50]Max Components:[/grey50] {max_components}\n"
#             f"[grey50]Init Params:[/grey50] {init_params}   "
#             f"[grey50]n_init:[/grey50] {n_init}",
#             box=ROUNDED,
#             border_style="cyan",
#             padding=(1, 2),
#             width=50
#         )
#     )

#     # Results table
#     table = Table(
#         title="Model Fit Summary",
#         box=ROUNDED,
#         header_style="bold cyan",
#         show_lines=False,
#         padding=(0, 1)
#     )
#     table.add_column("Components", justify="center")
#     table.add_column("BIC", justify="right")
#     table.add_column("KL Divergence", justify="right")

#     # Sweep components
#     for n in range(1, max_components + 1):

#         gm = GaussianMixture(
#             n_components=n,
#             covariance_type="full",
#             init_params=init_params,
#             n_init=n_init,
#             random_state=random_state
#         ).fit(X)

#         bic = gm.bic(X)
#         kl = calculate_gmm_kl_divergence(
#             gm, X.flatten(), n_samples, random_state, bins
#         )

#         results.append({
#             "n_components": n,
#             "bic": bic,
#             "kl_divergence": kl
#         })

#         table.add_row(
#             str(n),
#             f"{bic:,.2f}",
#             f"{kl:.4f}"
#         )

#         # Track best model
#         if kl < best_kl:
#             best_kl = kl
#             best_model = gm

#     # Display the results table
#     console.print(table)

#     # Best model panel
#     console.print(
#         Panel(
#             f"[bold green]Best Model:[/bold green] {best_model.n_components} components\n"
#             f"[grey50]Lowest KL Divergence:[/grey50] {best_kl:.4f}",
#             box=ROUNDED,
#             border_style="green",
#             padding=(1, 2),
#             width=50
#         )
#     )

#     return {pd.DataFrame(results), best_model}



def optimize_gmm_components(
    data: pd.DataFrame,
    max_components: int = 10,
    init_params: str = "kmeans",
    n_init: int = 10,
    random_state: int = None,
    n_samples: int = 10_000,
    bins: int = 30
):
    """
    Fit GMM models for 1..max_components, compute BIC + KL divergence,
    display a compact Rich summary, and return all models + diagnostics.

    Returns
    -------
    dict
        {
            "results_df": DataFrame of metrics,
            "all_models": {
                k: {
                    "k": int,
                    "model": GaussianMixture,
                    "gmm_features": np.ndarray
                }
            },
            "best_model": GaussianMixture,
            "best_k": int,
            "best_kl": float,
            "gmm_features": np.ndarray  # predict_proba for best model
        }
    """

    console = Console()

    # --- Normalize input into a clean 1D numeric array ---
    if isinstance(data, (pd.DataFrame, pd.Series)):
        X_flat = data.to_numpy().ravel()
    else:
        X_flat = np.asarray(data).ravel()

    X_flat = X_flat[~np.isnan(X_flat)]
    X = X_flat.reshape(-1, 1)

    # --- Header panel ---
    console.print(
        Panel(
            f"[bold white]GMM Component Optimization[/bold white]\n"
            f"[grey50]max={max_components}, init={init_params}, n_init={n_init}[/grey50]",
            box=ROUNDED,
            border_style="cyan",
            padding=(0, 1),
            width=50
        )
    )

    # --- Results table ---
    table = Table(
        title="Model Fit Summary",
        box=ROUNDED,
        header_style="bold cyan",
        padding=(0, 1),
        show_lines=False
    )
    table.add_column("k", justify="center")
    table.add_column("BIC", justify="right")
    table.add_column("KL", justify="right")

    results = []
    all_models = {}
    best_kl = float("inf")
    best_model = None

    # --- Sweep components ---
    for k in range(1, max_components + 1):

        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            init_params=init_params,
            n_init=n_init,
            random_state=random_state
        ).fit(X)

        bic = gm.bic(X)
        kl = calculate_gmm_kl_divergence(gm, X_flat, n_samples, random_state, bins)

        results.append({"n_components": k, "bic": bic, "kl_divergence": kl})

        # Store full model info for this k
        all_models[k] = {
            "k": k,
            "model": gm,
            "gmm_features": gm.predict_proba(X)
        }

        table.add_row(str(k), f"{bic:,.2f}", f"{kl:.4f}")

        if kl < best_kl:
            best_kl = kl
            best_model = gm

    # --- Display table ---
    console.print(table)

    # --- Best model panel ---
    console.print(
        Panel(
            f"[bold green]Best Model: k={best_model.n_components}[/bold green]\n"
            f"[grey50]KL={best_kl:.4f}[/grey50]",
            box=ROUNDED,
            border_style="green",
            padding=(0, 1),
            width=50
        )
    )

    # --- Posterior probabilities for best model ---
    gmm_features = best_model.predict_proba(X)

    # --- Final structured output ---
    return {
        "results_df": pd.DataFrame(results),
        "all_models": all_models,
        "best_model": best_model,
        "best_k": best_model.n_components,
        "best_kl": best_kl,
        "gmm_features": gmm_features
    }





# -------------------------------------------------------------------------------------------
# 2: Impute Missing Values with GMM and Optional Clipping
# -------------------------------------------------------------------------------------------
def gmm_impute(
    data: pd.DataFrame,
    col: str,
    model: GaussianMixture,
    mode: str = "clustering",
    random_state: int = None,
    clip_min: float = None,
    clip_max: float = None,
    return_mask: bool = False
):
    """
    Impute missing values in a numeric column using a fitted univariate GMM.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    col : str
        Column to impute.
    model : GaussianMixture
        Fitted univariate GMM.
    mode : {"clustering", "modeling"}, default="clustering"
        - "clustering": sample from mixture components.
        - "modeling": fill with mixture mean.
    random_state : int
        RNG seed.
    clip_min, clip_max : float or None
        Optional clipping bounds.
    return_mask : bool
        If True, return (imputed_df, missing_mask).

    Returns
    -------
    DataFrame or (DataFrame, Series)
        Imputed dataset, optionally with mask of imputed rows.
    """

    # --- Validation ---
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    if mode not in {"clustering", "modeling"}:
        raise ValueError("mode must be 'clustering' or 'modeling'")
    if not hasattr(model, "means_") or model.means_.shape != (model.n_components, 1):
        raise ValueError("Model must be a fitted univariate GaussianMixture.")

    # Work on a copy
    out = data.copy()
    mask = out[col].isna()
    n_missing = mask.sum()

    if n_missing == 0:
        return (out, mask) if return_mask else out

    # Extract mixture parameters
    means = model.means_.ravel()
    weights = model.weights_.ravel()
    rng = np.random.default_rng(random_state)

    # --- Clustering mode: sample from mixture ---
    if mode == "clustering":
        # Choose component indices
        comp = rng.choice(model.n_components, size=n_missing, p=weights)

        # Extract variances depending on covariance type
        ct = model.covariance_type
        if ct == "full":
            variances = np.array([model.covariances_[k][0, 0] for k in range(model.n_components)])
        elif ct == "tied":
            variances = np.repeat(model.covariances_[0, 0], model.n_components)
        else:  # diag or spherical
            variances = np.asarray(model.covariances_).ravel()

        # Sample from selected components
        sampled = rng.normal(loc=means[comp], scale=np.sqrt(variances[comp]))

        # Optional clipping
        if clip_min is not None or clip_max is not None:
            sampled = np.clip(
                sampled,
                clip_min if clip_min is not None else -np.inf,
                clip_max if clip_max is not None else np.inf
            )

        out.loc[mask, col] = sampled

    # --- Modeling mode: fill with mixture mean ---
    else:
        imputed_value = float(np.dot(means, weights))

        if clip_min is not None or clip_max is not None:
            imputed_value = float(np.clip(
                imputed_value,
                clip_min if clip_min is not None else -np.inf,
                clip_max if clip_max is not None else np.inf
            ))

        out[col] = out[col].fillna(imputed_value)

    return (out, mask) if return_mask else out


# -------------------------------------------------------------------------------------------
# 3: Calculate KL Divergence Between Real Data and GMM Samples
# -------------------------------------------------------------------------------------------
def calculate_gmm_kl_divergence(model, 
                                real_data, 
                                n_samples: int, 
                                random_state: int, 
                                bins: int):
    """
    Estimate KL Divergence between real data and a fitted univariate GMM.

    Parameters
    ----------
    model : GaussianMixture
        Fitted GMM model.
    real_data : array-like
        Original observations.
    n_samples : int
        Number of synthetic samples to draw from the GMM.
    random_state : int
        RNG seed for reproducibility.
    bins : int
        Number of histogram bins for discretization.

    Returns
    -------
    float
        KL divergence (lower = better fit).
    """

    # Generate synthetic samples from the GMM
    synth = gmm_sample(model, n_samples, random_state)

    # Shared histogram range
    lo = min(real_data.min(), synth.min())
    hi = max(real_data.max(), synth.max())
    hist_range = (lo, hi)

    # Density histograms (PDF approximations)
    p_hist, _ = np.histogram(real_data, bins=bins, range=hist_range, density=True)
    q_hist, _ = np.histogram(synth,     bins=bins, range=hist_range, density=True)

    # Add epsilon to avoid log(0)
    eps = 1e-10
    p = p_hist + eps
    q = q_hist + eps

    # KL(P || Q)
    return entropy(p, q)


# -------------------------------------------------------------------------------------------
# 4: Full GMM Diagnostic with Visualization and KL Divergence
# -------------------------------------------------------------------------------------------
def gmm_full_diagnostic(
    data: pd.Series,
    n_components: int = 2,
    init_params: str = 'kmeans',
    n_init: int = 10,
    n: int = 10_000,
    random_state: int = None,
    title_suffix: str = ""
):
    """
    Perform a full diagnostic evaluation of a univariate Gaussian Mixture Model (GMM).

    Purpose
    -------
    This function assesses how well a GMM represents the underlying data distribution by:
      • Fitting a univariate GMM with the specified number of components.
      • Generating synthetic samples from the fitted mixture model.
      • Comparing real vs synthetic distributions using KL divergence.
      • Visualizing both distributions with overlaid KDE curves.
      • Reporting BIC and KL as quantitative goodness‑of‑fit metrics.

    This diagnostic helps determine whether the chosen GMM configuration
    is an appropriate generative model for the data.

    Parameters
    ----------
    data : array-like or pandas.Series
        Input numeric data.
    n_components : int
        Number of mixture components.
    init_params : str
        Initialization strategy for GMM.
    n_init : int
        Number of random initializations.
    n : int
        Number of synthetic samples to draw.
    random_state : int
        RNG seed.
    title_suffix : str
        Optional label for plot title.

    Returns
    -------
    dict
        {
            "model": GaussianMixture,
            "kl_score": float,
            "bic": float,
            "synthetic_samples": np.ndarray,
            "real_data": np.ndarray
        }
    """

    # Convert Series → NumPy
    if isinstance(data, pd.Series):
        data = data.dropna().values

    # Ensure 2D for sklearn
    X = data.reshape(-1, 1) if data.ndim == 1 else data

    # Fit GMM
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        init_params=init_params,
        n_init=n_init,
        random_state=random_state
    ).fit(X)

    # RNG
    rng = np.random.default_rng(random_state)

    # Sample components based on mixture weights
    comp_idx = rng.choice(model.n_components, size=n, p=model.weights_)

    # Extract means and stds
    means = model.means_.ravel()
    stds = np.sqrt(model.covariances_.ravel())

    # Generate synthetic samples
    X_synth = rng.normal(means[comp_idx], stds[comp_idx])

    # KL divergence (discretized)
    lo = min(X.min(), X_synth.min())
    hi = max(X.max(), X_synth.max())
    hist_range = (lo, hi)

    p_hist, _ = np.histogram(X,        bins=50, range=hist_range, density=True)
    q_hist, _ = np.histogram(X_synth,  bins=50, range=hist_range, density=True)

    eps = 1e-10
    kl_score = entropy(p_hist + eps, q_hist + eps)

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.kdeplot(X.flatten(),   label="Real Data", fill=True, color="royalblue",  alpha=0.4)
    sns.kdeplot(X_synth,       label="GMM Synthetic", fill=True, color="darkorange", alpha=0.3)

    plt.title(
        f"GMM Validation ({n_components} components) {title_suffix}\n"
        f"KL(real‖synth): {kl_score:.4f} | BIC: {model.bic(X):.2f}",
        loc="left"
    )
    plt.xlabel(title_suffix or "Value")
    plt.ylabel("Density")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return structured diagnostic output
    return {
        "model": model,
        "kl_score": kl_score,
        "bic": model.bic(X),
        "synthetic_samples": X_synth,
        "real_data": X.flatten()
    }