from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def gmm_sample(model, n, random_state):
    """
    Draws n samples from a fitted GaussianMixture model using modern NumPy RNG.
    
    Parameters:
    -----------
    model : GaussianMixture (A fitted sklearn GMM model)
    n : int
        Number of samples to draw.
    random_state : int, np.random.Generator, or None
        Seed or Generator for reproducibility.
    """
    # Initialize the modern Generator
    rng = np.random.default_rng(random_state)
    
    # Choose component indices according to mixture weights
    comp = rng.choice(
        model.weights_.size,
        size=n,
        p=model.weights_
    )
    
    # Extract means and variances using .ravel() for efficiency
    # .ravel() provides a view rather than a copy
    means = model.means_.ravel()
    covs = model.covariances_.ravel()
    
    # Sample using the Generator's normal method
    # loc and scale accept arrays, allowing for vectorized sampling
    return rng.normal(
        loc=means[comp],
        scale=np.sqrt(covs[comp])
    )


def gmm_full_diagnostic(data, random_state, n_components=2, init_params='kmeans', n_init=10, 
                        n=10_000, title_suffix=""):
    """
    Fits a GMM, generates synthetic samples, calculates KL Divergence, 
    and plots the result for a complete data science validation.
    """
    # Prep Data: Convert Series to NumPy and handle dimensions
    if isinstance(data, pd.Series):
        data = data.dropna().values
        
    X = data.reshape(-1, 1) if data.ndim == 1 else data
    
    # Fit Model
    model = GaussianMixture(
        n_components=n_components, 
        covariance_type="full",    # capturing complex relationships
        init_params=init_params,   # how the starting parameters of the GMM are chosen
        n_init=n_init,             # run GMM X times each time with a different random initialization 
                                   # keep the best model (highest log-likelihood)
        random_state=random_state
    ).fit(X)

    # Initialize the modern generator
    rng = np.random.default_rng(random_state)
    
    # Select components based on weights
    comp_idx = rng.choice(model.weights_.size, size=n, p=model.weights_)
    means = model.means_.flatten()
    stds = np.sqrt(model.covariances_.flatten())
    X_synth = np.random.normal(means[comp_idx], stds[comp_idx])
    
    # Calculate KL Divergence (Discretized)
    common_range = (min(X.min(), X_synth.min()), max(X.max(), X_synth.max()))
    p_hist, _ = np.histogram(X, bins=50, range=common_range, density=True)
    q_hist, _ = np.histogram(X_synth, bins=50, range=common_range, density=True)
    kl_score = entropy(p_hist + 1e-10, q_hist + 1e-10)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.kdeplot(X.flatten(), label='Real Data', fill=True, color='royalblue', alpha=0.4)
    sns.kdeplot(X_synth, label=f'GMM (n={n_components})', fill=True, color='darkorange', alpha=0.3)

    # Formatting
    plt.title(f"GMM Validation: {model.n_components} Components {title_suffix}\n"
              f"KL Divergence: {kl_score:.4f} | BIC: {model.bic(X_synth.reshape(-1,1)):.2f}")
    plt.xlabel(title_suffix)
    plt.ylabel("Density")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    
    return model, kl_score


def optimize_gmm_components(data, random_state, max_components=10, 
                            init_params='kmeans', 
                            n_init=10, 
                            n_samples=10_000,
                            bins=30):
    """
    Sweeps through n_components to find the optimal GMM fit based on BIC and KL Divergence.
    
    Returns:
        results_df: DataFrame containing n, BIC, and KL scores.
        best_model: The GaussianMixture object with the lowest KL Divergence.
    """
    # initialize variables
    results = []
    best_kl = float('inf')
    best_model = None

    # Prep Data: Convert Series to NumPy and handle dimensions
    if isinstance(data, pd.Series):
        data = data.dropna().values
    
    # Ensure data is 2D for sklearn (N, 1)
    X = data.reshape(-1, 1) if data.ndim == 1 else data
    
    print(f"{'n':>3} | {'BIC':>12} | {'KL Div':>8}")
    print("-" * 30)

    # iterate components
    for n in range(1, max_components + 1):
        # n_init=n_init provides stability against local optima
        gm = GaussianMixture(
            n_components=n, 
            covariance_type="full",
            init_params=init_params,
            n_init=n_init, 
            random_state=random_state
        ).fit(X)
        
        # Calculate KL using the function we built
        kl = calculate_gmm_kl_divergence(gm, X.flatten(), n_samples, random_state, bins)
        bic = gm.bic(X)
        
        results.append({
            'n_components': n,
            'bic': bic,
            'kl_divergence': kl
        })
        
        print(f"{n:3d} | {bic:12.2f} | {kl:8.4f}")
        
        # Track the best performing model based on KL
        if kl < best_kl:
            best_kl = kl
            best_model = gm
            
    return pd.DataFrame(results), best_model
    

def calculate_gmm_kl_divergence(model, real_data, n_samples, random_state, bins):
    """
    Estimates the KL Divergence between the real data distribution and 
    the GMM generative model.
    
    Parameters:
    -----------
    model : GaussianMixture object
        The fitted GMM.
    real_data : array-like
        The original observations (e.g., Lung P/F ratios).
    n_samples : int
        Number of synthetic samples to generate for the estimation.
    bins : int
        Number of bins for discretizing the distributions.
        
    Returns:
    --------
    kl_dist : float
        The KL Divergence score (lower is better).
    """
    # Generate synthetic data using a custom sampler
    synth_data = gmm_sample(model, n_samples, random_state)
    
    # Define a common range for binning to ensure alignment
    data_min = min(real_data.min(), synth_data.min())
    data_max = max(real_data.max(), synth_data.max())
    common_range = (data_min, data_max)
    
    # Calculate histograms (PDFs)
    # Add a tiny epsilon (1e-10) to avoid division by zero or log(0)
    p_hist, _ = np.histogram(real_data, bins=bins, range=common_range, density=True)
    q_hist, _ = np.histogram(synth_data, bins=bins, range=common_range, density=True)
    # tiny epsilon
    p_prob = p_hist + 1e-10
    q_prob = q_hist + 1e-10
    
    # Compute KL Divergence (P || Q)
    # entropy(pk, qk) computes S = sum(pk * log(pk / qk))
    kl_dist = entropy(p_prob, q_prob)
    
    return kl_dist
    

def gmm_impute(
    data,
    col,
    model,
    random_state,
    mode="clustering",
    clip_min=None,
    clip_max=None,
    return_mask=False
):
    """
    Impute missing values in a 1D numeric column using a fitted GaussianMixture.

    mode="clustering":
        Sample from the fitted marginal mixture distribution.

    mode="modeling":
        Fill missing values with the mixture mean.
    """
    # Validations
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    if mode not in {"clustering", "modeling"}:
        raise ValueError("mode must be 'clustering' or 'modeling'")
    if not hasattr(model, "means_") or not hasattr(model, "weights_"):
        raise ValueError("model must be a fitted sklearn.mixture.GaussianMixture object.")
    if getattr(model, "means_", None) is None or model.means_.ndim != 2 or model.means_.shape[1] != 1:
        raise ValueError("This function is intended for a univariate GMM fit on one numeric column.")

    # Work on a copy to avoid mutating
    out = data.copy()
    mask = out[col].isna()
    n_missing = int(mask.sum())

    # Nothing to impute
    if n_missing == 0:
        return (out, mask) if return_mask else out

    # array of means & weights and collapse it into a simple 1D list
    means = model.means_.ravel()
    weights = model.weights_.ravel()
    rng = np.random.default_rng(random_state) # sets a global state (stays local to the variable)

    if mode == "clustering":
        comp = rng.choice(model.n_components, size=n_missing, p=weights)

        ct = model.covariance_type
        if ct == "full":
            variances = np.array([model.covariances_[k][0, 0] for k in range(model.n_components)])
        elif ct == "tied":
            variances = np.repeat(model.covariances_[0, 0], model.n_components)
        elif ct in {"diag", "spherical"}:
            variances = np.asarray(model.covariances_).ravel()
        else:
            raise ValueError(f"Unsupported covariance_type: {ct}")

        sampled = rng.normal(loc=means[comp], scale=np.sqrt(variances[comp]))

        if clip_min is not None or clip_max is not None:
            sampled = np.clip(
                sampled,
                clip_min if clip_min is not None else -np.inf,
                clip_max if clip_max is not None else np.inf,
            )

        out.loc[mask, col] = sampled

    else:
        imputed_value = float(np.dot(means, weights))
        if clip_min is not None or clip_max is not None:
            imputed_value = float(np.clip(
                imputed_value,
                clip_min if clip_min is not None else -np.inf,
                clip_max if clip_max is not None else np.inf,
            ))
        out[col] = out[col].fillna(imputed_value)

    return (out, mask) if return_mask else out