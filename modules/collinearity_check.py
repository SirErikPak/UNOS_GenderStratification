import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Define severity levels and corresponding actions
SEVERITY_ORDER = ["Severe", "High", "Moderate"]

# Mapping from severity level - recommended action
ACTION_MAP = {
    "Severe": "Drop or transform variable or consider removing it.",
    "High": "Consider dropping one variable or combining it with another.",
    "Moderate": "Monitor or Consider removing one of the features or combining them.",
}


# -------------------------------------------------------------------------------
# 1: Compute collinearity
# -------------------------------------------------------------------------------
def compute_collinearity(df, handle_missing_vif='median', verbose_vif_errors=True):
    """
    Compute mixed-type collinearity metrics for a dataset.

    Computes:
    - Pearson correlation (numeric–numeric)
    - Cramér's V (categorical–categorical)
    - Eta / correlation ratio (categorical–numeric)
    - VIF (numeric multicollinearity)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    handle_missing_vif : {'median', 'drop'}, default='median'
        How to handle missing values before VIF:
        - 'median': fill missing numeric values with medians
        - 'drop'  : drop rows with missing numeric values
    verbose_vif_errors : bool, default=True
        If True, prints features with R²≈1 (perfect collinearity).

    Returns
    -------
    dict
        Dictionary with keys: 'pearson', 'cramers_v', 'eta', 'vif'.
    """

    # Work on a copy to avoid modifying original data
    local_df = df.copy()

    # Identify numeric and categorical columns
    numeric_cols = local_df.select_dtypes(include=['float', 'int']).columns.tolist()
    cat_cols     = local_df.select_dtypes(include=['object', 'category']).columns.tolist()

    results = {}

    # ------------------------------------------------------------------
    # 1-1: Pearson correlation (numeric–numeric)
    # ------------------------------------------------------------------
    results['pearson'] = (
        local_df[numeric_cols].corr() if len(numeric_cols) > 1 else None
    )

    # ------------------------------------------------------------------
    # 1-2: Cramér's V (categorical–categorical)
    # ------------------------------------------------------------------
    def cramers_v(x, y):
        """Compute Cramér's V for two categorical variables."""
        mask = x.notna() & y.notna()  # remove missing pairs
        x, y = x[mask], y[mask]

        if len(x) == 0:
            return np.nan

        table = pd.crosstab(x, y)
        n = table.sum().sum()
        r, k = table.shape

        # If only one category exists, association undefined
        if min(r, k) <= 1:
            return np.nan

        chi2 = chi2_contingency(table)[0]
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    # Build full symmetric Cramér's V matrix
    if len(cat_cols) > 1:
        mat = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

        for i, c1 in enumerate(cat_cols):
            for j, c2 in enumerate(cat_cols):
                if i == j:
                    mat.iloc[i, j] = 1.0  # perfect with itself
                elif i < j:
                    mat.iloc[i, j] = cramers_v(local_df[c1], local_df[c2])
                else:
                    mat.iloc[i, j] = mat.iloc[j, i]  # mirror upper triangle

        results['cramers_v'] = mat
    else:
        results['cramers_v'] = None

    # ------------------------------------------------------------------
    # 1-3: Eta (categorical–numeric)
    # ------------------------------------------------------------------
    def correlation_ratio(categories, values):
        """Compute correlation ratio (η) for categorical → numeric."""
        mask = pd.Series(categories).notna().values & pd.Series(values).notna().values

        if not mask.any():
            return np.nan

        categories = np.array(categories)[mask]
        values     = np.array(values, dtype=float)[mask]

        cats = np.unique(categories)
        if len(cats) <= 1:
            return np.nan

        overall = values.mean()

        # Between-group variance
        ss_between = sum(
            values[categories == c].size *
            (values[categories == c].mean() - overall) ** 2
            for c in cats
        )

        # Total variance
        ss_total = ((values - overall) ** 2).sum()

        return np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0

    # Build categorical–numeric Eta matrix
    if len(cat_cols) > 0 and len(numeric_cols) > 0:
        eta_mat = pd.DataFrame(index=cat_cols, columns=numeric_cols, dtype=float)

        for cat in cat_cols:
            for num in numeric_cols:
                eta_mat.loc[cat, num] = correlation_ratio(local_df[cat], local_df[num])

        results['eta'] = eta_mat
    else:
        results['eta'] = None

    # ------------------------------------------------------------------
    # 1-4: VIF (numeric multicollinearity)
    # ------------------------------------------------------------------
    if len(numeric_cols) > 1:

        # Handle missing values before VIF
        if handle_missing_vif == 'median':
            X = local_df[numeric_cols].fillna(local_df[numeric_cols].median())
        else:
            X = local_df[numeric_cols].dropna()

        # If no rows remain, VIF cannot be computed
        if X.shape[0] == 0:
            raise ValueError("VIF failed: no complete rows remain. Impute data first.")

        # Add constant column for VIF computation
        X_vif = X.assign(const=1)
        X_vif_values = X_vif.values

        # Optional diagnostic: detect perfect collinearity
        if verbose_vif_errors:
            problem_features = []
            for i, col in enumerate(numeric_cols):
                y = X_vif_values[:, i]
                x_others = np.delete(X_vif_values, i, axis=1)
                r_sq = sm.OLS(y, x_others).fit().rsquared
                if r_sq >= 0.9999:  # nearly perfect R²
                    problem_features.append((col, r_sq))

            if problem_features:
                print("⚠ VIF divide-by-zero / inf detected for:")
                for col, r_sq in problem_features:
                    print(f"  - {col}  (R²={r_sq:.10f})")
                print("  These features are perfectly collinear with the rest.")

        # Compute VIF for each numeric feature
        results['vif'] = pd.DataFrame({
            'feature': numeric_cols,
            'VIF': [
                variance_inflation_factor(X_vif_values, i)
                for i in range(len(numeric_cols))
            ]
        }).sort_values('VIF', ascending=False).reset_index(drop=True)

    else:
        results['vif'] = None

    return results


# -------------------------------------------------------------------------------
# 2a: Assess collinearity severity
# -------------------------------------------------------------------------------
# Define severity levels and corresponding actions
def get_severity(metric_type: str, value: float) -> str:
    """Assign severity level based on metric type and value."""
    if metric_type == "VIF":
        return "Severe" if value >= 10 else "Moderate"
    return "High" if value >= 0.90 else "Moderate"

# -------------------------------------------------------------------------------
# 2b: Define the recommended actions for each severity level
# -------------------------------------------------------------------------------
def build_flag(metric_type, var1, var2, value, pair_type):
    """Construct a standardized flag record."""
    severity = get_severity(metric_type, value)
    return {
        "type": metric_type,
        "var1": var1,
        "var2": var2,
        "value": round(abs(value), 4),
        "severity": severity,
        "pair_type": pair_type,
        "recommended_action": ACTION_MAP[severity],
    }

# -------------------------------------------------------------------------------
# 2c: Define the recommended actions for each severity level
# -------------------------------------------------------------------------------
def scan_symmetric_matrix(matrix, metric_type, threshold, pair_type):
    """Scan upper triangle of a symmetric matrix (Pearson or Cramér's V)."""
    flags = []
    cols = matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = matrix.iloc[i, j]
            if pd.notna(value) and abs(value) >= threshold:
                flags.append(build_flag(metric_type, cols[i], cols[j], value, pair_type))

    return flags


# -------------------------------------------------------------------------------
# 2d: Scan Eta matrix
# -------------------------------------------------------------------------------
def scan_eta_matrix(matrix, threshold):
    """Scan categorical–numeric Eta matrix."""
    flags = []
    for cat in matrix.index:
        for num in matrix.columns:
            value = matrix.loc[cat, num]
            if pd.notna(value) and value >= threshold:
                flags.append(build_flag("Eta", cat, num, value, "categorical-numeric"))
    return flags


# -------------------------------------------------------------------------------
# 2e: Scan VIF table
# -------------------------------------------------------------------------------
def scan_vif(vif_df, threshold):
    """Scan VIF table for problematic multicollinearity."""
    flags = []
    for _, row in vif_df.iterrows():
        if (
            row["feature"] != "const"
            and pd.notna(row["VIF"])
            and np.isfinite(row["VIF"])
            and row["VIF"] >= threshold
        ):
            flags.append(build_flag("VIF", row["feature"], "—", row["VIF"], "numeric"))
    return flags

# -------------------------------------------------------------------------------
# 3: Summarize collinearity issues
# -------------------------------------------------------------------------------
def summarise_collinearity(
    results,
    pearson_thresh=0.70,
    cramers_thresh=0.70,
    eta_thresh=0.70,
    vif_thresh=5.0,
):
    """
    Summarize all detected collinearity issues across:
    - Pearson
    - Cramér's V
    - Eta
    - VIF

    Parameters
    ----------
    results : dict
        Output from compute_collinearity().
    pearson_thresh, cramers_thresh, eta_thresh : float
        Thresholds for flagging correlation-based metrics.
    vif_thresh : float
        Threshold for flagging VIF.

    Returns
    -------
    pd.DataFrame or None
        Sorted table of flagged collinearity issues.
    """

    # Map metric → scanning function
    scanners = {
        "pearson": lambda: scan_symmetric_matrix(
            results["pearson"], "Pearson", pearson_thresh, "numeric-numeric"
        ),
        "cramers_v": lambda: scan_symmetric_matrix(
            results["cramers_v"], "Cramér's V", cramers_thresh, "categorical-categorical"
        ),
        "eta": lambda: scan_eta_matrix(results["eta"], eta_thresh),
        "vif": lambda: scan_vif(results["vif"], vif_thresh),
    }

    flags = []

    # Run scanners only for metrics that exist
    for key, scanner in scanners.items():
        if results.get(key) is not None:
            flags.extend(scanner())

    # No issues found
    if not flags:
        print("No collinearity flags above thresholds.")
        return None

    # Build output DataFrame
    output = pd.DataFrame(flags)

    # Ensure severity sorts correctly
    output["severity"] = pd.Categorical(
        output["severity"], categories=SEVERITY_ORDER, ordered=True
    )

    return (
        output.sort_values(["severity", "value"], ascending=[True, False])
        .reset_index(drop=True)
    )