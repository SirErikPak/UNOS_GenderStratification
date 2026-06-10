import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEVERITY_ORDER = ["Severe", "High", "Moderate"]

# Maps severity levels to recommended actions
ACTION_MAP = {
    "Severe": "Drop or transform variable",
    "High": "Consider dropping one variable",
    "Moderate": "Monitor",
}


def compute_collinearity(df, handle_missing_vif='median'):
    """
    Compute mixed-type collinearity metrics for a dataset.

    This function calculates:
    - Pearson correlation (numeric–numeric)
    - Cramér's V (categorical–categorical)
    - Eta / correlation ratio (categorical–numeric)
    - VIF (numeric multicollinearity)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing numeric and/or categorical variables.

    handle_missing_vif : {'median', 'drop'}, default='median'
        Strategy for handling missing values before VIF computation:
        - 'median': fill missing numeric values with column medians
        - 'drop'  : drop rows with missing numeric values (may reduce sample size)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'pearson'   : DataFrame of Pearson correlations
        - 'cramers_v' : DataFrame of Cramér's V values
        - 'eta'       : DataFrame of Eta values
        - 'vif'       : DataFrame of VIF values
    """

    # Work on a local copy to avoid modifying original data
    local_df = df.copy()

    # Identify numeric and categorical columns
    numeric_cols = local_df.select_dtypes(include=['float', 'int']).columns.tolist()
    cat_cols     = local_df.select_dtypes(include=['object', 'category']).columns.tolist()

    results = {}

    # ------------------------------------------------------------------
    # 1. Pearson correlation (numeric–numeric)
    # ------------------------------------------------------------------
    results['pearson'] = (
        local_df[numeric_cols].corr() if len(numeric_cols) > 1 else None
    )

    # ------------------------------------------------------------------
    # 2. Cramér's V (categorical–categorical)
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

        # If only one category exists, association is undefined
        if min(r, k) <= 1:
            return np.nan

        chi2 = chi2_contingency(table)[0]
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    if len(cat_cols) > 1:
        mat = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

        # Fill symmetric matrix
        for i, c1 in enumerate(cat_cols):
            for j, c2 in enumerate(cat_cols):
                if i == j:
                    mat.iloc[i, j] = 1.0  # perfect association with itself
                elif i < j:
                    mat.iloc[i, j] = cramers_v(local_df[c1], local_df[c2])
                else:
                    mat.iloc[i, j] = mat.iloc[j, i]  # mirror upper triangle

        results['cramers_v'] = mat
    else:
        results['cramers_v'] = None

    # ------------------------------------------------------------------
    # 3. Eta (categorical–numeric)
    # ------------------------------------------------------------------
    def correlation_ratio(categories, values):
        """Compute correlation ratio (η) for categorical → numeric association."""
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

    if len(cat_cols) > 0 and len(numeric_cols) > 0:
        eta_mat = pd.DataFrame(index=cat_cols, columns=numeric_cols, dtype=float)

        # Compute eta for each categorical–numeric pair
        for cat in cat_cols:
            for num in numeric_cols:
                eta_mat.loc[cat, num] = correlation_ratio(local_df[cat], local_df[num])

        results['eta'] = eta_mat
    else:
        results['eta'] = None

    # ------------------------------------------------------------------
    # 4. VIF (numeric multicollinearity)
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

        # Add constant term for VIF computation
        X_vif = X.assign(const=1)

        # Compute VIF for each numeric feature
        results['vif'] = pd.DataFrame({
            'feature': numeric_cols,
            'VIF': [
                variance_inflation_factor(X_vif.values, i)
                for i in range(len(numeric_cols))
            ]
        }).sort_values('VIF', ascending=False).reset_index(drop=True)

    else:
        results['vif'] = None

    return results



def get_severity(metric_type: str, value: float) -> str:
    """
    Determine severity level for a collinearity metric.

    Parameters
    ----------
    metric_type : str
        The type of metric ("Pearson", "Cramér's V", "Eta", "VIF").
    value : float
        The metric value.

    Returns
    -------
    str
        Severity category: "Severe", "High", or "Moderate".
    """

    # VIF uses different thresholds than correlation metrics
    if metric_type == "VIF":
        return "Severe" if value >= 10 else "Moderate"

    # Correlation-based metrics: High if ≥ 0.90
    return "High" if value >= 0.90 else "Moderate"


def build_flag(metric_type, var1, var2, value, pair_type):
    """
    Construct a standardized flag record for a collinearity issue.

    Parameters
    ----------
    metric_type : str
        Type of metric producing the flag.
    var1, var2 : str
        Variable names involved in the flagged relationship.
    value : float
        Strength of association.
    pair_type : str
        Relationship type (numeric-numeric, categorical-categorical, etc.)

    Returns
    -------
    dict
        A dictionary describing the flagged collinearity issue.
    """

    severity = get_severity(metric_type, value)

    return {
        "type": metric_type,
        "var1": var1,
        "var2": var2,
        "value": round(abs(value), 4),  # absolute value for consistency
        "severity": severity,
        "pair_type": pair_type,
        "recommended_action": ACTION_MAP[severity],
    }


def scan_symmetric_matrix(matrix, metric_type, threshold, pair_type):
    """
    Scan the upper triangle of a symmetric association matrix
    (Pearson or Cramér's V).

    Parameters
    ----------
    matrix : pd.DataFrame
        Symmetric matrix of pairwise associations.
    metric_type : str
        Name of the metric.
    threshold : float
        Minimum value required to flag a pair.
    pair_type : str
        Relationship type.

    Returns
    -------
    list of dict
        List of flagged collinearity issues.
    """

    flags = []
    cols = matrix.columns.tolist()

    # Only scan upper triangle to avoid duplicates
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = matrix.iloc[i, j]

            # Flag if above threshold
            if pd.notna(value) and abs(value) >= threshold:
                flags.append(
                    build_flag(metric_type, cols[i], cols[j], value, pair_type)
                )

    return flags


def scan_eta_matrix(matrix, threshold):
    """
    Scan categorical–numeric Eta matrix.

    Parameters
    ----------
    matrix : pd.DataFrame
        Eta values for categorical rows × numeric columns.
    threshold : float
        Minimum eta required to flag.

    Returns
    -------
    list of dict
        Flagged categorical–numeric associations.
    """

    flags = []

    for cat in matrix.index:
        for num in matrix.columns:
            value = matrix.loc[cat, num]

            # Eta is always positive; no abs() needed
            if pd.notna(value) and value >= threshold:
                flags.append(
                    build_flag("Eta", cat, num, value, "categorical-numeric")
                )

    return flags


def scan_vif(vif_df, threshold):
    """
    Scan VIF results for problematic multicollinearity.

    Parameters
    ----------
    vif_df : pd.DataFrame
        DataFrame with columns ["feature", "VIF"].
    threshold : float
        Minimum VIF required to flag.

    Returns
    -------
    list of dict
        Flagged VIF issues.
    """

    flags = []

    for _, row in vif_df.iterrows():

        # Skip intercept term
        if (
            row["feature"] != "const"
            and pd.notna(row["VIF"])
            and row["VIF"] >= threshold
        ):
            flags.append(
                build_flag("VIF", row["feature"], "—", row["VIF"], "numeric")
            )

    return flags


def summarise_collinearity(
    results,
    pearson_thresh=0.70,
    cramers_thresh=0.70,
    eta_thresh=0.70,
    vif_thresh=5.0,
):
    """
    Summarize all detected collinearity issues across:
    - Pearson (numeric–numeric)
    - Cramér's V (categorical–categorical)
    - Eta (categorical–numeric)
    - VIF (numeric multicollinearity)

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
        Sorted table of flagged collinearity issues,
        or None if no issues exceed thresholds.
    """

    # Map each metric to its scanning function
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

    # Run scanners only for metrics present in results
    for key, scanner in scanners.items():
        if results.get(key) is not None:
            flags.extend(scanner())

    # No collinearity detected
    if not flags:
        print("No collinearity flags above thresholds.")
        return None

    # Build output DataFrame
    output = pd.DataFrame(flags)

    # Ensure severity sorts correctly
    output["severity"] = pd.Categorical(
        output["severity"], categories=SEVERITY_ORDER, ordered=True
    )

    # Sort by severity then by strength
    return (
        output.sort_values(["severity", "value"], ascending=[True, False])
        .reset_index(drop=True)
    )
