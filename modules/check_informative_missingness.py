import numpy as np
import pandas as pd
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def _adjust_pvalues(p_raw, method):
    """Shared multiple-comparison correction. method: None, 'bonferroni', 'fdr_bh'."""
    p_raw = np.asarray(p_raw, dtype=float)
    m = len(p_raw)
    if m == 0 or method is None:
        return p_raw.copy()
    if method == 'bonferroni':
        return np.clip(p_raw * m, 0, 1)
    if method == 'fdr_bh':
        order = np.argsort(p_raw)
        ranked = p_raw[order]
        ranks = np.arange(1, m + 1)
        bh = np.clip(ranked * m / ranks, 0, 1)
        bh = np.minimum.accumulate(bh[::-1])[::-1]   # enforce monotonicity
        adj = np.empty(m)
        adj[order] = bh
        return adj
    raise ValueError(f"Unknown correction method: {method!r}")


def _cramers_v(chi2_stat, table):
    """Bias-corrected Cramer's V (Bergsma & Wicher 2013). The naive/uncorrected
    formula systematically inflates when a table has many categories relative
    to n (high df, sparse cells) -- it can report V near 1 even for data
    generated with NO relationship at all. This correction shrinks toward 0
    in exactly that regime. Rough magnitude guide on the corrected value:
    <0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, >=0.5 large."""
    n = table.values.sum()
    r, k = table.shape
    if n <= 1 or min(r, k) < 2:
        return 0.0
    phi2 = chi2_stat / n
    phi2_tilde = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_tilde = r - (r - 1) ** 2 / (n - 1)
    k_tilde = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_tilde - 1, r_tilde - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_tilde / denom))


def _collapse_rare_categories(series, min_count=5, max_categories=10):
    """Bucket low-frequency / excess categories into '__other__' before a
    chi-square test. A table with many categories and few observations per
    cell (e.g. 20 categories across 45 rows) violates the chi-square
    approximation badly and can produce a wildly misleading effect size and
    p-value even from pure noise -- collapsing first is the standard fix,
    not just adjusting the formula after the fact.
    Returns (collapsed_series, n_categories_collapsed).
    """
    counts = series.value_counts()
    keep = set(counts[counts >= min_count].index)
    # also cap total categories: keep the top `max_categories` by frequency
    if len(keep) > max_categories:
        keep = set(counts.index[:max_categories])
    n_collapsed = (~series.isin(keep)).sum()
    if n_collapsed == 0:
        return series, 0
    collapsed = series.where(series.isin(keep), other="__other__")
    return collapsed, int(n_collapsed)


def _rank_biserial(U_stat, n1, n2):
    """Effect size for Mann-Whitney U, range [-1, 1]. Same rough magnitude
    guide as Cramer's V: <0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium,
    >=0.5 large."""
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(1 - (2 * U_stat) / (n1 * n2))


def _effect_strength(abs_effect):
    if abs_effect < 0.1:
        return "Negligible"
    elif abs_effect < 0.3:
        return "Small"
    elif abs_effect < 0.5:
        return "Medium"
    return "Large"


def _compute_outcome_stats(data, col, target, unknown_val):
    """Part 1 numbers only -- no rendering. Returns dict, or None if too few obs."""
    if unknown_val is not None:
        is_unknown = (data[col] == unknown_val) | (data[col].isna())
    else:
        is_unknown = data[col].isna()

    unknown = data.loc[is_unknown, target].dropna()
    known = data.loc[~is_unknown, target].dropna()

    if len(unknown) < 2 or len(known) < 2:
        return {"col": col, "insufficient": True, "is_unknown": is_unknown}

    n_u, n_k = len(unknown), len(known)
    m_u, m_k = unknown.mean(), known.mean()
    var_u, var_k = unknown.var(ddof=1), known.var(ddof=1)

    dof = n_u + n_k - 2
    pooled_std = np.sqrt(((n_u - 1) * var_u + (n_k - 1) * var_k) / dof)
    d = (m_u - m_k) / pooled_std if pooled_std != 0 else 0

    se_d = np.sqrt((n_u + n_k) / (n_u * n_k) + (d ** 2) / (2 * (n_u + n_k)))
    z = stats.norm.ppf(0.975)
    lower, upper = d - z * se_d, d + z * se_d

    abs_d = abs(d)
    if abs_d < 0.2:
        strength = "Negligible"
    elif abs_d < 0.5:
        strength = "Small"
    elif abs_d < 0.8:
        strength = "Medium"
    else:
        strength = "Large"

    t_stat, p_val = stats.ttest_ind(unknown, known, equal_var=False)

    return {
        "col": col, "insufficient": False, "is_unknown": is_unknown,
        "n_u": n_u, "n_k": n_k, "m_u": m_u, "m_k": m_k,
        "d": d, "lower": lower, "upper": upper, "strength": strength,
        "t_stat": t_stat, "p_raw": p_val,
    }


def _compute_predictor_stats(data, col, target, is_unknown, predictors, alpha, correction,
                              effect_threshold=0.1, min_category_count=5, max_categories=10):
    """Part 2: missingness-indicator vs other observed columns. Self-contained
    family (corrected across predictors of THIS column only).

    With large n (thousands+), the chi-square / Mann-Whitney p-value alone
    becomes nearly useless -- trivial associations get p < 0.0001 just from
    sample size. `effect_threshold` sets the minimum |Cramer's V| / |rank-
    biserial r| (default 0.1, conventionally "small") required, IN ADDITION
    to p_adj < alpha, for a predictor to be flagged as practically
    meaningful rather than just statistically detectable noise.

    `min_category_count` / `max_categories` control rare-category collapsing
    for categorical predictors before the chi-square test (see
    _collapse_rare_categories). The default max_categories=10 will collapse
    e.g. a 50-level US-state column down to its top 10 states + "Other" --
    fine for screening, but raise this if a specific high-cardinality
    predictor is one you actually want to inspect at full resolution.
    """
    indicator = is_unknown.astype(int)
    candidate_cols = [c for c in data.columns if c not in (col, target)] \
        if predictors is None else list(predictors)

    rows = []
    for p_col in candidate_cols:
        series = data[p_col]
        valid = series.notna()
        if valid.sum() < 5:
            continue
        ind_valid = indicator[valid]
        if ind_valid.nunique() < 2:
            continue

        sparse = False
        n_collapsed = 0
        if pd.api.types.is_numeric_dtype(series):
            g0 = series[valid][ind_valid == 0]
            g1 = series[valid][ind_valid == 1]
            if len(g0) < 3 or len(g1) < 3:
                continue
            test_stat, p_raw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            test_name = "Mann-Whitney U"
            effect = _rank_biserial(test_stat, len(g0), len(g1))
            n_used = len(g0) + len(g1)
        else:
            cat_series, n_collapsed = _collapse_rare_categories(
                series[valid].astype(str), min_count=min_category_count, max_categories=max_categories
            )
            table = pd.crosstab(cat_series, ind_valid)
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            n_used = int(table.values.sum())
            expected = stats.contingency.expected_freq(table.values)
            sparse = bool((expected < 5).any())
            chi2_stat, p_chi2, _, _ = stats.chi2_contingency(table)
            effect = _cramers_v(chi2_stat, table)
            if sparse and table.shape == (2, 2):
                # chi-square is unreliable with low expected counts in a 2x2;
                # Fisher's exact is the standard fallback there
                _, p_raw = stats.fisher_exact(table.values)
                test_name = "Fisher exact (sparse)"
            else:
                p_raw = p_chi2
                test_name = "Chi-square" + (" (sparse)" if sparse else "")

        rows.append({"predictor": p_col, "test": test_name, "n": n_used,
                      "statistic": test_stat if pd.api.types.is_numeric_dtype(series) else chi2_stat,
                      "p_raw": p_raw, "effect_size": effect, "sparse": sparse,
                      "n_collapsed": n_collapsed})

    pred_df = pd.DataFrame(rows)
    if not pred_df.empty:
        pred_df["p_adj"] = _adjust_pvalues(pred_df["p_raw"].values, correction)
        pred_df["effect_strength"] = pred_df["effect_size"].abs().apply(_effect_strength)
        pred_df["statistically_significant"] = pred_df["p_adj"] < alpha
        pred_df["significant"] = (
            pred_df["statistically_significant"] & (pred_df["effect_size"].abs() >= effect_threshold)
        )
        pred_df["_abs_effect"] = pred_df["effect_size"].abs()
        pred_df = pred_df.sort_values(["significant", "_abs_effect"], ascending=[False, False])
        pred_df = pred_df.drop(columns="_abs_effect")
    return pred_df


def _render_panel(data, stats_dict, target, txt, predictors, alpha, correction,
                   effect_threshold=0.1, min_category_count=5, max_categories=10,
                   family_note=None):
    col = stats_dict["col"]
    console = Console()

    if stats_dict["insufficient"]:
        console.print(
            Panel(
                f"[bold yellow]⚠ Insufficient Sample Size[/bold yellow]\n"
                f"[grey50]Feature '[bold cyan]{col}[/bold cyan]' requires at least 2 entries per group to run Welch's t-test.[/grey50]",
                title=f"[bold grey27]⚙ Pipeline Warning: {col}[/bold grey27]",
                title_align="left",
                border_style="yellow",
                expand=False,
            )
        )
        return

    n_u, n_k = stats_dict["n_u"], stats_dict["n_k"]
    m_u, m_k = stats_dict["m_u"], stats_dict["m_k"]
    d, lower, upper, strength = stats_dict["d"], stats_dict["lower"], stats_dict["upper"], stats_dict["strength"]
    p_val = stats_dict["p_adj"]  # NOTE: adjusted p-value drives the verdict from here on

    if p_val < alpha:
        result_color = "red"
        result_badge = "[bold red]⚠ INFORMATIVE MISSINGNESS DETECTED[/bold red]"
        interpretation = "[orange3]Statistical Significance:[/orange3] Not MCAR. Outcome differs by missingness status."
    else:
        result_color = "green"
        result_badge = "[bold green]✔ RANDOM MISSINGNESS DETECTED[/bold green]"
        interpretation = "[grey50]Statistical Significance: Consistent with MCAR (Missing Completely at Random).[/grey50]"

    grid = Table.grid(expand=False)

    stats_table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold grey50")
    stats_table.add_column("Cohort Group", min_width=18, justify="left")
    stats_table.add_column("Group Sizes", min_width=18, justify="right", style="magenta")
    stats_table.add_column("Mean Survival", min_width=20, justify="right", style="yellow")
    stats_table.add_row("Unknown / Missing", f"Unknown={int(n_u):,}", f"{m_u:,.1f}d")
    stats_table.add_row("Known / Complete", f"Known={int(n_k):,}", f"{m_k:,.1f}d")

    grid.add_row(f"[bold white]Cohort Overview[/bold white] {f'[grey35]({txt})[/grey35]' if txt else ''}")
    grid.add_row(stats_table)
    grid.add_row(f"[grey35]Difference:[/grey35]            [bold white]{m_u - m_k:+,.1f} days[/bold white]\n")

    grid.add_row("─" * 58)
    grid.add_row("\n[bold white]Part 1: Outcome Relationship (Welch's t-test)[/bold white]")
    p_label = f"p ({correction})" if family_note else "p (raw)"
    grid.add_row(f"[grey35]{p_label}:[/grey35]  [bold cyan]{p_val:.4f}[/bold cyan]"
                 + (f"  [grey35](raw {stats_dict['p_raw']:.4f})[/grey35]" if family_note else "") + "\n")
    if family_note:
        grid.add_row(f"[grey35]{family_note}[/grey35]\n")
    grid.add_row(result_badge)
    grid.add_row(interpretation)

    grid.add_row("\n[bold white]Cohen's Analysis[/bold white]")
    grid.add_row(f"[grey35]Cohen's d:[/grey35]             [bold cyan]{d:.4f}[/bold cyan] ({strength})")
    grid.add_row(f"[grey35]95% CI:[/grey35]                [bold cyan][{lower:.4f}, {upper:.4f}][/bold cyan]\n")

    # ---------------- Part 2: predictor panel ----------------
    pred_df = _compute_predictor_stats(
        data, col, target, stats_dict["is_unknown"], predictors, alpha, correction,
        effect_threshold=effect_threshold, min_category_count=min_category_count,
        max_categories=max_categories,
    )

    grid.add_row("\n" + "─" * 58)
    grid.add_row("\n[bold white]Part 2: Predictor Relationship (MAR check)[/bold white]")

    if pred_df.empty:
        grid.add_row("[grey50]No eligible predictor columns to test.[/grey50]")
        any_predictor_sig = False
    else:
        pred_table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold grey50")
        pred_table.add_column("Predictor", min_width=16, justify="left")
        pred_table.add_column("Test", min_width=18, justify="left", style="grey50")
        pred_table.add_column("n", min_width=7, justify="right", style="grey50")
        pred_table.add_column(f"p ({correction or 'raw'})", min_width=10, justify="right", style="cyan")
        pred_table.add_column("Effect", min_width=16, justify="right", style="cyan")
        pred_table.add_column("Flag", min_width=10, justify="center")
        any_collapsed = False
        for _, row in pred_df.iterrows():
            if row["significant"]:
                flag_str = "[bold red]MAR[/bold red]"
            elif row["statistically_significant"]:
                flag_str = "[grey50]p-only[/grey50]"
            else:
                flag_str = "[grey50]No[/grey50]"
            predictor_label = row["predictor"]
            if row.get("n_collapsed", 0) > 0:
                predictor_label += " [yellow]*[/yellow]"
                any_collapsed = True
            pred_table.add_row(
                predictor_label, row["test"], f"{row['n']:,}", f"{row['p_adj']:.4f}",
                f"{row['effect_size']:.3f} ({row['effect_strength']})", flag_str,
            )
        grid.add_row(pred_table)
        if any_collapsed:
            grid.add_row("[yellow]*[/yellow][grey35] = rare/excess categories collapsed into 'Other' "
                          "before testing (high-cardinality predictor, avoids unreliable sparse-table stats)[/grey35]")
        grid.add_row(
            f"\n[grey35]Flag legend: [bold red]MAR[/bold red] = significant AND |effect| ≥ 0.1  |  "
            f"[grey50]p-only[/grey50] = significant p but negligible effect (likely sample-size artifact at n={n_u+n_k:,})  |  "
            f"[grey50]No[/grey50] = not significant[/grey35]\n"
        )
        any_predictor_sig = bool(pred_df["significant"].any())
        p_only_count = int((pred_df["statistically_significant"] & ~pred_df["significant"]).sum())

        if any_predictor_sig:
            sig_preds = ", ".join(pred_df.loc[pred_df["significant"], "predictor"])
            grid.add_row(f"[orange3]MAR-consistent:[/orange3] missingness in '{col}' relates to: [bold]{sig_preds}[/bold]")
        else:
            grid.add_row("[grey50]No tested predictor explains the missingness pattern with a non-negligible effect size.[/grey50]")
        if p_only_count:
            grid.add_row(f"[grey35]({p_only_count} predictor(s) hit p < {alpha} but effect size < 0.1 -- "
                          f"likely noise given the large sample size, not flagged as MAR.)[/grey35]")

    # ---------------- Combined verdict ----------------
    grid.add_row("\n" + "─" * 58)
    if p_val < alpha and not any_predictor_sig:
        grid.add_row(
            "[bold red]Combined read:[/bold red] outcome-related missingness with no observed-variable "
            "explanation -- cannot rule out MNAR. Requires domain judgment, not just more tests."
        )
    elif p_val < alpha and any_predictor_sig:
        grid.add_row(
            "[bold orange3]Combined read:[/bold orange3] outcome-related missingness that is plausibly "
            "explained by an observed predictor -- MAR-consistent. Conditioning/imputing on that "
            "predictor may resolve the bias."
        )
    else:
        grid.add_row("[bold green]Combined read:[/bold green] no strong evidence against MCAR for this column.")

    panel = Panel(
        grid,
        title=f"[bold {result_color}]⚙ Informative Missingness Log ({txt}): {col}[/bold {result_color}]",
        title_align="left",
        border_style=result_color,
        padding=(1, 2),
        expand=False,
    )
    console.print(panel)


def check_informative_missingness(
    data,
    col,
    txt='',
    target='TransplantSurvivalDay',
    unknown_val=None,
    predictors=None,
    alpha=0.05,
    correction='bonferroni',  # None, 'bonferroni', or 'fdr_bh'
    effect_threshold=0.1,
    min_category_count=5,
    max_categories=10,
):
    """
    Two-part missingness diagnostic for one or more columns, with
    multiple-comparison correction applied at BOTH levels:

    Part 1 (outcome panel): does the OUTCOME differ between Known vs
    Missing/Unknown groups? When `col` is a list, the Part 1 p-values are
    corrected as a FAMILY across all columns tested in this call --
    testing 10 columns at raw alpha=0.05 each gives a ~40% chance of at
    least one false positive, so this matters as soon as you loop over
    more than a couple of columns.

    Part 2 (predictor panel): does missingness relate to OTHER observed
    columns? Corrected as its own family, separately, per column (the
    predictors tested for column A are a different family than those for
    column B).

    Parameters
    ----------
    predictors : list-like or None
        Columns to test against the missingness indicator in Part 2.
        Defaults to all other columns in `data` (excluding `col` and
        `target`).
    alpha : float
        Significance threshold applied to ADJUSTED p-values.
    correction : str or None
        'bonferroni', 'fdr_bh', or None (no correction, raw p-values).
        Applied independently to the Part 1 family and each column's
        Part 2 family.
    effect_threshold : float
        Minimum |Cramer's V| / |rank-biserial r| required (on top of
        p_adj < alpha) for a Part 2 predictor to be flagged "MAR" rather
        than "p-only". At large n, p-values alone flag trivial
        associations as "significant" -- this keeps the MAR-consistent
        verdict tied to a non-negligible effect, not just sample size.
    min_category_count : int
        Categorical predictor levels with fewer than this many observations
        get collapsed into "Other" before the chi-square test (default 5,
        matching the usual expected-cell-count rule of thumb).
    max_categories : int
        Hard cap on distinct categories kept per categorical predictor
        (default 10); the rest are collapsed into "Other" regardless of
        count. Raise this for a predictor where you want full resolution
        (e.g. all 50 states) at the cost of a less reliable chi-square test
        if n is small relative to the category count.
    """
    cols = list(col) if isinstance(col, (list, tuple)) else [col]

    # ---- Part 1: compute all outcome stats first, then correct as a family ----
    raw_results = [_compute_outcome_stats(data, c, target, unknown_val) for c in cols]
    valid_results = [r for r in raw_results if not r["insufficient"]]

    if valid_results:
        p_raw_array = np.array([r["p_raw"] for r in valid_results])
        p_adj_array = _adjust_pvalues(p_raw_array, correction)
        for r, p_adj in zip(valid_results, p_adj_array):
            r["p_adj"] = p_adj

    family_note = None
    if len(cols) > 1 and correction is not None and valid_results:
        family_note = f"Family-wise correction: {correction} across {len(valid_results)} columns"

    for r in raw_results:
        _render_panel(data, r, target, txt, predictors, alpha, correction,
                      effect_threshold=effect_threshold, min_category_count=min_category_count,
                      max_categories=max_categories, family_note=family_note)
