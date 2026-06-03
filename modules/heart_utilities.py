import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.box import ROUNDED
from rich.text import Text


# -------------------------------------------------------------------------------------------
# 1: Row Removal with Masking
# Provides a robust, efficient function to remove rows based on a boolean mask, with
# Rich console output summarizing the operation and its impact on the dataset.
# -------------------------------------------------------------------------------------------
def remove_row_using_mask(
    data: pd.DataFrame,
    remove_col_lst: list,
    colstr: str,
    string: str = '',
    display: bool = True
) -> pd.DataFrame:
    # Initialize the Rich console for output
    console = Console()

    # Initial row count
    pRow = data.shape[0]

    # Optimized boolean mask
    mask = ~data[colstr].isin(remove_col_lst)

    # Apply mask (shallow copy)
    data = data.loc[mask].copy(deep=False).reset_index(drop=True)

    # Remaining rows
    cRow = data.shape[0]
    removed_count = pRow - cRow

    # Stylish status panel
    if display:
        if string == '':
            msg = (
                f"[grey27]Removed "
                f"[bold red]{removed_count}[/bold red] row(s) based on filter applied to "
                f"[cyan]{colstr}[/cyan].[/grey27]"
            )
        else:
            msg = (
                f"[grey27]Removed "
                f"[bold red]{removed_count}[/bold red] row(s) from "
                f"[cyan]{string}[/cyan] dataset.[/grey27]"
            )

        # Display the message in a styled panel
        console.print(
            Panel(
                msg,
                title="[bold cyan]Filter Applied[/bold cyan]",
                border_style="cyan",
                padding=(0,0),
                expand=False
            )
        )

    return data


# -------------------------------------------------------------------------------------------
# 2: Missing Data Summary
# Provides a concise summary of columns with missing data, including counts and percentages,
# and highlights those exceeding a specified threshold with a stylish Rich table.
# -------------------------------------------------------------------------------------------
def percentage_null(df: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
    """
    Vectorized missing-data profiler with modern Rich UI output.
    Fully optimized for PyArrow backends.
    Sorted by missingness (descending).
    """
    console = Console()
    total_rows = len(df)

    # Edge case: empty DataFrame
    if total_rows == 0:
        console.print(
            Panel(
                "[bold red]✕ Error[/bold red]\n"
                "[grey27]The DataFrame contains 0 rows.[/grey27]",
                border_style="red",
                padding=(0, 1),
                expand=False
            )
        )
        return pd.DataFrame(columns=["Feature", "NaNCount", "percentage", "above_threshold"])

    # Vectorized missingness
    nan_counts = df.isna().sum()
    percentages = nan_counts * (100.0 / total_rows)

    # Build metrics table
    metrics = pd.DataFrame({
        "Feature": nan_counts.index,
        "NaNCount": nan_counts.values,
        "percentage": percentages.values,
    })
    metrics["above_threshold"] = metrics["percentage"] >= threshold

    # Filter + sort (DESCENDING by missingness)
    filtered = (
        metrics[metrics["above_threshold"]]
        .sort_values("percentage", ascending=False)
        .reset_index(drop=True)
    )

    # UI output
    if filtered.empty:
        console.print(
            Panel(
                f"[bold green]✔ Compliance Check Passed[/bold green]\n"
                f"[grey27]No features exceed the [/grey27][bold cyan]{threshold}%[/bold cyan][grey27] limit.[/grey27]",
                border_style="green",
                padding=(0, 1),
                expand=False
            )
        )
    else:
        table = Table(
            title=f"[bold red]⚠ High Missingness Alert (≥ {threshold}%)[/bold red]\n",
            header_style="bold white",
            show_header=True,
            box=None,
            padding=(0, 2),
            collapse_padding=True
        )

        table.add_column("Feature", justify="left")
        table.add_column("Null Count", justify="right")
        table.add_column("Percentage", justify="right")

        # Populate rows
        for f, n, p in zip(filtered["Feature"], filtered["NaNCount"], filtered["percentage"]):
            table.add_row(str(f), f"{n:,}", f"{p:.2f}%")

        console.print(table)

    return filtered


# -------------------------------------------------------------------------------------------
# 3: Remove Columns with Feedback
# Provides a function to remove specified columns from a DataFrame, with a stylish Rich panel
# summarizing the operation, including which columns were removed, which were not found, and
# the change in dataset shape.
# -------------------------------------------------------------------------------------------
def remove_column(df: pd.DataFrame, cols, display: bool = True) -> pd.DataFrame:
    """
    Remove one or more columns from a DataFrame with modern Rich UI feedback.
    Always displays feature names (strings), never numeric indices.
    """
    console = Console()

    # Normalize input → always a list of strings
    if isinstance(cols, (str, int)):
        cols = [str(cols)]
    elif hasattr(cols, "tolist"):
        cols = [str(x) for x in cols.tolist()]
    else:
        cols = [str(x) for x in cols]

    colset = set(df.columns)

    # Membership checks
    existing = sorted([c for c in cols if c in colset])
    missing  = sorted([c for c in cols if c not in colset])

    before_rows, before_cols = df.shape

    # Drop only existing columns
    if existing:
        df = df.drop(columns=existing)

    after_rows, after_cols = df.shape

    # -----------------------------
    # Modern Rich UI Output
    # -----------------------------
    if display:
        # Left panel: action summary
        lines = []

        if existing:
            fmt_existing = ", ".join(existing)   # plain black text
            lines.append(f"[bold white]Removed:[/bold white]\n  {fmt_existing}\n")

        if missing:
            fmt_missing = ", ".join(missing)
            lines.append(f"[bold grey50]Skipped (Not Found):[/bold grey50]\n  {fmt_missing}")

        if not lines:
            lines.append("[grey50]No matching column names found.[/grey50]")

        left_panel = Panel(
            "\n".join(lines),
            title="[bold cyan]Schema Update[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=False
        )

        # Right panel: shape delta
        table = Table(
            box=None,
            show_header=True,
            header_style="bold white",
            padding=(0, 2)
        )

        table.add_column("State", style="grey50")
        table.add_column("Rows", justify="right", style="cyan")
        table.add_column("Columns", justify="right", style="magenta")

        table.add_row("Before", f"{before_rows:,}", f"{before_cols:,}")
        table.add_row("After",  f"{after_rows:,}",  f"{after_cols:,}")

        right_panel = Panel(
            table,
            title="[bold cyan]Footprint Delta[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=False
        )

        console.print(Columns([left_panel, right_panel], equal=False, expand=False))

    return df

# -------------------------------------------------------------------------------------------
# 4: Convert Columns to Categorical
# Provides a function to convert specified columns to categorical dtype
# -------------------------------------------------------------------------------------------
def convert_to_category(data: pd.DataFrame, 
                        cols: list, ordered=False) -> pd.DataFrame:
    """
    Convert one or more DataFrame columns to (optionally ordered) categorical dtype.
    """
    # Normalize input → always a list
    if isinstance(cols, (str, int)):
        cols = [cols]
    else:
        cols = list(cols)

    cat_type = pd.CategoricalDtype(ordered=ordered)

    for col in cols:
        data[col] = data[col].astype(cat_type)

    return data


# -------------------------------------------------------------------------------------------
# 5: Display Feature & Data Dictionary Information
# -------------------------------------------------------------------------------------------
def feature_information(
    data: pd.DataFrame,
    datadict: pd.DataFrame,
    strCol: str,
    unique: bool = False,
    indexView: bool = True
):
    # Establish a clean workspace baseline
    console = Console(width=160)

    # 5-1. Feature filtering
    feature = data.columns[data.columns.str.contains(strCol, case=False)].tolist()
    idx = datadict[datadict.Feature.isin(feature)].index

    # 5-1a. No matches → clean error panel
    if not feature:
        console.print(
            Panel(
                f"[bold red]No matching features found[/bold red]\n"
                f"[grey27]Search term:[/grey27] [yellow]{strCol}[/yellow]",
                border_style="red",
                padding=(0, 1),
                expand=False  # Box hugs error text
            )
        )
        return [], None

    # 5-2. Descriptive statistics table
    stats_df = data[feature].describe(include="all").T

    stats_table = Table(
        title="[bold cyan]Descriptive Statistics[/bold cyan]",
        title_justify="left",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
        collapse_padding=True,
        expand=False  # Table stays as compact as its data
    )

    # Dynamic column setup based on describe() output, always showing 'Feature' if indexView is True
    if indexView:
        stats_table.add_column("Feature", justify="left", style="white", no_wrap=True)
    
    # Add columns for all available metrics in the describe() output
    for col in stats_df.columns:
        stats_table.add_column(col, justify="right", style="grey27")

    # Populate rows with dynamic formatting, handling NaN values and applying consistent styling
    for feat, row in stats_df.iterrows():
        row_vals = []
        if indexView:
            row_vals.append(f"[white]{feat}[/white]")

        # Format each value based on its type and presence, applying a consistent color scheme and handling NaNs gracefully
        for val in row.values:
            if pd.isna(val):
                row_vals.append("[grey35]–[/grey35]")
            elif isinstance(val, float):
                row_vals.append(f"[grey27]{val:.2f}[/grey27]")
            else:
                row_vals.append(f"[grey27]{val}[/grey27]")

        # Add the formatted row to the table
        stats_table.add_row(*row_vals)

    console.print(stats_table)

    # 5-3. Metadata panel -> NOW DYNAMIC
    meta_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 3), # Adds balanced spacing between dynamic columns
        collapse_padding=True,
        expand=False  # Crucial: tightens the internal structure
    )

    # Removed 'ratio' arguments so columns shrink to fit content
    meta_table.add_column("Feature", style="white", no_wrap=True)
    meta_table.add_column("DataType", style="green")
    meta_table.add_column("NaNs Count", justify="right", style="red")

    # Calculate NaN counts and dtypes for the relevant features
    nan_counts = data[feature].isna().sum()
    dtypes = data[feature].dtypes

    # Populate the metadata table with dynamic values, applying consistent styling and formatting
    for f in feature:
        meta_table.add_row(
            f"[white]{f}[/white]",
            f"[grey27]{dtypes[f]}[/grey27]",
            f"[grey27]{nan_counts[f]:,}[/grey27]"
        )

    # Render the metadata panel with a dynamic title and tight layout that adjusts to the number of features displayed
    console.print(
        Panel(
            meta_table,
            title="[bold cyan]Feature Metadata[/bold cyan]",
            title_align="left",
            border_style="grey27",
            padding=(0, 1),
            expand=False  # Crucial: Forces the panel box to hug the table tightly
        )
    )

    # 5-4. Data dictionary table
    dict_table = Table(
        title="[bold cyan]Data Dictionary[/bold cyan]",
        title_justify="left",
        show_header=True,
        header_style="bold cyan", 
        box=None,
        padding=(0, 1),          
        collapse_padding=True,
        expand=True  # Keeps this true so your descriptions have room to wrap nicely            
    )

    # Centralized column specification for the data dictionary, with dynamic inclusion based on actual availability in the DataFrame
    dict_cols = ["Feature", "Description", "FormSection", "DataType", "SASAnalysisFormat", "Comment", "Information"]
    available_cols = [c for c in dict_cols if c in datadict.columns]
    filtered_dict = datadict.loc[datadict.Feature.isin(feature), available_cols]

    # Dynamic column setup based on availability, with consistent styling and formatting rules applied to each column 
    # type for a cohesive look
    if indexView:
        dict_table.add_column("Idx", justify="left", style="grey35", width=5)
   
    if "Feature" in available_cols:
        dict_table.add_column("Feature", justify="left", style="white", ratio=3, no_wrap=True)
    if "Description" in available_cols:
        dict_table.add_column("Description", justify="left", style="grey27", ratio=6, no_wrap=False)
    if "FormSection" in available_cols:
        dict_table.add_column("FormSection", justify="left", style="grey27", ratio=3, no_wrap=False)
    if "DataType" in available_cols:
        dict_table.add_column("DataType", justify="left", style="grey27", ratio=2)
    if "SASAnalysisFormat" in available_cols:
        dict_table.add_column("SASAnalysisFormat", justify="left", style="grey27", ratio=3, no_wrap=True)
    if "Comment" in available_cols:
        dict_table.add_column("Comment", justify="left", style="grey27", ratio=2)
    if "Information" in available_cols:
        dict_table.add_column("Information", justify="left", style="grey27", ratio=2)

    # Populate the data dictionary table with dynamic values, applying consistent formatting and handling 
    # missing values gracefully, while also applying a cohesive color scheme to differentiate between feature 
    # names and their associated metadata   
    for i, row in filtered_dict.iterrows():
        row_vals = []
        if indexView:
            row_vals.append(f"[grey35]{i}[/grey35]")
        # Iterate through available columns in a consistent order, applying formatting rules based on 
        # column type and value presence    
        for col_name in available_cols:
            val = row[col_name]
            if pd.notna(val) and str(val).strip() != "":
                clean_val = " ".join(str(val).split())
                if col_name == "Feature":
                    row_vals.append(f"[white]{clean_val}[/white]")
                else:
                    row_vals.append(f"[grey27]{clean_val}[/grey27]")
            else:
                row_vals.append("[grey35]–[/grey35]")

        # Add the formatted row to the data dictionary table
        dict_table.add_row(*row_vals)

    console.print(dict_table)

    # 5-5. Unique values panel -> NOW DYNAMIC
    if unique:
        unique_table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            collapse_padding=True,
            expand=False  # Tightens the internal layout
        )
        unique_table.add_column("Feature", style="cyan", no_wrap=True)
        unique_table.add_column("Values", style="grey27", no_wrap=False)
        # Iterate through each feature and extract unique values, applying dynamic truncation for features 
        # with many unique values to maintain readability, while also applying consistent styling to differentiate 
        # between feature names and their unique value previews   
        for f in feature:
            vals = data[f].dropna().unique()
            preview = ", ".join([str(v) for v in vals[:10]])
            if len(vals) > 10:
                preview += f" … (+{len(vals)-10} more)"
            
            unique_table.add_row(f, preview)

        console.print(
            Panel(
                unique_table,
                title="[bold cyan]Unique Values[/bold cyan]",
                title_align="left",
                border_style="grey27",
                padding=(0, 1),
                expand=False  # Crucial: Forces the panel box to hug unique list length
            )
        )

    return feature, idx


# -------------------------------------------------------------------------------------------
# 6: Categorical Feature with Mapping Update
# -------------------------------------------------------------------------------------------
def mapping_data_and_dictionary(
    data: pd.DataFrame, 
    dataDict: pd.DataFrame, 
    mapping: dict, 
    indices: list, 
    feature_info: str = '', 
    txt: str = ''
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simultaneously renames dataset features and dynamically synchronizes, 
    updates, and documents the accompanying data dictionary metadata.
    """
    # 6-1: Standard defensive copy check for sliced DataFrame structures
    if hasattr(data, '_is_copy') and data._is_copy is not None:
        data = data.copy()
    if hasattr(dataDict, '_is_copy') and dataDict._is_copy is not None:
        dataDict = dataDict.copy()

    # 6-2: Execute column transformation on the primary dataset
    data = data.rename(columns=mapping)

    # 6-3: Synchronize data dictionary feature names safely
    # (Using .replace() preserves the original index alignment better than .map().fillna())
    dataDict['Feature'] = dataDict['Feature'].replace(mapping)

    # 6-4: Mutate targeted metadata records using explicit location indexing
    # Ensures indices exist to prevent pandas from inadvertently creating junk outer-join rows
    valid_indices = [idx for idx in indices if idx in dataDict.index]
    if valid_indices:
        dataDict.loc[valid_indices, 'Information'] = str(txt)
        dataDict.loc[valid_indices, 'FeatureType'] = str(feature_info)

    # 6-5: Slick, Low-Contrast Logging UI
    console = Console()
    mapped_count = len(mapping)
    meta_count = len(valid_indices)
    
    # Generate visual change string for renamed elements
    changes = "  ".join([f"[cyan]• {old} ➔ {new}[/cyan]" for old, new in mapping.items()])
    
    # Final summary panel with clear mutation counts and a dynamic list of changes, 
    # or a fallback message if no transformations were applied
    console.print(
        Panel(
            f"[bold white]Metadata Synchronization Pipeline Completed[/bold white]\n"
            f"[grey35]Dataset Mutations:    [/grey35][bold green]{mapped_count} columns updated[/bold green]\n"
            f"[grey35]Dictionary Mutations: [/grey35][bold green]{meta_count} rows annotated[/bold green]\n\n"
            f"[bold white]Active Naming Mapping Tracked:[/bold white]\n"
            f"{changes if changes else '[grey35]No column transformations applied.[/grey35]'}",
            title="[bold grey27]⚙ Metadata Synchronization Log[/bold grey27]",
            title_align="left",
            border_style="grey27",
            padding=(1, 2),
            expand=False
        )
    )

    return data, dataDict



# -------------------------------------------------------------------------------------------
# 7: Feature Info Dashboard
# Provides a compact, visually appealing summary of features matching a substring, including
# summary statistics and optional categorical diagnostics.
# --------------------------------------------------------------------------------------------
def get_feature_info(data: pd.DataFrame, colstr: str, 
                     cat: bool = False, max_cats: Optional[int] = None) -> List[str]:
    """
    Search for DataFrame columns matching a substring and display an ultra-compact,
    space-optimized Rich dashboard containing summary stats and categorical diagnostic grids.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to inspect.
    colstr : str
        Case‑insensitive substring used to match column names.
    cat : bool, default False
        If True, displays categorical inspection cards.
    max_cats : int or None, default None
        The maximum number of categories to display in the preview string before 
        truncating. Set to None to list ALL categories without restriction.
    """
    # 7-0: Initialize the Rich console for output
    console = Console()

    # Case‑insensitive substring match across all columns
    features = sorted(
        data.columns[data.columns.str.contains(colstr, case=False, na=False)].tolist()
    )

    if not features:
        console.print(
            Panel(
                f"[bold red]✕ No matching features found for:[/bold red] [yellow]{colstr}[/yellow]",
                box=ROUNDED, border_style="red", expand=False
            )
        )
        return []

    # 7-1: Isolate target features first to maximize efficiency
    target_df = data[features]
    desc_df = target_df.describe(include="all").T
    nan_counts = target_df.isna().sum()

    # 7-2: Consolidated Title Header
    title_banner = Text.from_markup(
        f"[bold cyan]✦ Feature Profiler[/bold cyan] [dim]│ Match: '{colstr}' │ Found: {len(features)}[/dim]"
    )
    title_banner.justify = "left"

    # 7-3: Create a unified table for all features with dynamic metrics based on availability
    stats_table = Table(
        show_header=True,
        header_style="bold magenta",
        title=title_banner,
        box=ROUNDED,
        border_style="dim",
        padding=(0, 1),
        expand=False
    )

    # 7-4: Dynamic Column Setup: Always show 'Missing', then add available metrics from describe()
    stats_table.add_column("Feature", justify="left", style="bold white")
    stats_table.add_column("Missing", justify="right", style="red")

    # 7-5: Define the order of metrics to display if they exist in the describe() output
    all_possible_metrics = ["unique", "top", "freq", "mean", "std", "min", "25%", "50%", "75%", "max"]
    metrics = [m for m in all_possible_metrics if m in desc_df.columns]

    # 7-6: Add columns for each available metric
    for m in metrics:
        stats_table.add_column(m.capitalize(), justify="right", style="cyan")

    # 7-7: Populate the table rows with feature statistics, applying formatting and handling NaN values
    for col_name, row in desc_df.iterrows():
        row_vals = [str(nan_counts[col_name])]
        for m in metrics:
            val = row[m]
            if pd.isna(val):
                row_vals.append("[dim]–[/dim]")
            elif isinstance(val, (int, float)):
                if float(val).is_integer():
                    row_vals.append(f"{int(val)}")
                else:
                    row_vals.append(f"{val:.2f}")
            else:
                row_vals.append(str(val))
        stats_table.add_row(col_name, *row_vals)

    # 7-8: Render the consolidated stats table
    console.print(stats_table)

    # 7-9: Optional categorical deep‑dive card matrix
    if cat:
        cat_cards = []

        # Iterate through each feature to determine if it's categorical and 
        # create a corresponding card with dynamic truncation of categories
        for col in features:
            series = target_df[col]
            dtype_str = str(series.dtype)
            is_cat = dtype_str.startswith("category")

            # For categorical features, extract categories and ordered status, then apply dynamic truncation logic
            if is_cat:
                cats = list(series.cat.categories)
                ordered = series.cat.ordered
                ord_str = "[g]True[/g]" if ordered else "[r]False[/r]"
                
                # Dynamic Truncation Logic based on max_cats parameter value
                if max_cats is None or len(cats) <= max_cats:
                    preview = str(cats)
                else:
                    preview = f"{cats[:max_cats]}.. (+{len(cats)-max_cats})"
                
                inline_meta = f"[dim]type:[/dim][y]{dtype_str}[/y] [dim]│ cats:[/dim][g]{preview}[/g] [dim]│ ord:[/dim]{ord_str}"
                border = "green"
            else:
                inline_meta = "[dim]type:[/dim] [y]numeric[/y] [dim]│[/dim] [red]Non-Categorical[/red]"
                border = "dim"

            # Apply color coding to the inline metadata string for enhanced visual distinction
            inline_meta = inline_meta.replace("[y]", "[yellow]").replace("[/y]", "[/yellow]")
            inline_meta = inline_meta.replace("[g]", "[green]").replace("[/g]", "[/green]")
            inline_meta = inline_meta.replace("[r]", "[red]").replace("[/r]", "[/red]")
            
            # 1-10: Create a Rich Panel for each feature with the formatted metadata and appropriate styling
            card = Panel(
                inline_meta,
                title=f"[white]{col}[/white]",
                border_style=border,
                box=ROUNDED,
                expand=False
            )
            cat_cards.append(card)

        # 7-11: Render the categorical cards in a compact column layout with minimal padding
        console.print(Columns(cat_cards, padding=(0, 1)))
        console.print()

    return features


# -------------------------------------------------------------------------------------------
# 8: Update Dictionary Information
# Provides a function to update the 'Information' and optionally 'FeatureType' fields in a 
# data dictionary for specified feature names, with robust handling of various input formats.
# -------------------------------------------------------------------------------------------
def update_dictionary_information(dataDict, indices, txt='', feature_type=None):
    """
    Update the Information and (optionally) FeatureType fields
    for one or more feature names in a data dictionary.
    """
    # Normalize indices → always a list of strings
    if isinstance(indices, (str, int)):
        indices = [str(indices)]
    elif hasattr(indices, "tolist"):
        indices = [str(x) for x in indices.tolist()]
    else:
        indices = [str(x) for x in indices]

    # Update fields
    dataDict.loc[dataDict["Feature"].isin(indices), "Information"] = txt

    if feature_type is not None:
        dataDict.loc[dataDict["Feature"].isin(indices), "FeatureType"] = feature_type

    return dataDict


# -------------------------------------------------------------------------------------------
# 9: Symmetric Difference Utility
# Provides a simple utility function to compute the sorted symmetric difference between two sets,
# with a stylish console printout of the results.
# -------------------------------------------------------------------------------------------
def symmetric_difference(set1, set2, set1_name="Set A", set2_name="Set B"):
    """
    Display the symmetric difference with clear context showing WHERE the missing values are.
    """
    console = Console()

    # Build an internal table structure with clear columns
    table = Table(
        show_header=True,
        header_style="bold grey50",
        box=None,
        padding=(0, 3),
        expand=False
    )

    table.add_column("Value", justify="left", style="white", min_width=12)
    table.add_column("Found In", justify="left", style="cyan", min_width=12)

    # Track mismatches and their source
    has_mismatch = False
    
    # Sort all unique values across both sets to check them cleanly
    all_values = sorted(set1.union(set2))

    # Compute symmetric difference (returned to caller)
    sym_set = set1.symmetric_difference(set2)
    
    for val in all_values:
        in_set1 = val in set1
        in_set2 = val in set2
        
        # If it's in one but not both, it's a symmetric difference
        if in_set1 != in_set2:
            has_mismatch = True
            source = set1_name if in_set1 else set2_name
            table.add_row(str(val), source)

    if has_mismatch:
        renderable = table
    else:
        renderable = "[grey35]No mismatched values found (Sets are identical)[/grey35]"

    # Wrap in a clean, tight dynamic panel
    panel = Panel(
        renderable,
        title="[bold cyan]Symmetric Difference[/bold cyan]",
        title_align="left",
        border_style="cyan",
        padding=(0, 2),
        expand=False
    )

    console.print(panel)

    # Return the actual symmetric difference set for backend use
    return sym_set


# -------------------------------------------------------------------------------------------
# 10: Mapping Columns with Dictionary and Category Conversion
# -------------------------------------------------------------------------------------------
def mapping_columns(data: pd.DataFrame, colstr: str, mapdict: dict, display: bool = True) -> pd.DataFrame:
    """
    Map values in a specified DataFrame column using a dictionary, preserving 
    unmapped entries, convert to a categorical dtype.
    """
    # 10-1: Safe, explicit defensive copy to prevent SettingWithCopyWarnings
    data = data.copy()

    # 10-2: Convert to object first to lift preexisting category constraints,
    # map values, and cast cleanly back to a category
    data[colstr] = data[colstr].astype(object).replace(mapdict).astype('category')

    # 10-3: Modern Minimalist Terminal Output
    if display:
        console = Console()
        
        # Get unique values cleanly sorted for presentation
        unique_vals = [str(x) for x in sorted(data[colstr].dropna().unique())]
        formatted_vals = "  ".join([f"[cyan]• {val}[/cyan]" for val in unique_vals])
        
        console.print(
            Panel(
                f"[bold white]Column Pipeline Successfully Mutated[/bold white]\n"
                f"[grey35]Target Feature [/grey35][bold cyan]{colstr}[/bold cyan]  [grey35]➔[/grey35]  [bold magenta]category[/bold magenta]\n\n"
                f"[bold white]Unique Categories Established:[/bold white]\n"
                f"{formatted_vals}",
                title="[bold grey27]⚙ Pipeline Log[/bold grey27]",
                title_align="left",
                border_style="grey27",
                padding=(1, 2),
                expand=False
            )
        )

    return data


# -------------------------------------------------------------------------------------------
# 11: Combine Unique Values from Two Columns 
# -------------------------------------------------------------------------------------------
def combine_get_unique(df, col1, col2, nan_val, display=True):
    """
    Combine unique values from two columns, replace missing values safely
    regardless of data types, and display ONLY the clean unique values table.
    No data is returned.
    """
    if not display:
        return

    console = Console()

    # 11-1: Extract unique values cleanly using Pandas 
    u1_raw = df[col1].unique()
    u2_raw = df[col2].unique()

    # 10-2: Safely fill missing values
    s1 = pd.Series(u1_raw).fillna(nan_val)
    s2 = pd.Series(u2_raw).fillna(nan_val)

    def try_int(val):
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val).strip()

    # Apply the try_int function to all values to ensure consistent formatting for numeric strings
    clean_u1 = [try_int(x) for x in s1]
    clean_u2 = [try_int(x) for x in s2]

    # 11-3: Combine + Deduplicate
    combined_set = set(clean_u1).union(set(clean_u2))
    
    try:
        combined = sorted(list(combined_set), key=lambda x: int(x))
    except ValueError:
        combined = sorted(list(combined_set))

    # 11-4: Modern Minimalist Terminal Output (No repeating summary messages)
    table = Table(
        show_header=True,
        header_style="bold grey50",
        box=None,
        padding=(0, 3),
        expand=False
    )

    # Only the unique values are displayed in the table, with a simple index for reference
    table.add_column("Index", justify="right", style="grey50", width=6)
    table.add_column("Merged Unique Value", justify="left", style="white", min_width=20)

    # Populates only the table rows
    for i, val in enumerate(combined, start=1):
        table.add_row(str(i), str(val))

    # Wrapped inside the panel directly with zero extra string text
    panel = Panel(
        table,
        title=f"[bold cyan]Combined Unique Values[/bold cyan]",
        title_align="left",
        border_style="cyan",
        padding=(0, 2),
        expand=False
    )

    console.print(panel)



# -------------------------------------------------------------------------------------------
# 12: Remove Unused Categories from Categorical Columns
# -------------------------------------------------------------------------------------------
def remove_cat_zero_count(data: pd.DataFrame):
    """
    Remove categories that have zero observations from all categorical columns.
    """
    for col in data.select_dtypes(["category"]).columns:
        counts = data[col].value_counts()
        keep = counts[counts > 0].index
        data[col] = data[col].cat.remove_categories(
            [cat for cat in data[col].cat.categories if cat not in keep]
        )
    return data


# -------------------------------------------------------------------------------------------
# 13: Write DataFrame to Disk with Rich Confirmation
# -------------------------------------------------------------------------------------------
def write_to_file(df, filename, path="../data/", format="csv"):
    """
    Write a DataFrame to disk in CSV, Pickle, Feather, or Parquet format.
    Automatically fixes PyArrow conversion issues by converting object columns to string.
    """
    console = Console()
    fmt = format.lower()

    # Fix PyArrow conversion issues: convert object columns to string
    if fmt in ("parquet", "feather"):
        df = df.copy()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype("string")

    # Build full file path
    file_path = f"{path}{filename}.{fmt}"

    # Write file
    if fmt == "csv":
        df.to_csv(file_path, index=False)

    elif fmt in ("pkl", "pickle"):
        df.to_pickle(file_path)

    elif fmt == "feather":
        df.to_feather(file_path)

    elif fmt == "parquet":
        df.to_parquet(file_path, index=False)

    else:
        raise ValueError(f"Unsupported format: {format}")

    # Stylish confirmation panel
    panel = Panel(
        f"[bold cyan]{len(df):,}[/bold cyan] records written to:\n[white]{file_path}[/white]",
        title="[bold green]✔ File Saved[/bold green]",
        border_style="green",
        padding=(1, 2),
        expand=False
    )

    console.print(panel)


# -------------------------------------------------------------------------------------------
# 14: Search Data Dictionary for Specific Features
# -------------------------------------------------------------------------------------------
def dictionary_search(datadic: pd.DataFrame, colList: list, indexView: bool = True):
    """
    Search for specific column features inside the data dictionary.
    Clean, optimized, and dead‑code‑free with modern Rich UI formatting.
    """
    console = Console(width=160)

    # Columns we care about (keep only those that exist)
    dict_cols = [
        "Feature", "Description", "FormSection", "DataType",
        "SASAnalysisFormat", "Comment", "Information"
    ]
    available_cols = [c for c in dict_cols if c in datadic.columns]

    # Filter rows matching requested features
    filtered = datadic.loc[datadic["Feature"].isin(colList), available_cols]

    # No matches → clean error panel
    if filtered.empty:
        console.print(
            Panel(
                "[bold red]✕ No Matches Found[/bold red]\n"
                "[grey27]None of the provided features exist in the dictionary.[/grey27]",
                border_style="red",
                padding=(0, 1),
                expand=False
            )
        )
        return

    # -----------------------------
    # Build Rich Table
    # -----------------------------
    table = Table(
        title="[bold cyan]Data Dictionary Lookup[/bold cyan]\n",
        show_header=True,
        header_style="bold white",
        box=None,
        padding=(0, 2),
        collapse_padding=True,
        expand=False
    )

    # Optional index column
    if indexView:
        table.add_column("Index", justify="center", style="grey27", max_width=8)

    # Column formatting rules (clean + centralized)
    col_specs = {
        "Feature":            dict(justify="left",  style="cyan",   min_width=20, max_width=25, no_wrap=True),
        "Description":        dict(justify="left",  style="grey27", max_width=45),
        "FormSection":        dict(justify="left",  style="grey27", min_width=13, max_width=15),
        "DataType":           dict(justify="left",  style="grey27", min_width=10, max_width=12),
        "SASAnalysisFormat":  dict(justify="left",  style="grey27", min_width=18, max_width=20),
        "Comment":            dict(justify="left",  style="grey27", min_width=10, max_width=15),
        "Information":        dict(justify="left",  style="grey27", min_width=15, max_width=40),
    }

    # Add columns with rules
    for col in filtered.columns:
        spec = col_specs.get(col, dict(justify="left", style="grey27", max_width=15))
        table.add_column(col, **spec)

    # -----------------------------
    # Populate rows (vectorized clean formatting)
    # -----------------------------
    for idx, row in filtered.iterrows():
            row_vals = []

            if indexView:
                row_vals.append(str(idx))   # plain black

            for val in row:
                if pd.notna(val) and str(val).strip():
                    clean = " ".join(str(val).split())
                    row_vals.append(clean)   # plain black text
                else:
                    row_vals.append("–")     # plain black dash

            table.add_row(*row_vals)

    console.print(table)


# -------------------------------------------------------------------------------------------
# 15: Identify and Display Columns with Null Values
# -------------------------------------------------------------------------------------------
def any_nans(data: pd.DataFrame, txt: str = "") -> None:
    """
    Display a modern Rich UI summary of columns containing NaN values.
    """
    console = Console()
    total_rows = len(data)

    # Vectorized null counts (fast on PyArrow)
    null_counts = data.isna().sum()
    null_counts = null_counts[null_counts > 0]

    # No missing values → clean success panel
    if null_counts.empty:
        console.print(
            Panel(
                f"[bold green]✔ Clean Dataset[/bold green]\n"
                f"[grey50]No missing values across {total_rows:,} rows.[/grey50]",
                border_style="green",
                padding=(1, 2),
                expand=False
            )
        )
        return

    # Compute percentages
    percent = (null_counts / total_rows) * 100

    # Build sorted summary
    summary = (
        pd.DataFrame({"Count": null_counts, "Percentage": percent})
        .sort_values("Count", ascending=False)
    )

    # Build Rich table - removed the 'title=' parameter to fix column shifting
    table = Table(
        show_header=True,
        header_style="bold grey50",
        box=None,
        padding=(0, 3), # Clean, clear grid spacing
        expand=False
    )

    # Added explicit text justification rules and locking sizes
    table.add_column("Column", style="cyan", justify="left", min_width=25)
    table.add_column("Null Count", style="magenta", justify="right", width=12)
    table.add_column("Percentage", style="yellow", justify="right", width=12)

    for col, row in summary.iterrows():
        # Added explicit typecasting to handle integers cleanly without floating point anomalies (.0)
        table.add_row(
            str(col),
            f"{int(row['Count']):,}",
            f"{row['Percentage']:.4f}%"
        )

    # Construct clean internal layout strings
    header_title = f"[bold red]⚠ Missing Values Found {f'({txt})' if txt else ''}[/bold red]"
    
    content = Table.grid()
    content.add_row(header_title)
    content.add_row("─" * 55) # Clean visual separator line
    content.add_row(table)

    # Wrap in a polished panel with explicit box parameters
    console.print(
        Panel(
            content,
            title=f"[bold yellow]Total Rows: {total_rows:,}[/bold yellow]",
            title_align="center",
            border_style="red",
            padding=(0, 2),
            expand=False # Tells the red border to perfectly hug your columns
        )
    )


# -------------------------------------------------------------------------------------------
# 16: Check for Informative Missingness with Statistical Testing and Effect Size
# (Handles both single columns and lists of columns with clean iteration)
# -------------------------------------------------------------------------------------------
def check_informative_missingness(data, col, txt='', target='TransplantSurvivalDay', unknown_val=None):
    """
    Compare survival outcomes between Known vs Missing/Unknown groups for one or more columns.
    Computes Welch's t-test, Cohen's d, and 95% CI for effect size. Displays a polished Rich UI report.
    """
    # If col is a list, iterate cleanly
    if isinstance(col, (list, tuple)):
        for c in col:
            check_informative_missingness(data, c, txt=txt, target=target, unknown_val=unknown_val)
        return

    # Single-column logic with robust handling of unknown value definitions and missing data
    if unknown_val is not None:
        is_unknown = (data[col] == unknown_val) | (data[col].isna())
    else:
        is_unknown = data[col].isna()

    unknown = data.loc[is_unknown, target].dropna()
    known   = data.loc[~is_unknown, target].dropna()

    console = Console()

    # Handle insufficient data gracefully inside a warning panel
    if len(unknown) < 2 or len(known) < 2:
        console.print(
            Panel(
                f"[bold yellow]⚠ Insufficient Sample Size[/bold yellow]\n"
                f"[grey50]Feature '[bold cyan]{col}[/bold cyan]' requires at least 2 entries per group to run Welch's t-test.[/grey50]",
                title=f"[bold grey27]⚙ Pipeline Warning: {col}[/bold grey27]",
                title_align="left",
                border_style="yellow",
                expand=False
            )
        )
        return

    # Group stats
    n_u, n_k = len(unknown), len(known)
    m_u, m_k = unknown.mean(), known.mean()
    var_u, var_k = unknown.var(ddof=1), known.var(ddof=1)

    # Cohen's d calculation with pooled standard deviation
    dof = n_u + n_k - 2
    pooled_std = np.sqrt(((n_u - 1) * var_u + (n_k - 1) * var_k) / dof)
    d = (m_u - m_k) / pooled_std if pooled_std != 0 else 0

    # 95% CI for d
    se_d = np.sqrt((n_u + n_k) / (n_u * n_k) + (d**2) / (2 * (n_u + n_k)))
    z = stats.norm.ppf(0.975)
    lower, upper = d - z * se_d, d + z * se_d

    # Effect size interpretation
    abs_d = abs(d)
    if abs_d < 0.2:
        strength = "Negligible"
    elif abs_d < 0.5:
        strength = "Small"
    elif abs_d < 0.8:
        strength = "Medium"
    else:
        strength = "Large"

    # Welch's t-test computation
    t_stat, p_val = stats.ttest_ind(unknown, known, equal_var=False)

    # Determine structural outcomes and messaging
    if p_val < 0.05:
        result_color = "red"
        result_badge = "[bold red]⚠ INFORMATIVE MISSINGNESS DETECTED[/bold red]"
        interpretation = "[orange3]Statistical Significance:[/orange3] Not MCAR. Likely MAR or MNAR (Missingness pattern affects target profile)."
    else:
        result_color = "green"
        result_badge = "[bold green]✔ RANDOM MISSINGNESS DETECTED[/bold green]"
        interpretation = "[grey50]Statistical Significance: Consistent with MCAR (Missing Completely at Random).[/grey50]"

    # Build Aligned Layout
    grid = Table.grid(expand=False)
    
    # Updated Structured Cohort Comparison Table
    stats_table = Table(box=None, padding=(0, 2), show_header=True, header_style="bold grey50")
    stats_table.add_column("Cohort Group", min_width=18, justify="left")
    stats_table.add_column("Group Sizes", min_width=18, justify="right", style="magenta")
    stats_table.add_column("Mean Survival", min_width=20, justify="right", style="yellow")
    
    # Dynamically injects clean formatted strings matching your layout targets
    stats_table.add_row("Unknown / Missing", f"Unknown={int(n_u):,}", f"{m_u:,.1f}d")
    stats_table.add_row("Known / Complete", f"Known={int(n_k):,}", f"{m_k:,.1f}d")
    
    # Add Content Rows to main layout grid
    grid.add_row(f"[bold white]Cohort Overview[/bold white] {f'[grey35]({txt})[/grey35]' if txt else ''}")
    grid.add_row(stats_table)
    grid.add_row(f"[grey35]Difference:[/grey35]            [bold white]{m_u - m_k:+,.1f} days[/bold white]\n")
    
    grid.add_row("─" * 58)
    grid.add_row("\n[bold white]Welch's t-test Analysis[/bold white]")
    grid.add_row(f"[grey35]ρ-Value:[/grey35]               [bold cyan]{p_val:.4f}[/bold cyan]\n")
    
    grid.add_row(result_badge)
    grid.add_row(interpretation)
    grid.add_row("\n" + "─" * 58)
    
    grid.add_row("\n[bold white]Cohen's Analysis[/bold white]")
    grid.add_row(f"[grey35]Cohen's d:[/grey35]             [bold cyan]{d:.4f}[/bold cyan] ({strength})")
    grid.add_row(f"[grey35]95% CI:[/grey35]                [bold cyan][{lower:.4f}, {upper:.4f}][/bold cyan]\n")
    
    if lower > 0 or upper < 0:
        grid.add_row("[bold red]Result:[/bold red] INFORMATIVE MISSINGNESS (statistically significant difference)")
    else:
        grid.add_row("[bold green]Result:[/bold green] Likely Random Missingness (Effect size not significant)")

    # Outer Panel wrapper to prevent border clipping
    panel = Panel(
        grid,
        title=f"[bold {result_color}]⚙ Informative Missingness Log ({txt}): {col}[/bold {result_color}]",
        title_align="left",
        border_style=result_color,
        padding=(1, 2),
        expand=False
    )

    console.print(panel)
    
    # Suppress notebook trailing outputs completely
    pass


# -------------------------------------------------------------------------------------------
# 17: Get Column Summary Based on Unique Value Counts with Rich UI Display
# -------------------------------------------------------------------------------------------
def get_column_summary(data, cat=2, flag=True, dropna=True, ignore_list=None):
    """
    Categorizes columns based on unique value counts and displays a stylish Rich UI summary.
    """
    console = Console()
    ignore_list = ignore_list or []

    # Select columns based on threshold
    if flag:
        cols = [col for col in data.columns if data[col].nunique(dropna=dropna) > cat]
        condition = f"> {cat}"
    else:
        cols = [col for col in data.columns if data[col].nunique(dropna=dropna) <= cat]
        condition = f"<= {cat}"

    # Remove ignored columns
    cols = [col for col in cols if col not in ignore_list]

    # Build summary dictionary
    summary = {col: data[col].unique().tolist() for col in cols}

    # Build Rich table
    table = Table(
        title=f"[bold cyan]Column Summary (Unique Values {condition})[/bold cyan]",
        header_style="bold white",
        box=None,
        padding=(0, 2),
        collapse_padding=True
    )

    # Added explicit text justification rules and locking sizes for a clean, consistent layout
    table.add_column("Column", style="cyan", justify="left")
    table.add_column("Unique Values (truncated)", justify="left")

    # Populate rows with dynamic truncation for readability
    for col, values in summary.items():
        # Truncate long lists for readability
        display_vals = values[:5]
        val_str = f"{display_vals}..." if len(values) > 5 else f"{display_vals}"
        table.add_row(col, val_str)

    # Wrap in a panel
    panel = Panel(
        table,
        title=f"[bold green]✔ Found {len(cols)} Columns[/bold green]",
        border_style="green",
        padding=(1, 2),
        expand=False
    )

    console.print(panel)

    return cols


# -------------------------------------------------------------------------------------------
# 18: Get Columns Based on Cardinality Threshold
# -------------------------------------------------------------------------------------------
def get_cols_by_cardinality(data, cat, dropna=True, flag=False):
    """
    Return columns based on cardinality threshold.
    
    Args:
        data: DataFrame
        cat: cardinality threshold
        dropna: whether to count NaN as a unique value
        flag: 
            True  -> return columns with cardinality > cat
            False -> return columns with cardinality <= cat
    """
    if flag:
        return [col for col in data.columns if data[col].nunique(dropna=dropna) > cat]
    else:
        return [col for col in data.columns if data[col].nunique(dropna=dropna) <= cat]
    

# -------------------------------------------------------------------------------------------
# 19: Continuous Value Predicts Survival with Correlation Metrics and Rich UI Display
# -------------------------------------------------------------------------------------------
def continuous_value_predicts_survival(data, col, txt='', target='TransplantSurvivalDay'):
    """
    Assess whether a continuous feature has a linear relationship with survival time using Pearson 
    and Spearman correlations. Computes correlation coefficients, p-values, and R² values, and displays 
    a polished Rich UI report summarizing the findings with dynamic significance badges and clear interpretation.
    """
    # If col is a list, iterate cleanly
    if isinstance(col, (list, tuple)):
        for c in col:
            continuous_value_predicts_survival(data, c, txt=txt, target=target)
        return

    # 19-1: Single-column logic with robust handling of missing data and clean statistical testing
    console = Console()

    # Drop missing values across target and feature variables
    df = data[[col, target]].dropna()

    # Handle insufficient data gracefully inside a warning panel
    if len(df) < 3:
        console.print(
            Panel(
                f"[bold yellow]Insufficient data[/bold yellow]\n"
                f"[grey50]Need ≥3 valid rows for '{col}'.[/grey50]",
                title="Correlation Warning",
                border_style="yellow",
                box=ROUNDED,
                padding=(1, 2),
                expand=False
            )
        )
        return

    # 19-1: Compute Pearson and Spearman correlations
    pear_r, pear_p = stats.pearsonr(df[col], df[target])
    spear_r, spear_p = stats.spearmanr(df[col], df[target])

    # 19-2: Build compact internal stats table
    stats_table = Table(
        box=None,
        show_header=True,
        header_style="bold grey50",
        padding=(0, 2)
    )
    # Added explicit text justification rules and locking sizes for a clean, consistent layout
    stats_table.add_column("Metric", justify="left")
    stats_table.add_column("r", justify="right", style="cyan")
    stats_table.add_column("p", justify="right", style="magenta")
    stats_table.add_column("R²", justify="right", style="yellow")

    # Dynamically injects clean formatted strings matching your layout targets
    stats_table.add_row("Pearson", f"{pear_r:+.3f}", f"{pear_p:.2e}", f"{pear_r**2:.3%}")
    stats_table.add_row("Spearman", f"{spear_r:+.3f}", f"{spear_p:.2e}", f"{spear_r**2:.3%}")

    # Evaluate significance and dynamic badge colors
    significant = (pear_p < 0.05) or (spear_p < 0.05)
    
    # Build dynamic badge text based on significance and direction of Pearson correlation
    if significant:
        direction = "positive" if pear_r > 0 else "negative"
        badge = (
            f"[bold green]✔ Significant relationship[/bold green]\n"
            f"[grey50]Feature shows a linear {direction} directional association with survival.[/grey50]"
        )
    else:
        badge = (
            "[bold grey50]✖ No strong relationship[/bold grey50]\n"
            "[grey35]Pattern consistent with random variation.[/grey35]"
        )

    # 19-3: Build Aligned Layout
    layout_grid = Table.grid(expand=False)
    layout_grid.add_row(f"[bold white]Predictive Metrics Summary[/bold white] {f'[grey35]({txt})[/grey35]' if txt else ''}")
    layout_grid.add_row("")  # Visual break spacer
    layout_grid.add_row(stats_table)
    layout_grid.add_row("")  # Visual break spacer
    layout_grid.add_row(badge)

    # Print the outer wrapped panel frame safely
    console.print(
        Panel(
            layout_grid,
            title=f"{col}",
            title_align="left",
            border_style="cyan" if significant else "grey35",
            box=ROUNDED,
            padding=(1, 2),
            expand=False
        )
    )
    
    pass


# -------------------------------------------------------------------------------------------
# 20: Get Top Frequencies from a Delimited String Column with Rich UI Display
# -------------------------------------------------------------------------------------------   
def get_top_frequencies(data, column_name, top_n=20, sep=",", txt=''):
    """
    Explodes a string-delimited column into individual items, calculates 
    their frequency distribution, and displays a slick, modern ranking table.
    """
    console = Console()

    # 1. Gracefully handle edge case where column doesn't exist in data
    if column_name not in data.columns:
        console.print(
            Panel(
                f"[bold yellow]Column Missing[/bold yellow]\n"
                f"[grey50]The feature '[bold cyan]{column_name}[/bold cyan]' was not found in the dataset.[/grey50]",
                title="Pipeline Warning",
                border_style="yellow",
                box=ROUNDED,
                padding=(1, 2),
                expand=False
            )
        )
        return []

    # 2. Extract series, drop missing values, and enforce string mapping to avoid typing errors
    raw_series = data[column_name].dropna().astype(str)

    if raw_series.empty:
        console.print(
            Panel(
                f"[bold yellow]Empty Column Pool[/bold yellow]\n"
                f"[grey50]The feature '[bold cyan]{column_name}[/bold cyan]' contains zero valid entries to process.[/grey50]",
                title="Pipeline Warning",
                border_style="yellow",
                box=ROUNDED,
                padding=(1, 2),
                expand=False
            )
        )
        return []

    # 3. Explode delimited elements and strip whitespace blocks
    all_items = (
        raw_series
        .str.split(sep)
        .explode()
        .str.strip()
    )
    
    # Drop empty string remnants that occur from trailing commas (e.g., "Med_A, ")
    all_items = all_items[all_items != ""]

    # 4. Calculate distributions and grab top limits
    freq_counter = Counter(all_items)
    top_items = freq_counter.most_common(top_n)
    total_extracted_counts = sum(freq_counter.values())

    # 5. Build Compact Rich Layout Table
    stats_table = Table(
        box=None,
        show_header=True,
        header_style="bold grey50",
        padding=(0, 3)
    )
    stats_table.add_column("Rank", justify="center", style="grey50")
    stats_table.add_column("Categorical Item", justify="left", style="cyan", min_width=24)
    stats_table.add_column("Frequency", justify="right", style="magenta")
    stats_table.add_column("Share (%)", justify="right", style="yellow")

    # Populate rows with explicit percentage weights
    for rank, (item, count) in enumerate(top_items, start=1):
        percentage_share = count / total_extracted_counts if total_extracted_counts > 0 else 0
        stats_table.add_row(
            f"#{rank}",
            str(item),
            f"{count:,}",
            f"{percentage_share:.2%}"
        )

    # 6. Master Layout Assembly Grid
    layout_grid = Table.grid(expand=False)
    layout_grid.add_row(f"[bold white]Frequency Distribution Ranking[/bold white] {f'[grey35]({txt})[/grey35]' if txt else ''}")
    layout_grid.add_row(f"[grey50]Total Exploded Occurrences: {total_extracted_counts:,}[/grey50]\n")
    layout_grid.add_row(stats_table)

    # 7. Print Output Frame with complete emoji-safe top boundary assurance
    console.print(
        Panel(
            layout_grid,
            title=f"{column_name}",
            title_align="left",
            border_style="cyan",
            box=ROUNDED,
            padding=(1, 2),
            expand=False
        )
    )


# -------------------------------------------------------------------------------------------
# 21: Compare VIF Diagnostics for GMM Soft-Cluster Features Across Multiple Component Counts
# (Assumes results from optimize_gmm_components() with stored gmm_features for each k)
# -------------------------------------------------------------------------------------------
def compare_gmm_vif(results, component_list):
    """
    Compute and display VIF diagnostics for GMM soft-cluster features
    across multiple component counts.

    Parameters
    ----------
    results : dict
        Output dictionary from optimize_gmm_components(), containing:
        results["all_models"][k]["gmm_features"]
    component_list : list of int
        List of component counts to evaluate (e.g., [3, 9])

    Returns
    -------
    dict
        {
            k: DataFrame of VIF values for that model
        }
    """

    console = Console()
    vif_output = {}

    for k in component_list:

        gmm_features = results["all_models"][k]["gmm_features"]

        vif_df = pd.DataFrame({
            "Feature": [f"Cluster_{i+1}" for i in range(gmm_features.shape[1])],
            "VIF": [
                variance_inflation_factor(gmm_features, i)
                for i in range(gmm_features.shape[1])
            ]
        })

        vif_output[k] = vif_df

        # Build Rich table
        table = Table(
            title=f"VIF Diagnostic — {k} Components",
            box=ROUNDED,
            header_style="bold cyan",
            padding=(0, 1)
        )
        table.add_column("Feature", justify="left")
        table.add_column("VIF", justify="right")

        for _, row in vif_df.iterrows():
            table.add_row(row["Feature"], f"{row['VIF']:.3f}")

        console.print(table)

    return vif_output
