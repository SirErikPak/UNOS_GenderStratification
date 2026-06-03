import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from collections import Counter
from wordcloud import WordCloud



# -------------------------------------------------------------------------------------------
# 1: Plot Informative Missingness with Violin + Strip Plot
# -------------------------------------------------------------------------------------------
def plot_histogram(data, column, bins=30, txt='', title_font=10, tick_font=9, KDE=True):
    """
    Plotting histograms of one or more columns arranged cleanly side-by-side 
    in a multi-column grid layout
    """
    # 1-1: If it's a single feature string, wrap it in a list to use the same grid engine
    if isinstance(column, str):
        column = [column]

    # 1-2: Count the number of features to plot for dynamic grid sizing
    num_features = len(column)
    
    # 1-2: Calculate Grid Dimensions (Max 3 plots next to each other per row)
    n_cols = min(3, num_features)
    n_rows = math.ceil(num_features / n_cols)
    
    # Dynamically scale chart dimensions based on grid size
    fig_width = 5 * n_cols
    fig_height = 4 * n_rows
    
    # 1-3: Extract all items from the specified columns, handling NaNs and unknowns
    sns.set_theme(style="whitegrid", rc={"grid.color": "#f0f0f0"})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # 1-4:Flatten axes array for straightforward 1D array looping
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 1-5: Loop through features and populate the grid side-by-side
    for i, col in enumerate(column):
        ax = axes[i]
        
        # Extract all items from the column, treating NaNs and unknowns as empty strings
        sns.histplot(
            data=data,
            x=col,
            kde=KDE,
            bins=bins,
            color="#00a8ad",       # Clean cyan signature theme
            edgecolor="#ffffff",    # Crisp white dividers between bins
            linewidth=1,
            line_kws={"linewidth": 2, "color": "#ff5f40"} if KDE else {}, # Modern orange KDE line
            ax=ax,
        )

        # Subtle layout aesthetics per subplot
        ax.set_title(
            f"{col} {txt}" if txt else f"Distribution of {col}", 
            fontsize=title_font, 
            fontweight='bold', 
            color='#222222', 
            pad=10, 
            loc='left'
        )
        ax.set_ylabel("Frequency", fontsize=tick_font, fontweight='bold', color='#555555')
        ax.set_xlabel(col, fontsize=tick_font, fontweight='bold', color='#555555')
        ax.tick_params(axis='both', which='major', labelsize=tick_font, labelcolor='#555555')

    # 1-6: Clean up and hide empty leftover subplot grids if the list doesn't fill the last row perfectly
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    # Strip unnecessary top/right outer boundaries across the active grid
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plt.show()
    
    pass


# -------------------------------------------------------------------------------------------
# 2: Plot Item Word Cloud from Frequency Dictionary
# -------------------------------------------------------------------------------------------
def plot_item_wordcloud(data_series, title="Item Frequencies", bg_color="white"):
    """
    Generates and displays a word cloud from a frequency dictionary.
    
    This is ideal for visualizing the 'top hits' in categorical data like 
    medications or crime types, where the size of the word represents its 
    relative frequency.

    Parameters:
    -----------
    data_series : pd.Series
        A pandas Series containing the categorical data to analyze. Each entry can be a string of items (e.g., "MedA, MedB") or NaN.
    title : str, default "Item Frequencies"
        The title to display at the top of the plot.
    bg_color : str, default "white"
        The background color of the word cloud image.

    Returns:
    --------
    None
        Displays the plot using matplotlib.
    """

    # Clean + split + explode
    all_items = (
        data_series
        .dropna()
        .astype(str)
        .str.split(",")      # split on commas
        .explode()           # one item per row
        .str.strip()         # remove whitespace
    )

    # Remove empty strings (important!)
    all_items = all_items[all_items != ""]

    # Count frequencies
    frequencies = Counter(all_items)
    
    # Initialize the WordCloud object
    # 1400x900 provides high resolution for reports
    wc = WordCloud(
        width=1400, 
        height=900, 
        background_color=bg_color,
        colormap='viridis', # Aesthetic color scheme
        max_words=100
    ).generate_from_frequencies(frequencies)

    # Visualization Setup
    plt.figure(figsize=(16, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"{title}\n", fontsize=24, fontweight='bold')
    plt.axis("off") # Hide axes for a clean look
    plt.tight_layout(pad=0)
    plt.show()


# -------------------------------------------------------------------------------------------
# 3: Plot Informative Missingness with Violin + Strip Plot
# -------------------------------------------------------------------------------------------
def plot_informative_missingness(
    data,
    col,
    target='TransplantSurvivalDay',
    unknown_val=None,
    title_font=12,
    tick_font=9
):
    """
    Modern, minimal, optimized visualization of how missingness in a feature
    relates to survival outcomes using a violin + strip overlay.
    """

    # Handle multiple columns
    if isinstance(col, (list, tuple)):
        for c in col:
            plot_informative_missingness(
                data, c, target=target,
                unknown_val=unknown_val,
                title_font=title_font,
                tick_font=tick_font
            )
        return

    # Prepare data
    df = data[[col, target]].copy()
    df = df.dropna(subset=[target])  # keep valid survival values

    # Missingness mask
    if unknown_val is not None:
        missing_mask = (df[col] == unknown_val) | df[col].isna()
    else:
        missing_mask = df[col].isna()

    df["Status"] = np.where(missing_mask, "Missing/Unknown", "Known")

    # Theme
    sns.set_theme(style="whitegrid", rc={"grid.color": "#e8e8e8"})
    fig, ax = plt.subplots(figsize=(7, 5))

    palette = {
        "Known": "#00a8ad",
        "Missing/Unknown": "#ff5f40"
    }

    # Violin layer
    sns.violinplot(
        data=df,
        x="Status",
        y=target,
        hue="Status",
        palette=palette,
        inner="quart",
        cut=0,
        linewidth=1.4,
        legend=False,
        ax=ax
    )

    # Strip overlay
    sns.stripplot(
        data=df,
        x="Status",
        y=target,
        color="#222222",
        alpha=0.15,
        size=3,
        jitter=0.25,
        ax=ax
    )

    # Titles & labels
    ax.set_title(
        f"Missingness Impact on Survival: {col}",
        fontsize=title_font,
        fontweight="bold",
        color="#222222",
        loc="left",
        pad=12
    )
    ax.set_xlabel("Missingness Status", fontsize=tick_font, fontweight="bold", color="#555555")
    ax.set_ylabel(target, fontsize=tick_font, fontweight="bold", color="#555555")
    ax.tick_params(axis="both", labelsize=tick_font, labelcolor="#555555")

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------------------------
# 4: Plot Survival Trend vs. Continuous Clinical Feature with Scatter + Rolling Mean
# -------------------------------------------------------------------------------------------
def plot_survival_trend(data, feature_col, survival_col, window_size=100):
    """
    Visualizes how a continuous clinical feature relates to survival duration.
    
    This plot combines:
    - Raw patient-level scatter points (low alpha to show density)
    - A smoothed rolling-mean trend line (centered window)
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing the feature and survival columns.
    feature_col : str
        Continuous feature used on the X-axis (e.g., donor age, creatinine).
    survival_col : str
        Survival duration column used on the Y-axis.
    window_size : int, default=100
        Rolling window size for smoothing the survival trend.
        Larger windows produce smoother, more stable curves.
    """

    # Extract and sort valid rows
    plot_data = (
        data[[feature_col, survival_col]]
        .dropna()
        .sort_values(feature_col)
        .copy()
    )

    # Compute centered rolling mean for smooth trend
    plot_data["smoothed_survival"] = (
        plot_data[survival_col]
        .rolling(window=window_size, center=True, min_periods=1)
        .mean()
    )

    # Create figure
    plt.figure(figsize=(12, 6))

    # Scatter: raw patient points (low alpha reveals density)
    sns.scatterplot(
        data=plot_data,
        x=feature_col,
        y=survival_col,
        alpha=0.3,
        color="gray",
        label="Individual Patients"
    )

    # Trend line: rolling mean
    plt.plot(
        plot_data[feature_col],
        plot_data["smoothed_survival"],
        color="red",
        linewidth=2,
        label=f"Rolling Mean (Window={window_size})"
    )

    # Clean up plot borders
    sns.despine(left=False, bottom=False, right=True, top=True)

    # Titles and labels
    plt.title(f"Survival Trend vs. {feature_col}")
    plt.xlabel(f"{feature_col} (Clinical Value)")
    plt.ylabel("Survival Duration (Days)")

    # Legend + grid
    plt.legend()
    plt.grid(True, alpha=0.3)

    # This strips the left, bottom, right, and top outer box lines entirely
    sns.despine(left=False, bottom=False, right=True, top=True)

    plt.tight_layout()
    plt.show()