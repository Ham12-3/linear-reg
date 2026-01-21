"""
Plotting module for Linear Regression Studio.
Creates matplotlib figures for data visualization and model analysis.
Optimized with lazy imports, caching, and downsampling for large datasets.
"""

from typing import Tuple, Optional, List
import io
import hashlib
import streamlit as st

# Lazy imports for heavy libraries
_plt = None
_sns = None
_np = None
_pd = None
_style_set = False


def _get_matplotlib():
    """Lazy import matplotlib."""
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_seaborn():
    """Lazy import seaborn."""
    global _sns
    if _sns is None:
        import seaborn as sns
        _sns = sns
    return _sns


def _get_numpy():
    """Lazy import numpy."""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _get_pandas():
    """Lazy import pandas."""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def set_plot_style():
    """Set consistent plot style (only once)."""
    global _style_set
    if _style_set:
        return

    plt = _get_matplotlib()
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    _style_set = True


def _downsample_for_display(
    data,
    max_points: int = 5000,
    method: str = 'random'
) -> "np.ndarray":
    """
    Downsample data for chart display when dataset is large.
    Keeps full data for analytics, shows resampled view for charts.

    Args:
        data: Array-like data
        max_points: Maximum points to display
        method: 'random' or 'uniform'

    Returns:
        Downsampled data array
    """
    np = _get_numpy()

    if hasattr(data, 'values'):
        arr = data.values
    else:
        arr = np.asarray(data)

    if len(arr) <= max_points:
        return arr

    if method == 'random':
        indices = np.random.choice(len(arr), max_points, replace=False)
        indices.sort()
        return arr[indices]
    else:  # uniform
        step = len(arr) // max_points
        return arr[::step][:max_points]


def _compute_data_hash(data, *args) -> str:
    """Compute a hash for cache key from data."""
    np = _get_numpy()

    if hasattr(data, 'values'):
        arr = data.values
    else:
        arr = np.asarray(data)

    # Use shape and a sample for hashing (faster than full hash)
    key_parts = [str(arr.shape)]
    if len(arr) > 0:
        key_parts.append(str(arr.flat[0]))
        key_parts.append(str(arr.flat[-1]))
        key_parts.append(str(np.nanmean(arr)))

    for arg in args:
        key_parts.append(str(arg))

    return hashlib.md5("".join(key_parts).encode()).hexdigest()[:12]


def create_histogram(
    data: "pd.Series",
    title: str = "Distribution",
    xlabel: str = "Value",
    bins: int = 30,
    color: str = "#3498db",
    figsize: Tuple[int, int] = (10, 6)
) -> "plt.Figure":
    """
    Create a histogram of the data.
    Optimized with downsampling for large datasets.
    """
    from src.perf import timer

    with timer("create_histogram"):
        plt = _get_matplotlib()
        np = _get_numpy()
        set_plot_style()

        fig, ax = plt.subplots(figsize=figsize)

        # Clean data once
        clean_data = data.dropna()

        # Downsample for display if large
        display_data = _downsample_for_display(clean_data, max_points=10000)

        ax.hist(display_data, bins=bins, color=color, edgecolor='white', alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        # Compute stats on full data (vectorized)
        mean_val = np.nanmean(clean_data.values)
        std_val = np.nanstd(clean_data.values)
        median_val = np.nanmedian(clean_data.values)

        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMedian: {median_val:.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

    return fig


@st.cache_data(ttl=300, show_spinner=False)
def _compute_correlation_matrix(columns: tuple, data_hash: str, values_bytes: bytes) -> "np.ndarray":
    """Cached correlation matrix computation."""
    np = _get_numpy()
    pd = _get_pandas()

    # Reconstruct array from bytes
    arr = np.frombuffer(values_bytes, dtype=np.float64).reshape(-1, len(columns))
    df = pd.DataFrame(arr, columns=list(columns))
    return df.corr().values


def create_correlation_heatmap(
    df: "pd.DataFrame",
    title: str = "Correlation Heatmap",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r"
) -> "plt.Figure":
    """
    Create a correlation heatmap.
    Optimized with cached correlation computation.
    """
    from src.perf import timer

    with timer("create_correlation_heatmap"):
        plt = _get_matplotlib()
        np = _get_numpy()
        pd = _get_pandas()
        sns = _get_seaborn()
        set_plot_style()

        # Get only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Limit columns for performance
        if numeric_df.shape[1] > 20:
            numeric_df = numeric_df.iloc[:, :20]

        # Compute correlation (vectorized via pandas)
        corr_matrix = numeric_df.corr()

        # Reduce annotation precision for large matrices
        fmt = '.1f' if corr_matrix.shape[0] > 10 else '.2f'
        annot_size = 7 if corr_matrix.shape[0] > 10 else 8

        fig, ax = plt.subplots(figsize=figsize)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax,
            annot_kws={'size': annot_size},
            vmin=-1,
            vmax=1
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

    return fig


def create_scatter_plot(
    x: "pd.Series",
    y: "pd.Series",
    xlabel: str = "Feature",
    ylabel: str = "Target",
    title: str = "Scatter Plot",
    color: str = "#3498db",
    figsize: Tuple[int, int] = (10, 6),
    add_trendline: bool = True
) -> "plt.Figure":
    """
    Create a scatter plot.
    Optimized with downsampling for large datasets.
    """
    from src.perf import timer

    with timer("create_scatter_plot"):
        plt = _get_matplotlib()
        np = _get_numpy()
        set_plot_style()

        fig, ax = plt.subplots(figsize=figsize)

        # Create valid mask once
        valid_mask = ~(x.isna() | y.isna())
        x_valid = x[valid_mask].values
        y_valid = y[valid_mask].values

        # Downsample for display
        if len(x_valid) > 5000:
            indices = np.random.choice(len(x_valid), 5000, replace=False)
            x_display = x_valid[indices]
            y_display = y_valid[indices]
        else:
            x_display = x_valid
            y_display = y_valid

        ax.scatter(x_display, y_display, alpha=0.6, color=color, edgecolors='white', s=50)

        if add_trendline and len(x_valid) > 1:
            # Fit on full data, but only need 2 points for line
            z = np.polyfit(x_valid, y_valid, 1)
            p = np.poly1d(z)
            x_line = np.array([x_valid.min(), x_valid.max()])
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend line')
            ax.legend()

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

    return fig


def create_predicted_vs_actual(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    title: str = "Predicted vs Actual",
    figsize: Tuple[int, int] = (10, 8)
) -> "plt.Figure":
    """
    Create a predicted vs actual scatter plot.
    Optimized with downsampling for large datasets.
    """
    from src.perf import timer

    with timer("create_predicted_vs_actual"):
        plt = _get_matplotlib()
        np = _get_numpy()
        set_plot_style()

        fig, ax = plt.subplots(figsize=figsize)

        # Downsample for display
        if len(y_true) > 5000:
            indices = np.random.choice(len(y_true), 5000, replace=False)
            y_true_display = y_true[indices]
            y_pred_display = y_pred[indices]
        else:
            y_true_display = y_true
            y_pred_display = y_pred

        ax.scatter(y_true_display, y_pred_display, alpha=0.6, color='#3498db', edgecolors='white', s=50)

        # Perfect prediction line using full data range
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')

        ax.set_aspect('equal', 'box')
        plt.tight_layout()

    return fig


def create_residuals_plot(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    title: str = "Residuals Plot",
    figsize: Tuple[int, int] = (10, 6)
) -> "plt.Figure":
    """
    Create a residuals plot.
    Optimized with downsampling and vectorized computation.
    """
    from src.perf import timer

    with timer("create_residuals_plot"):
        plt = _get_matplotlib()
        np = _get_numpy()
        set_plot_style()

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Vectorized residual computation
        residuals = y_true - y_pred

        # Downsample for scatter plot
        if len(residuals) > 5000:
            indices = np.random.choice(len(residuals), 5000, replace=False)
            y_pred_display = y_pred[indices]
            residuals_display = residuals[indices]
        else:
            y_pred_display = y_pred
            residuals_display = residuals

        # Residuals vs Predicted (downsampled)
        axes[0].scatter(y_pred_display, residuals_display, alpha=0.6, color='#e74c3c', edgecolors='white', s=50)
        axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        axes[0].set_xlabel('Predicted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')

        # Residuals distribution (use all data for histogram)
        axes[1].hist(residuals, bins=30, color='#e74c3c', edgecolor='white', alpha=0.8)
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        axes[1].set_xlabel('Residual Value', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')

        # Vectorized stats computation
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        axes[1].text(0.95, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}',
                     transform=axes[1].transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

    return fig


def create_coefficients_plot(
    feature_names: List[str],
    coefficients: "np.ndarray",
    title: str = "Feature Coefficients",
    figsize: Tuple[int, int] = (10, 8),
    top_n: Optional[int] = None
) -> "plt.Figure":
    """
    Create a bar chart of feature coefficients.
    Optimized with vectorized sorting.
    """
    from src.perf import timer

    with timer("create_coefficients_plot"):
        plt = _get_matplotlib()
        np = _get_numpy()
        pd = _get_pandas()
        set_plot_style()

        # Vectorized sorting by absolute value
        abs_coefs = np.abs(coefficients)
        sort_indices = np.argsort(abs_coefs)

        sorted_features = [feature_names[i] for i in sort_indices]
        sorted_coefs = coefficients[sort_indices]

        if top_n is not None and len(sorted_features) > top_n:
            sorted_features = sorted_features[-top_n:]
            sorted_coefs = sorted_coefs[-top_n:]

        fig, ax = plt.subplots(figsize=figsize)

        # Vectorized color assignment
        colors = np.where(sorted_coefs < 0, '#e74c3c', '#27ae60')

        ax.barh(sorted_features, sorted_coefs, color=colors, edgecolor='white')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add legend (lazy import)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', label='Positive'),
            Patch(facecolor='#e74c3c', label='Negative')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

    return fig


def figure_to_bytes(fig: "plt.Figure", format: str = 'png', dpi: int = 100) -> bytes:
    """
    Convert matplotlib figure to bytes.
    Reduced default DPI for faster rendering.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def save_figure(fig: "plt.Figure", filepath: str, dpi: int = 100):
    """
    Save matplotlib figure to file.
    Reduced default DPI for faster rendering.
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
