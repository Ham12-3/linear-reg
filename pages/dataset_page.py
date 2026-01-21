"""
Dataset Page for Linear Regression Studio.
Handles dataset selection, loading, and exploration.
Optimized with lazy imports and performance tracking.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state import init_state

# Lazy imports
_plt = None
_pd = None
_np = None


def _get_matplotlib():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_pandas():
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def _get_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def show_dataset_page():
    """Display the dataset page."""
    # Initialize state first
    init_state()

    # Title and confirmation at very top
    st.title("📊 Dataset Explorer")
    st.caption("Page loaded successfully")
    st.markdown("---")

    from src.perf import timer

    with timer("dataset_page_render"):

        # Dataset selection
        st.subheader("1. Select Dataset")

        dataset_option = st.radio(
            "Choose a dataset source:",
            ["Built-in Datasets", "Upload CSV"],
            horizontal=True
        )

        df = None
        target_col = None
        dataset_name = None

        if dataset_option == "Built-in Datasets":
            df, target_col, dataset_name = load_builtin_dataset()
        else:
            df, target_col, dataset_name = load_custom_csv()

        # If dataset is loaded, show exploration
        if df is not None and target_col is not None:
            # Store in session state
            st.session_state.df = df
            st.session_state.target_col = target_col
            st.session_state.dataset_name = dataset_name

            # Reset model when dataset changes
            if st.session_state.dataset_name != dataset_name:
                st.session_state.pipeline = None
                st.session_state.training_info = None

            st.success(f"✓ Dataset '{dataset_name}' loaded successfully!")

            # Show exploration tabs
            show_data_exploration(df, target_col)


def load_builtin_dataset():
    """Load a built-in sklearn dataset."""
    from src.perf import timer
    from src.data import (
        load_california_housing,
        load_diabetes_dataset,
        AVAILABLE_DATASETS
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        dataset_choice = st.selectbox(
            "Select dataset:",
            list(AVAILABLE_DATASETS.keys())
        )

    with col2:
        st.info(AVAILABLE_DATASETS[dataset_choice]['description'])

    # Load the selected dataset
    try:
        with st.spinner(f"Loading {dataset_choice}..."):
            with timer("load_builtin_dataset"):
                if dataset_choice == "California Housing":
                    df, target_col = load_california_housing()
                else:
                    df, target_col = load_diabetes_dataset()

        return df, target_col, dataset_choice

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None, None


def load_custom_csv():
    """Load a custom CSV file."""
    from src.perf import timer
    from src.data import (
        load_csv_file,
        get_numeric_columns,
        validate_target_column
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file:",
        type=['csv'],
        help="CSV file with numeric columns. Last column will be used as target by default."
    )

    if uploaded_file is not None:
        with timer("load_csv_file"):
            df, error = load_csv_file(uploaded_file)

        if error:
            st.error(error)
            return None, None, None

        # Let user select target column
        numeric_cols = get_numeric_columns(df)

        if len(numeric_cols) < 2:
            st.error("CSV must have at least 2 numeric columns (features + target).")
            return None, None, None

        target_col = st.selectbox(
            "Select the target column:",
            numeric_cols,
            index=len(numeric_cols) - 1,
            help="The column you want to predict (must be numeric)"
        )

        # Validate target
        is_valid, msg = validate_target_column(df, target_col)
        if not is_valid:
            st.error(msg)
            return None, None, None

        return df, target_col, uploaded_file.name

    return None, None, None


def show_data_exploration(df, target_col: str):
    """Show data exploration section."""
    st.markdown("---")
    st.subheader("2. Data Exploration")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Preview", "📈 Statistics", "🔍 Missing Values", "📊 Visualizations"
    ])

    with tab1:
        show_data_preview(df)

    with tab2:
        show_data_statistics(df)

    with tab3:
        show_missing_values(df)

    with tab4:
        show_visualizations(df, target_col)


def show_data_preview(df):
    """Show data preview."""
    from src.perf import timer
    pd = _get_pandas()

    with timer("show_data_preview"):
        st.markdown("### Data Preview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Show head of data
        n_rows = st.slider("Number of rows to display:", 5, 50, 10)
        st.dataframe(df.head(n_rows), use_container_width=True, hide_index=True)

        # Column info (in expander to avoid computation unless needed)
        with st.expander("Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)


def show_data_statistics(df):
    """Show descriptive statistics."""
    from src.perf import timer
    np = _get_numpy()

    with timer("show_data_statistics"):
        st.markdown("### Descriptive Statistics")

        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe().T, use_container_width=True)


def show_missing_values(df):
    """Show missing values summary."""
    from src.perf import timer
    from src.data import get_missing_value_summary

    with timer("show_missing_values"):
        st.markdown("### Missing Values Analysis")

        missing_summary = get_missing_value_summary(df)

        if len(missing_summary) == 0:
            st.success("✓ No missing values found in the dataset!")
        else:
            total_missing = df.isnull().sum().sum()
            st.warning(f"Found {total_missing} missing values in {len(missing_summary)} columns.")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(missing_summary, use_container_width=True)

            with col2:
                # Visual representation (only if there are missing values)
                plt = _get_matplotlib()

                missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
                missing_pct = missing_pct[missing_pct > 0]

                if len(missing_pct) > 0:
                    fig, ax = plt.subplots(figsize=(8, max(4, len(missing_pct) * 0.3)))
                    missing_pct.plot(kind='barh', ax=ax, color='#e74c3c')
                    ax.set_xlabel('Missing %')
                    ax.set_title('Missing Values by Column')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)


def show_visualizations(df, target_col: str):
    """Show data visualizations."""
    st.markdown("### Data Visualizations")

    viz_type = st.selectbox(
        "Select visualization:",
        ["Target Distribution", "Correlation Heatmap", "Feature vs Target Scatter"]
    )

    if viz_type == "Target Distribution":
        show_target_histogram(df, target_col)

    elif viz_type == "Correlation Heatmap":
        show_correlation_heatmap(df)

    elif viz_type == "Feature vs Target Scatter":
        show_feature_scatter(df, target_col)


def show_target_histogram(df, target_col: str):
    """Show histogram of target variable."""
    from src.perf import timer
    from src.plots import create_histogram

    plt = _get_matplotlib()

    st.markdown(f"#### Distribution of Target: `{target_col}`")

    col1, col2 = st.columns([3, 1])

    with col2:
        bins = st.slider("Number of bins:", 10, 100, 30)
        color = st.color_picker("Bar color:", "#3498db")

    with col1:
        with timer("render_target_histogram"):
            fig = create_histogram(
                df[target_col],
                title=f"Distribution of {target_col}",
                xlabel=target_col,
                bins=bins,
                color=color
            )
            st.pyplot(fig)
            plt.close(fig)


def show_correlation_heatmap(df):
    """Show correlation heatmap."""
    from src.perf import timer
    from src.plots import create_correlation_heatmap

    plt = _get_matplotlib()
    np = _get_numpy()

    st.markdown("#### Correlation Heatmap")

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] > 20:
        st.warning("Large number of features. Showing correlation for first 20 numeric columns.")
        numeric_df = numeric_df.iloc[:, :20]

    with timer("render_correlation_heatmap"):
        fig = create_correlation_heatmap(
            numeric_df,
            title="Feature Correlation Matrix"
        )
        st.pyplot(fig)
        plt.close(fig)


def show_feature_scatter(df, target_col: str):
    """Show scatter plot of feature vs target."""
    from src.perf import timer
    from src.data import get_numeric_columns
    from src.plots import create_scatter_plot

    plt = _get_matplotlib()

    st.markdown("#### Feature vs Target Scatter Plot")

    numeric_cols = get_numeric_columns(df)
    feature_cols = [c for c in numeric_cols if c != target_col]

    if len(feature_cols) == 0:
        st.warning("No numeric feature columns available.")
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        selected_feature = st.selectbox("Select feature:", feature_cols)
        add_trendline = st.checkbox("Show trend line", value=True)
        color = st.color_picker("Point color:", "#3498db")

    with col1:
        with timer("render_feature_scatter"):
            fig = create_scatter_plot(
                df[selected_feature],
                df[target_col],
                xlabel=selected_feature,
                ylabel=target_col,
                title=f"{selected_feature} vs {target_col}",
                add_trendline=add_trendline,
                color=color
            )
            st.pyplot(fig)
            plt.close(fig)
