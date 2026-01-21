"""
Model Training Page for Linear Regression Studio.
Handles model configuration, training, and evaluation.
Optimized with st.form to reduce reruns and lazy imports.
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state import init_state

# Lazy imports - only import when needed
_plt = None
_np = None


def _get_matplotlib():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def show_training_page():
    """Display the model training page."""
    # Initialize state first
    init_state()

    # Title and confirmation at very top
    st.title("🔧 Model Training")
    st.caption("Page loaded successfully")
    st.markdown("---")

    # Check if dataset is loaded
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first on the Dataset page.")
        st.stop()

    from src.perf import timer

    with timer("training_page_render"):

        # Show current dataset info
        st.info(f"**Current Dataset:** {st.session_state.dataset_name} | "
                f"**Shape:** {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns | "
                f"**Target:** {st.session_state.target_col}")

        # Configuration section using st.form to batch inputs
        st.subheader("1. Model Configuration")

        with st.form("training_config_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Model Type")
                model_type = st.selectbox(
                    "Select regression model:",
                    ["linear", "ridge", "lasso"],
                    format_func=lambda x: {
                        "linear": "Linear Regression (OLS)",
                        "ridge": "Ridge Regression (L2)",
                        "lasso": "Lasso Regression (L1)"
                    }[x],
                    index=["linear", "ridge", "lasso"].index(st.session_state.model_type)
                )

                alpha = st.slider(
                    "Regularization strength (α):",
                    min_value=0.001,
                    max_value=10.0,
                    value=st.session_state.alpha,
                    step=0.001,
                    format="%.3f",
                    help="Higher values = stronger regularization (only for Ridge/Lasso)"
                )

            with col2:
                st.markdown("#### Data Split")
                test_size = st.slider(
                    "Test set size:",
                    min_value=0.1,
                    max_value=0.5,
                    value=st.session_state.test_size,
                    step=0.05,
                    format="%.2f",
                    help="Proportion of data for testing"
                )

                random_state = st.number_input(
                    "Random seed:",
                    min_value=0,
                    max_value=9999,
                    value=st.session_state.random_state,
                    help="For reproducible splits"
                )

            # Preprocessing section
            st.markdown("#### Preprocessing")

            col1, col2 = st.columns(2)

            with col1:
                missing_strategy = st.selectbox(
                    "Missing value strategy:",
                    ["mean", "median", "drop"],
                    format_func=lambda x: {
                        "mean": "Mean Imputation",
                        "median": "Median Imputation",
                        "drop": "Drop Rows with Missing Values"
                    }[x],
                    index=["mean", "median", "drop"].index(st.session_state.missing_strategy)
                )

            with col2:
                st.caption("Note: StandardScaler is automatically applied to all features.")

            # Submit button inside form
            submitted = st.form_submit_button(
                "🚀 Train Model",
                type="primary",
                use_container_width=True
            )

            if submitted:
                train_and_evaluate(
                    model_type=model_type,
                    alpha=alpha if model_type != "linear" else 1.0,
                    test_size=test_size,
                    random_state=int(random_state),
                    missing_strategy=missing_strategy
                )

        # Show results if model is trained
        if st.session_state.pipeline is not None:
            show_training_results()


def train_and_evaluate(
    model_type: str,
    alpha: float,
    test_size: float,
    random_state: int,
    missing_strategy: str
):
    """Train and evaluate the model with performance tracking."""
    from src.perf import timer
    from src.preprocess import prepare_data_for_training
    from src.train import train_model, split_data, predict
    from src.metrics import calculate_all_metrics

    with timer("full_training_pipeline"):
        with st.spinner("Preparing data..."):
            with timer("prepare_data"):
                X, y, feature_names, error = prepare_data_for_training(
                    st.session_state.df,
                    st.session_state.target_col,
                    missing_strategy=missing_strategy
                )

                if error:
                    st.error(f"Data preparation failed: {error}")
                    return

        with st.spinner("Splitting data..."):
            with timer("split_data"):
                X_train, X_test, y_train, y_test = split_data(
                    X, y,
                    test_size=test_size,
                    random_state=random_state
                )

        with st.spinner("Training model..."):
            with timer("train_model"):
                pipeline, training_info = train_model(
                    X_train, y_train,
                    model_type=model_type,
                    alpha=alpha,
                    missing_strategy=missing_strategy if missing_strategy != "drop" else "mean"
                )

        with st.spinner("Evaluating model..."):
            with timer("evaluate_model"):
                y_train_pred = predict(pipeline, X_train)
                y_test_pred = predict(pipeline, X_test)
                train_metrics = calculate_all_metrics(y_train.values, y_train_pred)
                test_metrics = calculate_all_metrics(y_test.values, y_test_pred)

    # Store in session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.pipeline = pipeline
    st.session_state.training_info = training_info
    st.session_state.train_metrics = train_metrics
    st.session_state.test_metrics = test_metrics
    st.session_state.y_train_pred = y_train_pred
    st.session_state.y_test_pred = y_test_pred
    st.session_state.model_type = model_type
    st.session_state.alpha = alpha
    st.session_state.test_size = test_size
    st.session_state.random_state = random_state
    st.session_state.missing_strategy = missing_strategy

    st.success("✓ Model trained successfully!")
    st.rerun()


def show_training_results():
    """Display training results."""
    from src.perf import timer
    from src.train import get_model_equation, get_coefficient_df, get_model_summary
    from src.metrics import interpret_r2

    with timer("show_training_results"):
        st.markdown("---")
        st.subheader("2. Training Results")

        # Model summary
        st.markdown("#### Model Summary")
        model_summary = get_model_summary(st.session_state.training_info)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", model_summary["Model Type"])
        with col2:
            st.metric("Training Samples", model_summary["Training Samples"])
        with col3:
            st.metric("Features", model_summary["Number of Features"])

        # Model equation
        st.markdown("#### Model Equation")
        equation = get_model_equation(st.session_state.training_info, top_n=10)
        st.code(equation, language=None)

        # Metrics
        st.markdown("#### Evaluation Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Training Set**")
            train_metrics = st.session_state.train_metrics
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("MAE", f"{train_metrics['MAE']:.4f}")
                st.metric("MSE", f"{train_metrics['MSE']:.4f}")
            with metrics_col2:
                st.metric("RMSE", f"{train_metrics['RMSE']:.4f}")
                st.metric("R²", f"{train_metrics['R²']:.4f}")

        with col2:
            st.markdown("**Test Set**")
            test_metrics = st.session_state.test_metrics
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("MAE", f"{test_metrics['MAE']:.4f}")
                st.metric("MSE", f"{test_metrics['MSE']:.4f}")
            with metrics_col2:
                st.metric("RMSE", f"{test_metrics['RMSE']:.4f}")
                st.metric("R²", f"{test_metrics['R²']:.4f}")

        # R² interpretation
        r2_interpretation = interpret_r2(test_metrics['R²'])
        st.info(f"**R² Interpretation:** {r2_interpretation}")

        # Visualizations - use tabs to defer rendering
        st.markdown("---")
        st.subheader("3. Model Visualizations")

        viz_tab1, viz_tab2, viz_tab3 = st.tabs([
            "Predicted vs Actual", "Residuals", "Coefficients"
        ])

        with viz_tab1:
            show_predicted_vs_actual()

        with viz_tab2:
            show_residuals()

        with viz_tab3:
            show_coefficients()


def show_predicted_vs_actual():
    """Show predicted vs actual plot."""
    from src.perf import timer
    from src.plots import create_predicted_vs_actual

    plt = _get_matplotlib()
    np = _get_numpy()

    col1, col2 = st.columns([3, 1])

    with col2:
        data_split = st.radio("Show data:", ["Test Set", "Training Set", "Both"], key="pva_split")

    with col1:
        with timer("render_predicted_vs_actual"):
            if data_split == "Test Set":
                fig = create_predicted_vs_actual(
                    st.session_state.y_test.values,
                    st.session_state.y_test_pred,
                    title="Predicted vs Actual (Test Set)"
                )
            elif data_split == "Training Set":
                fig = create_predicted_vs_actual(
                    st.session_state.y_train.values,
                    st.session_state.y_train_pred,
                    title="Predicted vs Actual (Training Set)"
                )
            else:
                # Combined plot - optimized
                fig, ax = plt.subplots(figsize=(10, 8))

                # Downsample for display if needed
                y_train_vals = st.session_state.y_train.values
                y_test_vals = st.session_state.y_test.values
                y_train_pred = st.session_state.y_train_pred
                y_test_pred = st.session_state.y_test_pred

                max_points = 3000
                if len(y_train_vals) > max_points:
                    idx = np.random.choice(len(y_train_vals), max_points, replace=False)
                    y_train_vals = y_train_vals[idx]
                    y_train_pred = y_train_pred[idx]

                if len(y_test_vals) > max_points:
                    idx = np.random.choice(len(y_test_vals), max_points, replace=False)
                    y_test_vals = y_test_vals[idx]
                    y_test_pred = y_test_pred[idx]

                ax.scatter(y_train_vals, y_train_pred, alpha=0.5, color='#3498db', label='Train', s=30)
                ax.scatter(y_test_vals, y_test_pred, alpha=0.6, color='#e74c3c', label='Test', s=40)

                all_values = np.concatenate([
                    st.session_state.y_train.values, st.session_state.y_test.values,
                    st.session_state.y_train_pred, st.session_state.y_test_pred
                ])
                min_val, max_val = all_values.min(), all_values.max()
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect')

                ax.set_xlabel('Actual Values', fontsize=12)
                ax.set_ylabel('Predicted Values', fontsize=12)
                ax.set_title('Predicted vs Actual (Both Sets)', fontsize=14, fontweight='bold')
                ax.legend()
                ax.set_aspect('equal', 'box')
                plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)


def show_residuals():
    """Show residuals plot."""
    from src.perf import timer
    from src.plots import create_residuals_plot

    plt = _get_matplotlib()

    col1, col2 = st.columns([3, 1])

    with col2:
        data_split = st.radio("Show data:", ["Test Set", "Training Set"], key="res_split")

    with col1:
        with timer("render_residuals"):
            if data_split == "Test Set":
                fig = create_residuals_plot(
                    st.session_state.y_test.values,
                    st.session_state.y_test_pred,
                    title="Residuals Analysis (Test Set)"
                )
            else:
                fig = create_residuals_plot(
                    st.session_state.y_train.values,
                    st.session_state.y_train_pred,
                    title="Residuals Analysis (Training Set)"
                )

            st.pyplot(fig)
            plt.close(fig)


def show_coefficients():
    """Show coefficients plot."""
    from src.perf import timer
    from src.plots import create_coefficients_plot
    from src.train import get_coefficient_df

    plt = _get_matplotlib()

    col1, col2 = st.columns([3, 1])

    with col2:
        n_features = len(st.session_state.training_info['feature_names'])
        top_n = st.slider(
            "Show top N features:",
            min_value=5,
            max_value=min(n_features, 50),
            value=min(n_features, 15),
            key="coef_top_n"
        )

    with col1:
        with timer("render_coefficients"):
            fig = create_coefficients_plot(
                st.session_state.training_info['feature_names'],
                st.session_state.training_info['coefficients'],
                title=f"Feature Coefficients (Top {top_n})",
                top_n=top_n
            )
            st.pyplot(fig)
            plt.close(fig)

    # Also show as table (lazy load)
    with st.expander("View all coefficients as table"):
        coef_df = get_coefficient_df(st.session_state.training_info)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
