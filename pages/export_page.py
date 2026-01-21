"""
Export Page for Linear Regression Studio.
Handles report generation and export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state import init_state
from src.io_export import (
    generate_html_report,
    generate_markdown_report,
    create_export_data
)
from src.plots import (
    create_predicted_vs_actual,
    create_residuals_plot,
    create_coefficients_plot
)
from src.train import get_model_summary

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show_export_page():
    """Display the export page."""
    # Initialize state first
    init_state()

    # Title and confirmation at very top
    st.title("📄 Export Report")
    st.caption("Page loaded successfully")
    st.markdown("---")

    # Check prerequisites
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first on the Dataset page.")
        st.stop()

    if st.session_state.pipeline is None:
        st.warning("⚠️ Please train a model first on the Model Training page.")
        st.stop()

    # Show current session summary
    st.subheader("📊 Session Summary")

    show_session_summary()

    # Export options
    st.markdown("---")
    st.subheader("📥 Export Options")

    tab1, tab2 = st.tabs(["HTML Report", "Markdown Report"])

    with tab1:
        export_html_report()

    with tab2:
        export_markdown_report()

    # Additional exports
    st.markdown("---")
    st.subheader("📁 Additional Exports")

    show_additional_exports()


def show_session_summary():
    """Display a summary of the current session."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Dataset Information")
        dataset_info = {
            "Dataset Name": st.session_state.dataset_name,
            "Rows": st.session_state.df.shape[0],
            "Columns": st.session_state.df.shape[1],
            "Target Column": st.session_state.target_col,
            "Missing Values": st.session_state.df.isnull().sum().sum()
        }
        for key, value in dataset_info.items():
            st.markdown(f"**{key}:** {value}")

    with col2:
        st.markdown("#### Model Information")
        model_summary = get_model_summary(st.session_state.training_info)
        for key, value in model_summary.items():
            st.markdown(f"**{key}:** {value}")

    # Metrics
    st.markdown("#### Model Performance")
    col1, col2, col3, col4 = st.columns(4)

    test_metrics = st.session_state.test_metrics
    with col1:
        st.metric("Test MAE", f"{test_metrics['MAE']:.4f}")
    with col2:
        st.metric("Test MSE", f"{test_metrics['MSE']:.4f}")
    with col3:
        st.metric("Test RMSE", f"{test_metrics['RMSE']:.4f}")
    with col4:
        st.metric("Test R²", f"{test_metrics['R²']:.4f}")


def export_html_report():
    """Generate and export HTML report."""
    st.markdown("Generate a comprehensive HTML report with visualizations.")

    # Preview option
    if st.checkbox("Preview charts to include", key="html_preview"):
        show_chart_preview()

    # Generate button
    if st.button("📄 Generate HTML Report", type="primary", key="gen_html"):
        with st.spinner("Generating report..."):
            # Create figures
            figures = create_report_figures()

            # Prepare data
            dataset_info = {
                "Dataset": st.session_state.dataset_name,
                "Rows": st.session_state.df.shape[0],
                "Columns": st.session_state.df.shape[1],
                "Target": st.session_state.target_col,
                "Test Size": f"{st.session_state.test_size * 100:.0f}%",
                "Random State": st.session_state.random_state
            }

            model_summary = get_model_summary(st.session_state.training_info)

            # Generate HTML
            html_content = generate_html_report(
                dataset_info=dataset_info,
                model_summary=model_summary,
                metrics=st.session_state.test_metrics,
                figures=figures,
                title=f"Linear Regression Report - {st.session_state.dataset_name}"
            )

            # Clean up figures
            for fig in figures.values():
                if fig is not None:
                    plt.close(fig)

            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regression_report_{timestamp}.html"

            st.download_button(
                label="📥 Download HTML Report",
                data=html_content,
                file_name=filename,
                mime="text/html"
            )

            st.success("✓ HTML report generated successfully!")


def export_markdown_report():
    """Generate and export Markdown report."""
    st.markdown("Generate a simple Markdown report (without images).")

    if st.button("📝 Generate Markdown Report", type="primary", key="gen_md"):
        with st.spinner("Generating report..."):
            # Prepare data
            dataset_info = {
                "Dataset": st.session_state.dataset_name,
                "Rows": st.session_state.df.shape[0],
                "Columns": st.session_state.df.shape[1],
                "Target": st.session_state.target_col,
                "Test Size": f"{st.session_state.test_size * 100:.0f}%",
                "Random State": st.session_state.random_state
            }

            model_summary = get_model_summary(st.session_state.training_info)

            # Generate Markdown
            md_content = generate_markdown_report(
                dataset_info=dataset_info,
                model_summary=model_summary,
                metrics=st.session_state.test_metrics,
                title=f"Linear Regression Report - {st.session_state.dataset_name}"
            )

            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regression_report_{timestamp}.md"

            st.download_button(
                label="📥 Download Markdown Report",
                data=md_content,
                file_name=filename,
                mime="text/markdown"
            )

            st.success("✓ Markdown report generated successfully!")

            # Preview
            with st.expander("Preview Markdown"):
                st.markdown(md_content)


def show_chart_preview():
    """Show preview of charts that will be included in the report."""
    figures = create_report_figures()

    col1, col2 = st.columns(2)

    with col1:
        if figures.get('predicted_vs_actual'):
            st.markdown("**Predicted vs Actual**")
            st.pyplot(figures['predicted_vs_actual'])

    with col2:
        if figures.get('coefficients'):
            st.markdown("**Feature Coefficients**")
            st.pyplot(figures['coefficients'])

    if figures.get('residuals'):
        st.markdown("**Residuals Analysis**")
        st.pyplot(figures['residuals'])

    # Clean up
    for fig in figures.values():
        if fig is not None:
            plt.close(fig)


def create_report_figures() -> dict:
    """Create all figures for the report."""
    figures = {}

    # Predicted vs Actual
    try:
        figures['predicted_vs_actual'] = create_predicted_vs_actual(
            st.session_state.y_test.values,
            st.session_state.y_test_pred,
            title="Predicted vs Actual (Test Set)"
        )
    except Exception as e:
        st.exception(e)
        figures['predicted_vs_actual'] = None

    # Residuals
    try:
        figures['residuals'] = create_residuals_plot(
            st.session_state.y_test.values,
            st.session_state.y_test_pred,
            title="Residuals Analysis"
        )
    except Exception as e:
        st.exception(e)
        figures['residuals'] = None

    # Coefficients
    try:
        figures['coefficients'] = create_coefficients_plot(
            st.session_state.training_info['feature_names'],
            st.session_state.training_info['coefficients'],
            title="Feature Coefficients",
            top_n=15
        )
    except Exception as e:
        st.exception(e)
        figures['coefficients'] = None

    return figures


def show_additional_exports():
    """Show additional export options."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Coefficients CSV")
        if st.button("📥 Export Coefficients", key="exp_coef"):
            coef_df = pd.DataFrame({
                'Feature': st.session_state.training_info['feature_names'],
                'Coefficient': st.session_state.training_info['coefficients']
            })
            coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
            coef_df = coef_df.drop('Abs_Coefficient', axis=1)

            csv = coef_df.to_csv(index=False)
            st.download_button(
                label="📥 Download",
                data=csv,
                file_name="model_coefficients.csv",
                mime="text/csv",
                key="dl_coef"
            )

    with col2:
        st.markdown("#### Metrics CSV")
        if st.button("📥 Export Metrics", key="exp_metrics"):
            metrics_df = pd.DataFrame({
                'Metric': list(st.session_state.test_metrics.keys()),
                'Training': list(st.session_state.train_metrics.values()),
                'Test': list(st.session_state.test_metrics.values())
            })

            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download",
                data=csv,
                file_name="model_metrics.csv",
                mime="text/csv",
                key="dl_metrics"
            )

    with col3:
        st.markdown("#### Predictions CSV")
        if st.button("📥 Export Test Predictions", key="exp_pred"):
            pred_df = st.session_state.X_test.copy()
            pred_df['Actual'] = st.session_state.y_test.values
            pred_df['Predicted'] = st.session_state.y_test_pred
            pred_df['Residual'] = pred_df['Actual'] - pred_df['Predicted']

            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download",
                data=csv,
                file_name="test_predictions.csv",
                mime="text/csv",
                key="dl_pred"
            )

    # Dataset export
    st.markdown("---")
    st.markdown("#### Export Processed Dataset")

    if st.button("📥 Export Full Dataset", key="exp_data"):
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset",
            data=csv,
            file_name=f"{st.session_state.dataset_name.replace(' ', '_').lower()}_data.csv",
            mime="text/csv",
            key="dl_data"
        )
