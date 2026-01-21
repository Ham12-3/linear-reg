"""
Prediction Page for Linear Regression Studio.
Handles making predictions with the trained model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state import init_state
from src.train import predict, get_model_equation, get_coefficient_df


def show_prediction_page():
    """Display the prediction page."""
    # Initialize state first
    init_state()

    # Title and confirmation at very top
    st.title("🎯 Make Predictions")
    st.caption("Page loaded successfully")
    st.markdown("---")

    # Check prerequisites
    if st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first on the Dataset page.")
        st.stop()

    if st.session_state.pipeline is None:
        st.warning("⚠️ Please train a model first on the Model Training page.")
        st.stop()

    # Show model info
    training_info = st.session_state.training_info
    test_metrics = st.session_state.test_metrics

    st.info(
        f"**Trained Model:** {training_info['model_type'].capitalize()} Regression | "
        f"**Features:** {training_info['n_features']} | "
        f"**Test R²:** {test_metrics['R²']:.4f}"
    )

    # Model equation section
    st.subheader("📐 Model Equation")

    with st.expander("View model equation", expanded=False):
        equation = get_model_equation(training_info, top_n=10)
        st.code(equation, language=None)

        st.markdown("**Top 10 Coefficients:**")
        coef_df = get_coefficient_df(training_info).head(10)
        st.dataframe(coef_df, use_container_width=True)

    # Prediction section
    st.markdown("---")
    st.subheader("🔮 Enter Feature Values")

    # Get feature statistics for setting defaults and ranges
    feature_names = training_info['feature_names']
    X_train = st.session_state.X_train

    # Create input widgets dynamically
    input_values = create_feature_inputs(feature_names, X_train)

    # Prediction button
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🚀 Make Prediction",
            type="primary",
            use_container_width=True
        )

    if predict_button:
        make_prediction(input_values, feature_names)

    # Batch prediction section
    st.markdown("---")
    st.subheader("📊 Batch Prediction")

    show_batch_prediction(feature_names)


def create_feature_inputs(feature_names: list, X_train: pd.DataFrame) -> dict:
    """Create input widgets for each feature."""
    input_values = {}

    # Calculate number of columns based on feature count
    n_features = len(feature_names)
    n_cols = min(3, n_features)

    # Create columns
    cols = st.columns(n_cols)

    for i, feature in enumerate(feature_names):
        col_idx = i % n_cols

        with cols[col_idx]:
            # Get feature statistics
            feat_min = float(X_train[feature].min())
            feat_max = float(X_train[feature].max())
            feat_mean = float(X_train[feature].mean())
            feat_std = float(X_train[feature].std())

            # Expand range slightly for input
            range_margin = (feat_max - feat_min) * 0.1
            input_min = feat_min - range_margin
            input_max = feat_max + range_margin

            # Handle edge cases
            if abs(feat_max - feat_min) < 0.001:
                input_min = feat_mean - 1
                input_max = feat_mean + 1

            input_values[feature] = st.number_input(
                f"**{feature}**",
                min_value=float(input_min),
                max_value=float(input_max),
                value=float(feat_mean),
                step=float(feat_std / 10) if feat_std > 0 else 0.1,
                format="%.4f",
                help=f"Range: [{feat_min:.2f}, {feat_max:.2f}], Mean: {feat_mean:.2f}"
            )

    return input_values


def make_prediction(input_values: dict, feature_names: list):
    """Make a single prediction."""
    # Create DataFrame from input values
    input_df = pd.DataFrame([input_values])[feature_names]

    try:
        # Make prediction
        prediction = predict(st.session_state.pipeline, input_df)[0]

        # Display result
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success(f"**Predicted {st.session_state.target_col}:** {prediction:.4f}")

        # Show input summary
        with st.expander("View input values", expanded=False):
            input_summary = pd.DataFrame({
                'Feature': feature_names,
                'Value': [input_values[f] for f in feature_names]
            })
            st.dataframe(input_summary, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")


def show_batch_prediction(feature_names: list):
    """Show batch prediction section."""
    st.markdown("""
    Upload a CSV file with feature values to make predictions for multiple samples.
    The CSV should have the same columns as the training features.
    """)

    # Show expected columns
    with st.expander("View expected columns"):
        st.write("Your CSV file should contain these columns:")
        st.code(", ".join(feature_names))

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV for batch prediction:",
        type=['csv'],
        key="batch_pred_upload"
    )

    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)

            # Check for required columns
            missing_cols = set(feature_names) - set(batch_df.columns)
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {missing_cols}")
                return

            # Get only required columns in correct order
            batch_df = batch_df[feature_names]

            st.markdown("#### Preview of uploaded data:")
            st.dataframe(batch_df.head(10), use_container_width=True)

            # Make predictions button
            if st.button("🚀 Predict All", key="batch_predict_btn"):
                with st.spinner("Making predictions..."):
                    predictions = predict(st.session_state.pipeline, batch_df)

                    # Add predictions to dataframe
                    result_df = batch_df.copy()
                    result_df[f'Predicted_{st.session_state.target_col}'] = predictions

                    st.markdown("#### Predictions:")
                    st.dataframe(result_df, use_container_width=True)

                    # Summary statistics
                    st.markdown("#### Prediction Summary:")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Count", len(predictions))
                    with col2:
                        st.metric("Mean", f"{predictions.mean():.4f}")
                    with col3:
                        st.metric("Min", f"{predictions.min():.4f}")
                    with col4:
                        st.metric("Max", f"{predictions.max():.4f}")

                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # Template download
    st.markdown("---")
    st.markdown("**Need a template?**")

    # Create template with sample values
    template_df = st.session_state.X_train[feature_names].head(3)
    template_csv = template_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Template CSV",
        data=template_csv,
        file_name="prediction_template.csv",
        mime="text/csv",
        key="template_download"
    )
