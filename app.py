"""
Linear Regression Studio - Main Application Entry Point

A professional Streamlit dashboard for training and analyzing
linear regression models with sklearn datasets or custom CSV data.
Optimized for performance with lazy loading and caching.
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Linear Regression Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.state import init_state


def main():
    """Main application function."""
    init_state()

    # Reset performance tracker at start of each run
    from src.perf import reset_tracker, show_performance_ui

    # Sidebar navigation
    st.sidebar.title("📈 Linear Regression Studio")
    st.sidebar.markdown("---")

    # Navigation
    page = st.sidebar.radio(
        "Navigate to",
        ["🏠 Home", "📊 Dataset", "🔧 Model Training", "🎯 Prediction", "📄 Export"],
        index=0
    )

    st.sidebar.markdown("---")

    # Status indicators
    st.sidebar.subheader("Session Status")

    if st.session_state.df is not None:
        st.sidebar.success(f"✓ Dataset: {st.session_state.dataset_name}")
    else:
        st.sidebar.warning("○ No dataset loaded")

    if st.session_state.pipeline is not None:
        st.sidebar.success(f"✓ Model: {st.session_state.model_type.capitalize()}")
    else:
        st.sidebar.warning("○ No model trained")

    # Debug performance toggle
    st.sidebar.markdown("---")
    st.session_state.debug_performance = st.sidebar.checkbox(
        "🔍 Debug performance",
        value=st.session_state.debug_performance,
        help="Show timing information for expensive operations"
    )

    # Page routing (lazy imports for each page)
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Dataset":
        from pages.dataset_page import show_dataset_page
        show_dataset_page()
    elif page == "🔧 Model Training":
        from pages.training_page import show_training_page
        show_training_page()
    elif page == "🎯 Prediction":
        from pages.prediction_page import show_prediction_page
        show_prediction_page()
    elif page == "📄 Export":
        from pages.export_page import show_export_page
        show_export_page()

    # Show performance debug panel if enabled
    show_performance_ui()


def show_home_page():
    """Display the home page."""
    st.title("📈 Linear Regression Studio")
    st.caption("Page loaded successfully")
    st.markdown("---")

    st.markdown("""
    Welcome to **Linear Regression Studio** - a professional tool for training,
    analyzing, and deploying linear regression models.

    ### 🚀 Getting Started

    1. **📊 Dataset Page** - Load a dataset (California Housing, Diabetes, or your own CSV)
    2. **🔧 Model Training** - Configure and train your regression model
    3. **🎯 Prediction** - Make predictions with your trained model
    4. **📄 Export** - Generate a comprehensive report

    ### 📋 Available Models

    | Model | Description |
    |-------|-------------|
    | **Linear Regression** | Standard OLS regression |
    | **Ridge Regression** | L2 regularization to prevent overfitting |
    | **Lasso Regression** | L1 regularization for feature selection |

    ### 📁 Supported Datasets

    - **California Housing** - Predict median house values from census data
    - **Diabetes** - Predict disease progression from patient measurements
    - **Custom CSV** - Upload your own regression dataset

    """)

    # Quick stats if data is loaded
    if st.session_state.df is not None:
        st.markdown("---")
        st.subheader("📊 Current Session")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Dataset", st.session_state.dataset_name)
        with col2:
            st.metric("Rows", st.session_state.df.shape[0])
        with col3:
            st.metric("Features", st.session_state.df.shape[1] - 1)
        with col4:
            if st.session_state.test_metrics:
                st.metric("R² Score", f"{st.session_state.test_metrics['R²']:.4f}")
            else:
                st.metric("R² Score", "Not trained")

    st.markdown("---")
    st.caption("Built with Streamlit • scikit-learn • matplotlib")


if __name__ == "__main__":
    main()
