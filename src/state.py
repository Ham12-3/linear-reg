"""
Shared session state initialization for Linear Regression Studio.
Ensures all pages have access to required session state variables.
"""

import streamlit as st


def init_state():
    """
    Initialize all session state variables with defaults.
    Safe to call multiple times - only sets values if not already present.
    """
    defaults = {
        # Data state
        'dataset': None,
        'dataset_name': None,
        'target_col': None,
        'df': None,
        # Training state
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'pipeline': None,
        'training_info': None,
        'train_metrics': None,
        'test_metrics': None,
        'y_train_pred': None,
        'y_test_pred': None,
        # Config state (persisted across reruns)
        'model_type': 'linear',
        'alpha': 1.0,
        'test_size': 0.2,
        'random_state': 42,
        'missing_strategy': 'mean',
        # Performance debug
        'debug_performance': False,
        # Cache timestamps
        'last_data_load': None,
        'last_model_train': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
