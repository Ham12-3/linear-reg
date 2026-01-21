# Linear Regression Studio

A professional Streamlit dashboard for training, analyzing, and deploying linear regression models.

## Features

- **Multiple Datasets**: Use built-in sklearn datasets (California Housing, Diabetes) or upload your own CSV
- **Data Exploration**: View statistics, missing values, and visualizations
- **Model Training**: Train Linear, Ridge, or Lasso regression models
- **Comprehensive Metrics**: MAE, MSE, RMSE, R² with train/test comparison
- **Interactive Predictions**: Make single or batch predictions
- **Export Reports**: Generate HTML or Markdown reports with visualizations

## Requirements

- Python 3.10+
- Windows, macOS, or Linux

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## How to Use

### 1. Dataset Page

![Dataset Page Screenshot Placeholder]

**Purpose**: Load and explore your data.

**Steps**:
1. Navigate to "Dataset" in the sidebar
2. Choose a dataset source:
   - **Built-in Datasets**: Select California Housing or Diabetes
   - **Upload CSV**: Upload your own regression dataset
3. If uploading CSV, select the target column (must be numeric)
4. Explore your data:
   - **Preview**: View first N rows and column info
   - **Statistics**: Descriptive statistics for all numeric columns
   - **Missing Values**: Summary of missing data
   - **Visualizations**: Histograms, correlation heatmaps, scatter plots

### 2. Model Training Page

![Training Page Screenshot Placeholder]

**Purpose**: Configure and train your regression model.

**Steps**:
1. Navigate to "Model Training" in the sidebar
2. Configure your model:
   - **Model Type**: Linear, Ridge, or Lasso regression
   - **Alpha** (for Ridge/Lasso): Regularization strength
   - **Test Size**: Proportion of data for testing (10-50%)
   - **Random Seed**: For reproducible splits
   - **Missing Value Strategy**: Mean impute, median impute, or drop
3. Click "Train Model"
4. Review results:
   - Model equation and coefficients
   - Training and test metrics
   - Visualizations (Predicted vs Actual, Residuals, Coefficients)

### 3. Prediction Page

![Prediction Page Screenshot Placeholder]

**Purpose**: Make predictions with your trained model.

**Steps**:
1. Navigate to "Prediction" in the sidebar
2. **Single Prediction**:
   - Enter values for each feature using the input widgets
   - Click "Make Prediction" to see the result
3. **Batch Prediction**:
   - Download the template CSV
   - Fill in your data
   - Upload and click "Predict All"
   - Download predictions as CSV

### 4. Export Page

![Export Page Screenshot Placeholder]

**Purpose**: Generate and download reports.

**Options**:
- **HTML Report**: Comprehensive report with embedded charts
- **Markdown Report**: Simple text-based report
- **Additional Exports**:
  - Coefficients CSV
  - Metrics CSV
  - Test predictions CSV
  - Full dataset CSV

## Project Structure

```
linear-reg/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── pages/
│   ├── __init__.py
│   ├── dataset_page.py   # Dataset exploration page
│   ├── training_page.py  # Model training page
│   ├── prediction_page.py # Prediction page
│   └── export_page.py    # Report export page
├── src/
│   ├── __init__.py
│   ├── data.py           # Dataset loading utilities
│   ├── preprocess.py     # Preprocessing and pipeline building
│   ├── train.py          # Model training functions
│   ├── metrics.py        # Evaluation metrics
│   ├── plots.py          # Matplotlib visualizations
│   └── io_export.py      # Report generation
└── tests/
    ├── __init__.py
    ├── test_pipeline.py  # Pipeline tests
    └── test_metrics.py   # Metrics tests
```

## Running Tests

```bash
pytest tests/ -v
```

## Supported Models

| Model | Description | Hyperparameters |
|-------|-------------|-----------------|
| **Linear Regression** | Ordinary Least Squares | None |
| **Ridge Regression** | L2 regularization | Alpha (regularization strength) |
| **Lasso Regression** | L1 regularization | Alpha (regularization strength) |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error - average absolute difference |
| **MSE** | Mean Squared Error - average squared difference |
| **RMSE** | Root Mean Squared Error - square root of MSE |
| **R²** | Coefficient of determination - variance explained |

## CSV Upload Requirements

When uploading your own CSV:
- Must have at least 2 columns (features + target)
- Target column must be numeric
- Missing values are handled automatically
- Non-numeric columns are excluded from analysis

## Tips

1. **Start Simple**: Begin with Linear Regression before trying regularized models
2. **Check Residuals**: Look for patterns in residuals that indicate model issues
3. **Compare Metrics**: Watch for large gaps between train and test metrics (overfitting)
4. **Feature Importance**: Use the coefficients plot to identify important features
5. **Regularization**: Increase alpha if you suspect overfitting

## Troubleshooting

**App won't start**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (needs 3.10+)

**CSV upload fails**:
- Ensure file is valid CSV format
- Check for at least 2 numeric columns
- Remove any special characters from column names

**Model training fails**:
- Check for missing values in target column
- Ensure you have enough data (at least 10 rows recommended)
- Try a different missing value strategy

## License

MIT License - Feel free to use and modify for your projects.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Machine learning powered by [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
