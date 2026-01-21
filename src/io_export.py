"""
Export module for Linear Regression Studio.
Handles exporting reports in HTML/Markdown format.
"""

import base64
import io
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt


def figure_to_base64(fig: plt.Figure, format: str = 'png', dpi: int = 100) -> str:
    """
    Convert matplotlib figure to base64 string.

    Args:
        fig: matplotlib Figure
        format: Image format
        dpi: Resolution

    Returns:
        Base64 encoded image string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/{format};base64,{img_base64}"


def generate_html_report(
    dataset_info: Dict[str, Any],
    model_summary: Dict[str, Any],
    metrics: Dict[str, float],
    figures: Dict[str, plt.Figure],
    title: str = "Linear Regression Report"
) -> str:
    """
    Generate an HTML report.

    Args:
        dataset_info: Dictionary with dataset information
        model_summary: Dictionary with model summary
        metrics: Dictionary with evaluation metrics
        figures: Dictionary with matplotlib figures
        title: Report title

    Returns:
        HTML string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert figures to base64
    figure_images = {}
    for name, fig in figures.items():
        if fig is not None:
            figure_images[name] = figure_to_base64(fig)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-align: right;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="timestamp">Generated on: {timestamp}</p>

        <h2>Dataset Summary</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
"""

    # Add dataset info
    for key, value in dataset_info.items():
        html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"

    html += """        </table>

        <h2>Model Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
"""

    # Add model summary
    for key, value in model_summary.items():
        html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"

    html += """        </table>

        <h2>Evaluation Metrics</h2>
        <div class="metric-grid">
"""

    # Add metrics as cards
    for metric_name, metric_value in metrics.items():
        formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
        html += f"""            <div class="metric-card">
                <div class="metric-value">{formatted_value}</div>
                <div class="metric-label">{metric_name}</div>
            </div>
"""

    html += """        </div>

        <h2>Visualizations</h2>
"""

    # Add figures
    figure_titles = {
        'predicted_vs_actual': 'Predicted vs Actual',
        'residuals': 'Residuals Analysis',
        'coefficients': 'Feature Coefficients',
        'correlation': 'Correlation Heatmap',
        'target_distribution': 'Target Distribution'
    }

    for name, img_data in figure_images.items():
        title_text = figure_titles.get(name, name.replace('_', ' ').title())
        html += f"""        <div class="figure">
            <h3>{title_text}</h3>
            <img src="{img_data}" alt="{title_text}">
        </div>
"""

    html += """
        <div class="footer">
            <p>Generated by Linear Regression Studio</p>
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_markdown_report(
    dataset_info: Dict[str, Any],
    model_summary: Dict[str, Any],
    metrics: Dict[str, float],
    title: str = "Linear Regression Report"
) -> str:
    """
    Generate a Markdown report.

    Args:
        dataset_info: Dictionary with dataset information
        model_summary: Dictionary with model summary
        metrics: Dictionary with evaluation metrics
        title: Report title

    Returns:
        Markdown string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# {title}

*Generated on: {timestamp}*

---

## Dataset Summary

| Property | Value |
|----------|-------|
"""

    for key, value in dataset_info.items():
        md += f"| {key} | {value} |\n"

    md += """
---

## Model Configuration

| Parameter | Value |
|-----------|-------|
"""

    for key, value in model_summary.items():
        md += f"| {key} | {value} |\n"

    md += """
---

## Evaluation Metrics

| Metric | Value |
|--------|-------|
"""

    for metric_name, metric_value in metrics.items():
        formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
        md += f"| {metric_name} | {formatted_value} |\n"

    md += """
---

*Generated by Linear Regression Studio*
"""

    return md


def save_report(
    content: str,
    filepath: str,
    format: str = 'html'
):
    """
    Save report content to file.

    Args:
        content: Report content
        filepath: Output file path
        format: Report format
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def create_export_data(
    dataset_name: str,
    dataset_shape: tuple,
    target_col: str,
    model_type: str,
    alpha: Optional[float],
    test_size: float,
    missing_strategy: str,
    metrics: Dict[str, float],
    figures: Dict[str, plt.Figure]
) -> Dict[str, Any]:
    """
    Create a structured data dictionary for export.

    Args:
        dataset_name: Name of the dataset
        dataset_shape: Shape of the dataset
        target_col: Target column name
        model_type: Type of model used
        alpha: Regularization parameter
        test_size: Test set size ratio
        missing_strategy: Missing value strategy
        metrics: Evaluation metrics
        figures: Dictionary of figures

    Returns:
        Dictionary with all export data
    """
    dataset_info = {
        "Dataset": dataset_name,
        "Rows": dataset_shape[0],
        "Columns": dataset_shape[1],
        "Target Column": target_col
    }

    model_summary = {
        "Model Type": model_type.capitalize(),
        "Alpha": alpha if alpha else "N/A",
        "Test Size": f"{test_size * 100:.0f}%",
        "Missing Value Strategy": missing_strategy.capitalize()
    }

    return {
        "dataset_info": dataset_info,
        "model_summary": model_summary,
        "metrics": metrics,
        "figures": figures
    }
