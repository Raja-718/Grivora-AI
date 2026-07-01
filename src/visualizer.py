# visualizer.py - Generate charts using matplotlib and plotly
import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_chart(df: pd.DataFrame, chart_type: str = "bar", output_dir: str = "app/static") -> str:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    num_df = df.select_dtypes(include='number')

    if chart_type == "bar":
        num_df.mean().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title("Column Averages")
    elif chart_type == "hist":
        num_df.hist(ax=ax)
        ax.set_title("Distribution")
    elif chart_type == "line":
        num_df.plot(kind='line', ax=ax)
        ax.set_title("Line Chart")
    elif chart_type == "box":
        num_df.plot(kind='box', ax=ax)
        ax.set_title("Box Plot")

    path = os.path.join(output_dir, f"{chart_type}_chart.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
