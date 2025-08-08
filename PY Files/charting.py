# charting.py

import os
import matplotlib.pyplot as plt

def save_bar_chart(data: dict, title: str, x_label: str, y_label: str, output_path: str):
    """
    Generates and saves a bar chart.

    Args:
        data (dict): Keys are x-axis labels, values are heights.
        title (str): Title of the chart.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        output_path (str): Full path to save the image.
    """
    if not data:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(data.keys(), data.values())
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def save_scatter_plot(x, y, x_label, y_label, title, output_path):
    """
    Saves a scatter plot of x vs y.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
