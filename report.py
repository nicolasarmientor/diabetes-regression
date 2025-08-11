import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from data_extraction import diabetes_data_extraction

def generate_pdf(results: pd.DataFrame, pdf_path: Path):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:

        heatmap = plt.figure(figsize=(6,5))
        sns.heatmap(diabetes_data_extraction()[0].corr(), cmap="coolwarm")
        plt.title("Feature Correlation Map")
        pdf.savefig(heatmap)
        plt.close(heatmap)

        barcharts, axes = plt.subplots(1, 2, figsize=(15, 5))
        r2_barchart = sns.barplot(data=results, x="Model", y="R2", hue="Model", ax=axes[0], legend=False)
        axes[0].set_title(r"$R^2$ Scores")
        axes[0].set_ylim(0.44, 0.47)
        for container in r2_barchart.containers:
            axes[0].bar_label(container, fmt="%.3f", label_type="edge")

        rmse_barchart = sns.barplot(data=results, x="Model", y="RMSE", hue="Model", ax=axes[1], legend=False)
        axes[1].set_title("RMSE Scores")
        axes[1].set_ylim(2850, 2950)
        for container in rmse_barchart.containers:
            axes[1].bar_label(container, fmt="%.2f", label_type="edge")

        barcharts.tight_layout
        pdf.savefig(barcharts)
        plt.close(barcharts)
