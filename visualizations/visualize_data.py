import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

class VisualizeData:
    # Set global design aesthetic
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['figure.autolayout'] = True

    # Custom professional color palette
    CUSTOM_PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    sns.set_palette(CUSTOM_PALETTE)

    def __init__(self):
        self.data = None
        try:
            self.data = pd.read_csv("../data/processed/heart_disease_clean.csv")
            print(self.data.head())
            print(self.data.describe())
        except FileNotFoundError as e:
            print("File not found: ", e)

    # Check class balance statistics
    def plot_class_balance(self, target_col="num", df=None, save_path=None):
        if df is None:
            df = self.data

        fig, ax = plt.subplots(figsize=(7, 5))
        total = len(df)

        sns.countplot(
            data=df,
            x=target_col,
            hue=target_col,
            palette=self.CUSTOM_PALETTE,
            legend=False,
            ax=ax,
            edgecolor="black",
            linewidth=0.5
        )

        for p in ax.patches:
            height = p.get_height()
            percentage = f'{(100 * height / total):.1f}%'
            ax.annotate(
                f'{int(height)}\n({percentage})',
                (p.get_x() + p.get_width() / 2., height / 2),
                ha='center', va='center',
                color='white', weight='bold', fontsize=11
            )

        ax.set_title(f'Class Balance: {target_col.capitalize()}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        sns.despine(top=True, right=True)
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Free up memory
            print(f"Graph successfully saved to: {path.resolve()}")
        else:
            plt.show()

    # Plots grid of histograms with KDE, Mean, and Median indicators.
    def plot_numerical_distributions(self, num_cols=None, cols_per_row=3, df=None, save_path=None):
        if df is None:
            df = self.data

        if df is None:
            print("No dataset loaded.")
            return
        continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

        if num_cols is None:
            num_cols = continuous_features

        n_cols = len(num_cols)
        # Fixed line break
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row

        # Fixed line break
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for i, col in enumerate(num_cols):
            ax = axes[i]
            sns.histplot(df[col], kde=True, ax=ax, color=self.CUSTOM_PALETTE[0],
                         bins=25, edgecolor="white", alpha=0.7)

            mean_val = df[col].mean()
            median_val = df[col].median()

            ax.axvline(mean_val, color='#d95f02', linestyle='--', linewidth=2, label=f'Mean ({mean_val:.2f})')
            ax.axvline(median_val, color='#7570b3', linestyle='-', linewidth=2, label=f'Median ({median_val:.2f})')

            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel('')
            ax.set_ylabel('Frequency')
            ax.legend(frameon=True, facecolor='white', edgecolor='none')
            sns.despine(top=True, right=True)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if save_path:
            from pathlib import Path
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Graph successfully saved to: {path.resolve()}")
        else:
            plt.show()

    # Correlation heatmap
    def plot_correlation_heatmap(self, num_cols=None, df=None, save_path=None):
        """Plots lower-triangle annotated correlation heatmap and saves it."""
        if df is None:
            df = self.data

        if df is None:
            print("No dataset loaded.")
            return

        continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        if num_cols is None:
            num_cols = continuous_features

        plt.figure(figsize=(10, 8))
        corr = df[num_cols].corr()

        # Mask for upper triangle
        import numpy as np
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Diverging colormap centered at zero
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Heatmap plot
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    annot=True, fmt=".2f", square=True, linewidths=.8,
                    cbar_kws={"shrink": .8, "label": "Pearson Correlation"})

        plt.title('Feature Correlation Matrix', pad=15)

        # Save to disk if save_path is provided, otherwise show plot
        if save_path:
            from pathlib import Path
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()  # Free up memory
            print(f"Correlation heatmap successfully saved to: {path.resolve()}")
        else:
            plt.show()

    # Method to trigger all visualizations
    def visualize_data(self):
        self.plot_class_balance(
            target_col="num",
            save_path="graphs/class_balance.png"
        )
        self.plot_numerical_distributions(save_path="graphs/numerical_distribution.png")
        self.plot_correlation_heatmap(
            save_path="graphs/correlation_heatmap.png"
        )


if __name__ == "__main__":
    visualize = VisualizeData()
    visualize.visualize_data()

