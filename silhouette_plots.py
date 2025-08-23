
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_scores(csv_path):
    df = pd.read_csv(csv_path, comment='#', names=['approach', 'type', 'label', 'score'])
    df = df.dropna()
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    return df


def plot_bar_per_approach(df):
    for approach in df['approach'].unique():
        plt.figure(figsize=(10, 6))
        subset = df[(df['approach'] == approach) & (df['label'] != 'all')]
        sns.barplot(data=subset, x='label', y='score', hue='type')
        plt.title(f'Silhouette Scores by Object for {approach}')
        plt.ylabel('Silhouette Score')
        plt.ylim(-0.2, 0.2)
        plt.legend(title='Method')
        plt.tight_layout()
        plt.show()


def plot_all_scores_heatmap(df):
    pivot = df[df['label'] != 'all'].pivot(
        index='label', columns=['approach', 'type'], values='score')
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='coolwarm', center=0)
    plt.title('Silhouette Scores Heatmap (per object, approach, and method)')
    plt.ylabel('Object Label')
    plt.xlabel('Approach, Method')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(__file__), 'silhouette_scores.csv')
    df = load_scores(csv_path)
    plot_bar_per_approach(df)
    plot_all_scores_heatmap(df)
