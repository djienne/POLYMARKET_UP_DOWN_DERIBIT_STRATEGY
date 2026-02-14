#!/usr/bin/env python3
"""Plot probability data from the dry-run trading system."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare probability data."""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def plot_probabilities(df: pd.DataFrame):
    """Create plots for probability data."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot 1: UP Probability - Model vs Polymarket
    ax1 = axes[0]
    ax1.plot(df['timestamp'], df['model_prob_up'], label='Model Prob Up', color='green', linewidth=1.5)
    ax1.plot(df['timestamp'], df['poly_prob_up'], label='Polymarket Prob Up', color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Probability (%)')
    ax1.set_title('UP Probability: Model vs Polymarket')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2: DOWN Probability - Model vs Polymarket
    ax2 = axes[1]
    ax2.plot(df['timestamp'], df['model_prob_down'], label='Model Prob Down', color='red', linewidth=1.5)
    ax2.plot(df['timestamp'], df['poly_prob_down'], label='Polymarket Prob Down', color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Probability (%)')
    ax2.set_title('DOWN Probability: Model vs Polymarket')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    # Plot 3: Edge (model/polymarket ratio)
    ax3 = axes[2]
    ax3.plot(df['timestamp'], df['edge_up'], label='Edge Up', color='green', linewidth=1.5)
    ax3.plot(df['timestamp'], df['edge_down'], label='Edge Down', color='red', linewidth=1.5)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No Edge')
    ax3.set_ylabel('Edge (ratio)')
    ax3.set_title('Trading Edge (Model Prob / Polymarket Prob)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Time')

    # Format x-axis
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    return fig

def main():
    # Path to data
    project_root = Path(__file__).parent.parent
    csv_path = project_root / 'results' / 'probabilities.csv'

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    # Load data
    df = load_data(str(csv_path))
    print(f"Loaded {len(df)} data points")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Create plots
    fig = plot_probabilities(df)

    # Save plot
    output_path = project_root / 'results' / 'probabilities_plot.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Show plot
    plt.show()

if __name__ == '__main__':
    main()
