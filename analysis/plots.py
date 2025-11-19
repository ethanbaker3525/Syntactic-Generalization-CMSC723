import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_filler_effects(result, output_path, title_prefix):
    """
    Create a bar plot of filler effects centered at 0.
    
    Parameters:
    -----------
    result : pd.DataFrame
        Dataframe with gap_type, construction_type, and mean_filler_effect columns
    output_path : Path
        Path to save the figure
    title_prefix : str
        Prefix to include in the plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by construction_type first (simple before island), then gap_type for consistent ordering
    result_sorted = result.copy()
    result_sorted['construction_type'] = pd.Categorical(result_sorted['construction_type'], 
                                                         categories=['simple', 'island'], 
                                                         ordered=True)
    result_sorted = result_sorted.sort_values(['construction_type', 'gap_type']).reset_index(drop=True)
    
    # Create labels using just gap_type (since construction is shown by grouping)
    result_sorted['label'] = result_sorted['gap_type']

    # Create color mapping based on gap_type
    color_map = {'+gap': '#1f77b4', '-gap': '#ff7f0e'}
    colors = [color_map[gap] for gap in result_sorted['gap_type']]
    bars = ax.bar(range(len(result_sorted)), result_sorted['mean_filler_effect'], 
                  color=colors, alpha=0.7, edgecolor='black')
    
    
    # Set x-axis labels
    ax.set_xticks(range(len(result_sorted)))
    ax.set_xticklabels(result_sorted['label'], fontsize=10)

    
    # Add construction type labels as group separators
    # Find the midpoint for each construction type
    construction_positions = result_sorted.groupby('construction_type').apply(lambda x: (x.index[0] + x.index[-1]) / 2)
    for construction, pos in construction_positions.items():
        ax.text(pos, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15, 
                construction, ha='center', fontsize=11, fontweight='bold')
    
    # Add vertical separator between construction types
    if len(result_sorted['construction_type'].unique()) > 1:
        # Find boundary between construction types
        separator_pos = result_sorted.groupby('construction_type').size().iloc[0] - 0.5
        ax.axvline(x=separator_pos, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Customize plot
    ax.set_ylabel('Mean Filler Effect', fontsize=12)
    ax.set_xlabel('Gap Type', fontsize=12)
    ax.set_title(f'{title_prefix}: Filler Effects by Construction and Gap Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['+gap'], edgecolor='black', alpha=0.7, label='+gap'),
        Patch(facecolor=color_map['-gap'], edgecolor='black', alpha=0.7, label='-gap')
    ]
    #ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # Adjust layout to make room for construction type labels
    plt.subplots_adjust(bottom=0.15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot results from analysis.py')
    parser.add_argument('--output_path', type=str, required=False, default='./analysis/output',
                        help='Path to output directory for results')
    args = parser.parse_args()

    df = pd.read_csv(Path(args.output_path), sep="\t")

    models = df["model"].unique()
    constrs = df["constr"].unique()

    # total iterations for tqdm
    total = len(models) * len(constrs)

    for model, constr in tqdm(
        [(m, c) for m in models for c in constrs],
        total=total,
        desc="Fitting LME models"
    ):
    df.groupby(["model", "constr"])

if __name__ == "__main__":
    main()