import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_fge(df, title, output_path):
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
    result_sorted = df.copy()
    #result_sorted['constr'] = pd.Categorical(result_sorted['construction_type'], 
    #                                                     categories=['simple', 'island'], 
    #                                                     ordered=True)
    result_sorted = result_sorted.sort_values(['island', 'gap'], ascending=[True, False]).reset_index(drop=True)
    
    # Create labels using just gap_type (since construction is shown by grouping)
    result_sorted['island'] = result_sorted['island'].map({1:"island", -1:"simple"})
    result_sorted['gap'] = result_sorted['gap'].map({1:"+gap", -1:"-gap"})

    # Create color mapping based on gap_type
    color_map = {"+gap": '#1f77b4', "-gap": '#ff7f0e'}
    colors = [color_map[gap] for gap in result_sorted['gap']]
    bars = ax.bar(range(len(result_sorted)), result_sorted['effect'], 
                  color=colors, alpha=0.7, edgecolor='black')
    
    
    # Set x-axis labels
    ax.set_xticks(range(len(result_sorted)))
    ax.set_xticklabels(result_sorted['island'], fontsize=10)

    
    # Add construction type labels as group separators
    # Find the midpoint for each construction type
    construction_positions = result_sorted.groupby('island').apply(lambda x: (x.index[0] + x.index[-1]) / 2)
    for construction, pos in construction_positions.items():
        ax.text(pos, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15, 
                construction, ha='center', fontsize=11, fontweight='bold')
    
    # Add vertical separator between construction types
    if len(result_sorted['island'].unique()) > 1:
        # Find boundary between construction types
        separator_pos = result_sorted.groupby('island').size().iloc[0] - 0.5
        ax.axvline(x=separator_pos, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Customize plot
    ax.set_ylabel('Mean Filler Effect', fontsize=12)
    ax.set_xlabel('Gap Type', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["+gap"], edgecolor='black', alpha=0.7, label='+gap'),
        Patch(facecolor=color_map["-gap"], edgecolor='black', alpha=0.7, label='-gap')
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot results from analysis.py')
    parser.add_argument('--output_path', type=str, required=False, default='./analysis/output',
                        help='Path to output directory for results')
    args = parser.parse_args()

    output_path = Path(args.output_path)

    df = pd.read_csv(output_path / "fge.tsv", sep="\t")

    models = df["model"].unique()
    constrs = df["constr"].unique()

    # total iterations for tqdm
    total = len(models) * len(constrs)

    for model, constr in tqdm(
        [(m, c) for m in models for c in constrs],
        total=total,
        desc="Generating Plots"
    ):
        data = df[
            (df["model"] == model) &
            (df["constr"] == constr) #&
            #(df["island"] == -1)
        ]

        print(data)
    
        mean_fge = (data[["gap", "island", "effect"]]).groupby(["gap", "island"]).mean().reset_index()

        plot_fge(mean_fge, f"Filler Effects by Construction and Gap Type ({model}, {constr})", output_path / model / f"fge_{model}_{constr}.png")

if __name__ == "__main__":
    main()