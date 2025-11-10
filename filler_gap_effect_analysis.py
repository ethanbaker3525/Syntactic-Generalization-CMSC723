import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def calculate_filler_effects(df):
    """
    Calculate filler effects grouped by gap type from a dataframe with 'condition' and 'surprisal' columns.
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with 'condition' and 'surprisal' columns
    Returns:
    --------
    pd.DataFrame
        Dataframe with gap_type and mean_filler_effect columns
    """
    # Group by condition and calculate mean surprisal
    groups = df.groupby("condition")["surprisal"].mean()
    # Calculate filler effects for each condition pair
    filler_effects = []
    for condition in groups.index:
        # Check if this is a +filler condition (contains 'a')
        if '_a' in condition:
            # Get the corresponding -filler condition by replacing 'a' with 'x'
            plus_filler = condition
            minus_filler = condition.replace('_a', '_x')
            if minus_filler in groups.index:
                filler_effect = groups[plus_filler] - groups[minus_filler]
                filler_effects.append({
                    'condition': condition,
                    'filler_effect': filler_effect
                })
    # Create dataframe
    filler_df = pd.DataFrame(filler_effects)
    # Identify gap type and calculate average by gap type
    filler_df['gap_type'] = filler_df['condition'].apply(lambda x: '+gap' if 'b' in x else '-gap')
    filler_df['construction_type'] = filler_df['condition'].apply(lambda x: 'simple' if 's_i_' in x else 'island')
    gap_avg = filler_df.groupby(['gap_type', 'construction_type'])['filler_effect'].mean().reset_index()
    gap_avg.columns = ['gap_type', 'construction_type', 'mean_filler_effect']
    return gap_avg

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
    parser = argparse.ArgumentParser(description='Calculate filler effects from surprisal data')
    parser.add_argument('--eval_file_path', type=str, required=True,
                        help='Path to the evaluation TSV file (e.g., eval/baseline_evals/eval_cleft_baseline.tsv)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output directory for results')
    args = parser.parse_args()
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.eval_file_path)
    # Read input data
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path, sep="\t")
    # Calculate filler effects
    result = calculate_filler_effects(df)
    # Save results
    output_file = output_dir / f"{input_path.stem}_filler_effects.tsv"
    result.to_csv(output_file, sep="\t", index=False)
    print(f"\nResults saved to: {output_file}")
    print("\nFiller Effects by Gap Type:")
    print(result)
    
    # Create and save plot
    plot_file = output_dir / f"{input_path.stem}_filler_effects.png"
    plot_filler_effects(result, plot_file, input_path.stem)

if __name__ == "__main__":
    main()