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
                base_condition = condition.replace('s_', '')
                filler_effects.append({
                    'condition': base_condition,
                    'filler_effect': filler_effect
                })
    # Create dataframe
    filler_df = pd.DataFrame(filler_effects)
    # Identify gap type and calculate average by gap type
    filler_df['gap_type'] = filler_df['condition'].apply(lambda x: '+gap' if 'b' in x else '-gap')
    gap_avg = filler_df.groupby('gap_type')['filler_effect'].mean().reset_index()
    gap_avg.columns = ['gap_type', 'mean_filler_effect']
    return gap_avg

def plot_filler_effects(result, output_path, title_prefix):
    """
    Create a bar plot of filler effects centered at 0.
    
    Parameters:
    -----------
    result : pd.DataFrame
        Dataframe with gap_type and mean_filler_effect columns
    output_path : Path
        Path to save the figure
    title_prefix : str
        Prefix to include in the plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create bar plot
    colors = ['#1f77b4' if x < 0 else '#ff7f0e' for x in result['mean_filler_effect']]
    bars = ax.bar(result['gap_type'], result['mean_filler_effect'], color=colors, alpha=0.7, edgecolor='black')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Customize plot
    ax.set_ylabel('Mean Filler Effect', fontsize=12)
    ax.set_xlabel('Gap Type', fontsize=12)
    ax.set_title(f'{title_prefix}: Filler Effects by Gap Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')
    
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