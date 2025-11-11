import pandas as pd
import argparse
from pathlib import Path
from pymer4.models import lmer

# https://github.com/umd-psycholing/lm-syntactic-generalization/blob/main/analysis.py
# WIP

def calculate_linear_mixed_effect(df):
    raise NotImplementedError()

def main():
    parser = argparse.ArgumentParser(description='Calculate linear mixed effect model from surprisal data')
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
    result = calculate_linear_mixed_effect(df)

    """
    # Save results
    output_file = output_dir / f"{input_path.stem}_filler_effects.tsv"
    result.to_csv(output_file, sep="\t", index=False)
    print(f"\nResults saved to: {output_file}")
    print("\nFiller Effects by Gap Type:")
    print(result)
    """

if __name__ == "__main__":
    main()