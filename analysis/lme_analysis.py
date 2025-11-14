import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from pymer4.models import lmer
import polars as pl
from tqdm import tqdm

# https://github.com/umd-psycholing/lm-syntactic-generalization/blob/main/analysis.py
# WIP



def modify_base_dict(sentence_key, stim_set, base_dict):
    sent_copy = base_dict.copy()
    # replace 0 with -1 for island
    if sentence_key == "s_ab":
        sent_copy['wh'] = 1
        sent_copy['gap'] = 1
        sent_copy['surprisal'] = stim_set[sentence_key]['critical_surprisal']
    elif sentence_key == "s_xb":
        sent_copy['wh'] = -1
        sent_copy['gap'] = 1
        sent_copy['surprisal'] = stim_set[sentence_key]['critical_surprisal']
    elif sentence_key == "s_ax":
        sent_copy['wh'] = 1
        sent_copy['gap'] = -1
        sent_copy['surprisal'] = stim_set[sentence_key]['critical_surprisal']
    else: # s_xx
        sent_copy['wh'] = -1
        sent_copy['gap'] = -1
        sent_copy['surprisal'] = stim_set[sentence_key]['critical_surprisal']
    return sent_copy

def load_eval_data(eval_path):
    eval_path = Path(eval_path)
    df_list = []
    for model_path in eval_path.iterdir():
        model = model_path.stem
        if model_path.is_dir():
            for constr_path in model_path.iterdir():
                if constr_path.is_file() and constr_path.suffix == ".tsv":
                    constr = constr_path.stem
                    tmp = pd.read_csv(constr_path, sep="\t")
                    tmp["model"] = model
                    tmp["constr"] = constr
                    df_list.append(tmp)
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def extract_condition(df):
    df[["island", "wh", "gap"]] = df["condition"].str.extract(r"s(_i)?_(a|x)(b|x)")
    df["island"] = df["island"].map({"_i": 1, np.nan: -1})
    df["wh"] = df["wh"].map({"a": 1, "x": -1})
    df["gap"] = df["gap"].map({"b": 1, "x": -1})
    return df

def calculate_linear_mixed_effects(df):

    fg_formula = "surprisal ~ wh * gap + (1 | group)" # is it right to replace "item" with "group"?
    island_formula = "surprisal~wh*gap*island+(gap||group)"

    results = []

    models = df["model"].unique()
    constrs = df["constr"].unique()

    # total iterations for tqdm
    total = len(models) * len(constrs)

    for model, constr in tqdm(
        [(m, c) for m in models for c in constrs],
        total=total,
        desc="Fitting LME models"
    ):

        data = df[
            (df["model"] == model) &
            (df["constr"] == constr) #&
            #(df["island"] == -1)
        ]

        data = data[["group", "surprisal", "wh", "gap", "island"]]

        lme_model = lmer(island_formula, pl.DataFrame(data))
        lme_model.fit()
        result = (
            lme_model.result_fit[["term", "estimate", "p_value"]]
            .tail(4)
            .to_pandas()
            .iloc[[0, -1]] # wh:gap and wh:gap:island
        )
        result["model"] = model
        result["constr"] = constr
        results.append(result)

    return pd.concat(results, ignore_index=True)


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
    df = load_eval_data(input_path)
    df = extract_condition(df)

    # Calculate filler effects
    result = calculate_linear_mixed_effects(df)
    print(result)

    # Save results
    output_file = output_dir / f"{input_path.stem}_lme.tsv"
    result.to_csv(output_file, sep="\t", index=False)
    print(f"\nResults saved to: {output_file}")
    print("\nLME Model Results:")
    print(result)
    

if __name__ == "__main__":
    main()