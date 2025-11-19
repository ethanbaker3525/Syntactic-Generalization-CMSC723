import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from pymer4.models import lmer

def extract_condition(df):
    df[["island", "filler", "gap"]] = df["condition"].str.extract(r"s(_i)?_(a|x)(b|x)")
    df["island"] = df["island"].map({"_i": 1, np.nan: -1})
    df["filler"] = df["filler"].map({"a": 1, "x": -1})
    df["gap"] = df["gap"].map({"b": 1, "x": -1})
    return df

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

def calculate_filler_gap_effects(df):

    results = []

    models = df["model"].unique()
    constrs = df["constr"].unique()

    # total iterations for tqdm
    total = len(models) * len(constrs)

    for model, constr in tqdm(
        [(m, c) for m in models for c in constrs],
        total=total,
        desc="Calculating filler gap effects"
    ):

        data = df[
            (df["model"] == model) &
            (df["constr"] == constr) #&
            #(df["island"] == -1)
        ]

        data = data[["group", "surprisal", "filler", "gap", "island"]]
        pivoted = data.pivot_table(
            index=["group", "gap", "island"],
            columns="filler",
            values="surprisal"
        )

        pivoted["effect"] = pivoted[1] - pivoted[-1]
        pivoted["model"] = model
        pivoted["constr"] = constr
        pivoted = pivoted.reset_index()
        pivoted = pivoted[["model", "constr", "group", "gap", "island", "effect"]]
        results.append(pivoted)

    return pd.concat(results, ignore_index=True)

def calculate_linear_mixed_effects(df):

    fg_formula = "surprisal~filler*gap+(1|group)" # is it right to replace "item" with "group"?
    island_formula = "surprisal~filler*gap*island+(gap||group)"

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

        data = data[["group", "surprisal", "filler", "gap", "island"]]

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
        result["sig"] = result["p_value"].map(lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if x < 0.05 else "." if x < 0.1 else "")
        results.append(result)

    return pd.concat(results, ignore_index=True)


def main():
    '''
    Produces filled gap effect and LME model results for all eval surprisals
    '''
    parser = argparse.ArgumentParser(description='Calculate linear mixed effect model from surprisal data')
    parser.add_argument('--eval_file_path', type=str, required=False, default='./eval/',
                        help='Path to the evaluation eval folder')
    parser.add_argument('--output_path', type=str, required=False, default='./analysis/output',
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

    # filler gap effects
    fge_df = calculate_filler_gap_effects(df)

    # LME model
    lme_df = calculate_linear_mixed_effects(df)

    # Save results
    fge_df.to_csv(output_dir / "fge.tsv", sep="\t", index=False)
    lme_df.to_csv(output_dir / "lme.tsv", sep="\t", index=False)
    print(f"Saved results to: {output_dir}")
    

if __name__ == "__main__":
    main()