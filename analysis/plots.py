import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

def plot_fge(df, title, output_path):
    # Prepare dataframe
    result_sorted = df.copy()
    result_sorted = result_sorted.sort_values(['island', 'gap'], ascending=[True, False])

    result_sorted['island'] = result_sorted['island'].map({1: "island", -1: "simple"})
    result_sorted['gap'] = result_sorted['gap'].map({1: "+gap", -1: "-gap"})

    # Use seaborn’s default palette unless you want to enforce custom colors:
    palette = {"+gap": '#1f77b4', "-gap": '#ff7f0e'}

    # Create catplot
    g = sns.catplot(
        data=result_sorted,
        kind="bar",
        x="island",
        y="effect",
        hue="gap",
        palette=palette,
        height=6, aspect=1.3,
        legend=True
    )
    g._legend.set_loc("upper right")

    # Add horizontal 0-line
    ax = g.ax
    ax.set_ylim(-8, 8)
    ax.axhline(0, color='black', linewidth=1)

    # Add bar value labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=12)

    # Labels and title
    ax.set_xlabel("Construction type")
    ax.set_ylabel("Mean Filler Effect")
    ax.set_title(title)

    # Improve layout + save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")

def plot_paper_figure(df, output_path):

    df = (
        df
        .pivot_table(
            index=["model", "constr", "island", "group"],
            columns="gap",
            values="effect"
        )
        .assign(effect=lambda x: x[-1] - x[1])
        .reset_index()
        .drop(columns=[1, -1])
    )

    df = df.groupby(["model", "constr", "island"])["effect"].mean().reset_index()


    # Group by constr and take only the largest effect per model where island == -1
    df_simple = df[df["island"] == -1]
    #df_simple[df_simple["effect"] < 0] = np.nan
    df_simple = (
        df_simple
        .sort_values(by=["model", "constr", "effect"], ascending=[True, True, False])
        .groupby(["model", "constr"])
        .first()
        .reset_index()
    )

    # get the max effect per constr (including the model that gave the max effect)
    max_effects_baseline = df_simple[df_simple["model"]=="baseline"].groupby("constr")[["effect", "model"]].apply(lambda x: x.loc[x["effect"].idxmax()]).reset_index()
    max_effects_baseline["baseline"] = "Baseline"
    max_effects_models = df_simple[df_simple["model"]!="baseline"].groupby("constr")[["effect", "model"]].apply(lambda x: x.loc[x["effect"].idxmax()]).reset_index()
    max_effects_models["baseline"] = "Post Trained"
    max_effects = pd.concat([max_effects_baseline, max_effects_models], ignore_index=True)
    max_effects["island"] = "simple"

    # group by constr and take only the effect closest to 0 per model where island == 1
    df_island = df[df["island"] == 1]
    df_island = (
        df_island
        .sort_values(by=["model", "constr", "effect"], ascending=[True, True, True])
        .groupby(["model", "constr"])
        .first()
        .reset_index()
    )
    # get the abs min effect per constr (including the model that gave the abs min effect)
    min_effects_baseline = df_island[df_island["model"]=="baseline"].groupby("constr")[["effect", "model"]].apply(lambda x: x.loc[x["effect"].abs().idxmin()]).reset_index()
    min_effects_baseline["baseline"] = "Baseline"
    min_effects_models = df_island[df_island["model"]!="baseline"].groupby("constr")[["effect", "model"]].apply(lambda x: x.loc[x["effect"].abs().idxmin()]).reset_index()
    min_effects_models["baseline"] = "Post Trained"
    min_effects = pd.concat([min_effects_baseline, min_effects_models], ignore_index=True)
    min_effects["island"] = "island"

    df = pd.concat([max_effects, min_effects], ignore_index=True)
    # map models to their proper names
    df["model"] = df["model"].map({
        "baseline": "Baseline",
        "model_1_1":"Model-FG",
        "model_1_2":"Model-FG-HS",
        "model_2_1":"Model-HS",
        "model_2_2":"Model-HS-FG"
    })

    # map constr to proper names
    df["constr"] = df["constr"].map({
        "eval_cleft": "Cleft",
        "eval_intro_topic": "Topicalization\n(with intro)",
        "eval_nointro_topic": "Topicalization\n(no intro)",
        "eval_tough": "Tough Movement",
        "eval_wh": "Wh-Movement"
    })

    df = df.sort_values(by=["island", "constr", "baseline"], ascending=[False, True, True])
    print(df)
    df["x_label"] = df["constr"] + "\n" + df["baseline"]
    #print(df)
    

    g = sns.catplot(
        data=df,
        kind="bar",
        x="x_label",
        y="effect",
        row="island",
        hue="model",
        height=4, aspect=3.5,
        legend=True,
        sharex=True,
        sharey=False
    )

    # set title for the entire figure
    plt.subplots_adjust(top=0.9)
    g.figure.suptitle("Mean Filler Effects by Construction Type and Model", fontsize=16)

    g._legend.set_loc("upper right")

    g.set_axis_labels("Construction type", "Mean Filler Effect")

    for ax in g.axes.flat:
        if "simple" in ax.get_title():
            ax.set_ylabel("Mean Filler Effect (larger is better)")
        else:
            ax.set_ylabel("Mean Filler Effect (smaller is better)")
    g.set_titles("{row_name} constructions")
    

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Paper figure saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot results from analysis.py')
    parser.add_argument('--output_path', type=str, required=False, default='./analysis/output',
                        help='Path to output directory for results')
    parser.add_argument("--main_figure", action='store_true',
                        help='Generate main paper figure')
    args = parser.parse_args()

    output_path = Path(args.output_path)

    df = pd.read_csv(output_path / "fge.tsv", sep="\t")

    if args.main_figure:
        plot_paper_figure(df, output_path / "main_figure.png")
        return

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
    
        fge = (data[["gap", "island", "effect"]])

        plot_fge(fge, f"Filler Effects by Construction and Gap Type ({model}, {constr})", output_path / model / f"fge_{model}_{constr}.png")

if __name__ == "__main__":
    main()