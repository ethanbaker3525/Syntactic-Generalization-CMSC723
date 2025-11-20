import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
    
        fge = (data[["gap", "island", "effect"]])

        plot_fge(fge, f"Filler Effects by Construction and Gap Type ({model}, {constr})", output_path / model / f"fge_{model}_{constr}.png")

if __name__ == "__main__":
    main()