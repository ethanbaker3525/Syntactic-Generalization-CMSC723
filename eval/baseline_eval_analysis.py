import pandas as pd

eval_file = "eval_cleft"

df = pd.read_csv(f"eval/baseline_evals/{eval_file}_baseline.tsv", sep="\t")

print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())