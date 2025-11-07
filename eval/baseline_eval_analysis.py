import pandas as pd

eval_file = "eval_cleft"

baseline_sum = f"eval/baseline_evals_sum/{eval_file}_baseline.tsv"
df = pd.read_csv(baseline_sum, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

cpt = f"eval/cpt_evals_20251107_203314/{eval_file}_cpt.tsv"
df = pd.read_csv(cpt, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

cpt_0 = f"eval/cpt_0/{eval_file}_cpt.tsv"
df = pd.read_csv(cpt, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())