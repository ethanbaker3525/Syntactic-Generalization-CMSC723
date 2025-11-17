import pandas as pd

eval_file = "eval_cleft"

baseline = f"eval/baseline/{eval_file}.tsv"
df = pd.read_csv(baseline, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

temp0 = f"eval/temp0/{eval_file}.tsv"
df = pd.read_csv(temp0, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

temp1 = f"eval/temp1/{eval_file}.tsv"
df = pd.read_csv(temp1, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

temp2 = f"eval/temp2/{eval_file}.tsv"
df = pd.read_csv(temp2, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

temp3 = f"eval/temp3/{eval_file}.tsv"
df = pd.read_csv(temp3, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

temp10 = f"eval/temp10/{eval_file}.tsv"
df = pd.read_csv(temp10, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())