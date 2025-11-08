import pandas as pd

eval_file = "eval_cleft"

baseline = f"eval/baseline/{eval_file}.tsv"
df = pd.read_csv(baseline, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

howitt0 = f"eval/howitt0/{eval_file}.tsv"
df = pd.read_csv(howitt0, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

howitt1 = f"eval/howitt1/{eval_file}.tsv"
df = pd.read_csv(howitt1, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

howitt2 = f"eval/howitt2/{eval_file}.tsv"
df = pd.read_csv(howitt2, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

howitt5 = f"eval/howitt5/{eval_file}.tsv"
df = pd.read_csv(howitt5, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

howitt10 = f"eval/howitt10/{eval_file}.tsv"
df = pd.read_csv(howitt10, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())

howitt50 = f"eval/howitt50/{eval_file}.tsv"
df = pd.read_csv(howitt50, sep="\t")
print(f"Mean surprisal in {eval_file} grouped by condition")
print(df.groupby("condition")["surprisal"].mean())