### THIS FILE *WILL* REPRODUCE THE HOWITT EXPERIMENT ###

import pandas as pd

import surprisal

data_path = "Data/splits/eval_cleft.tsv"

df = pd.read_csv(data_path, sep="\t")

print(df)
