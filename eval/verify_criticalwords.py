# Verifies that in every eval .tsv file, there are exactly 2 underscores
# and the concatenation of criticalwords (i.e. "yesterday" => "yesterday", "the,cheese" => "the cheese")
# always appears between the underscores

import pandas as pd
import os

splits_folder = "Data/splits/"

m = 0

for name in os.listdir(splits_folder):
    if name.startswith("eval_"):
      
      df = pd.read_csv(os.path.join(splits_folder, name), sep="\t")

      for i in range(len(df)):
        tokens = df["tokens"][i]
        m = max(m, len(tokens.split()))
        criticalwords = df["criticalwords"][i]
        concat = " ".join(criticalwords.split(","))

        # 2 spaces + right "_" + exclusivity of end in [start:end] = 4
        if tokens.count("_") != 2 or tokens[tokens.index("_"): tokens.index("_") + len(concat) + 4] != f"_ {concat} _":
          print(f"Error on row {i} of eavl_wh.tsv: underscores don't surround critical words")
          raise Exception("Error")
        
print("Everything checks out if no exceptions were raised and this prints")
print(m)