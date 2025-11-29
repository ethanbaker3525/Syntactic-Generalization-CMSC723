import pandas as pd
import re

output = "analysesCombined"

df = pd.read_csv("fge.tsv", sep="\t")

#Simple ----------------------
simpleLME = pd.read_csv("simple_lme.tsv", sep="\t")
simple = df[df["island"] == -1]
simpleResult = simple.groupby(["model", "constr", "gap"], as_index=False)["effect"].mean()

#organize the fge
simpleClean = simpleResult.pivot(index=["model", "constr"], columns="gap", values="effect")
simpleClean = simpleClean.rename(columns={-1: "-gap effect", 1: "+gap effect"})
simpleClean["effectDifference"] =(simpleClean['-gap effect'] - simpleClean["+gap effect"]).abs()
simpleClean["-gap direction"] = simpleClean["-gap effect"].apply(lambda x: "p" if x > 0 else ("n" if x < 0 else 'zero'))
simpleClean["+gap direction"] = simpleClean["+gap effect"].apply(lambda x: "p" if x > 0 else ("n" if x < 0 else 'zero'))
simpleClean.to_csv(f"{output}/simpleFGE.tsv", sep="\t", float_format="%.3f")

#add the lme infor
simpleCleanWithLME = simpleClean.merge(
    simpleLME[["model", "constr", "term", "estimate", "p_value", "sig"]],
    on = ["model", "constr"],
    how="right"
)

#add info about the estimate direction
simpleCleanWithLME["estimateDirection"] = simpleCleanWithLME["estimate"].apply(lambda x: "p" if x > 0 else ("n" if x < 0 else 'zero'))
simpleCleanWithLME.to_csv(f"{output}/simpleAllInfo.tsv", sep="\t", float_format="%.3f")

#filter out the inccorrect lines
simpleCorrect = simpleCleanWithLME[(simpleCleanWithLME["-gap direction"] == "p") & (simpleCleanWithLME["+gap direction"] == "n") & (simpleCleanWithLME["estimateDirection"] == "n") & (simpleCleanWithLME["sig"].str.match("^\*{1,3}$"))]
simpleCorrect.to_csv(f"{output}/simpleCorrect.tsv", sep="\t", float_format="%.3f")




#Island -----------------------
island = df[df["island"] == 1]
islandLME = pd.read_csv("island_lme.tsv", sep="\t")
islandResult = island.groupby(["model", "constr", "gap"], as_index=False)["effect"].mean()


#organize the fge
islandClean = islandResult.pivot(index=["model", "constr"], columns="gap", values="effect")
islandClean = islandClean.rename(columns={-1: "-gap effect", 1: "+gap effect"})
islandClean["effectDifference"] = (islandClean['-gap effect'] - islandClean["+gap effect"]).abs()
islandClean["-gap direction"] = islandClean["-gap effect"].apply(lambda x: "p" if x > 0 else ("n" if x < 0 else 'zero'))
islandClean["+gap direction"] = islandClean["+gap effect"].apply(lambda x: "p" if x > 0 else ("n" if x < 0 else 'zero'))
islandClean.to_csv(f"{output}/islandFGE.tsv", sep="\t", float_format="%.3f")

#add the lme infor
islandCleanWithLME = islandClean.merge(
    islandLME[["model", "constr", "term", "estimate", "p_value", "sig"]],
    on = ["model", "constr"],
    how="right"
)

#add info about the estimate direction
islandCleanWithLME["estimateDirection"] = islandCleanWithLME["estimate"].apply(lambda x: "p" if x > 0 else ("n" if x < 0 else 'zero'))
islandCleanWithLME.to_csv(f"{output}/islandAllInfo.tsv", sep="\t", float_format="%.3f")
 
#filter out the inccorrect lines
islandCorrect = islandCleanWithLME[(simpleCleanWithLME["-gap direction"] == "p") & (islandCleanWithLME["+gap direction"] == "n") & (islandCleanWithLME["estimateDirection"] == "p") & (islandCleanWithLME["sig"].str.match("^\*{1,3}$"))]
islandCorrect.to_csv(f"{output}/islandCorrect.tsv", sep="\t", float_format="%.3f")

