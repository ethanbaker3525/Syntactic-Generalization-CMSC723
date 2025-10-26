import json
from collections import defaultdict

with open("basic_subj_wh_grnn.json", 'r') as file:
    data = json.load(file)

with open("basic_subj_wh.tsv", "w+") as f:
    f.write("condition\ttokens\tcriticalwords\tgrammaticality\n")

    foundItems = defaultdict(lambda:0)
    for item in data:
        s_ab = " ".join(item["s_ab"]['original_tokens']).strip("*").strip(" ")
        s_ab_crit = ",".join(item["s_ab"]['critical_tokens'])
        s_ab_gram = str(item["s_ab"]['grammatical'])
        s_ab_all = "s_ab" + "\t" + s_ab + "\t" + s_ab_crit + "\t" +s_ab_gram + "\n"

        s_xb = " ".join(item["s_xb"]['original_tokens']).strip("*").strip(" ")
        s_xb_crit = ",".join(item["s_xb"]['critical_tokens'])
        s_xb_gram = str(item["s_xb"]['grammatical'])
        s_xb_all = "s_xb" + "\t" + s_xb + "\t" + s_xb_crit + "\t" +s_xb_gram + "\n"

        s_ax = " ".join(item["s_ax"]['original_tokens']).strip("*").strip(" ")
        s_ax_crit = ",".join(item["s_ax"]['critical_tokens'])
        s_ax_gram = str(item["s_ax"]['grammatical'])
        s_ax_all = "s_ax" + "\t" + s_ax + "\t" + s_ax_crit + "\t" +s_ax_gram + "\n"

        s_xx = " ".join(item["s_xx"]['original_tokens']).strip("*").strip(" ")
        s_xx_crit = ",".join(item["s_xx"]['critical_tokens'])
        s_xx_gram = str(item["s_xx"]['grammatical'])
        s_xx_all = "s_xx" + "\t" + s_xx + "\t" + s_xx_crit + "\t" +s_xx_gram + "\n"

        if foundItems[s_ab_all] == 0:
            f.write(s_ab_all)

        if foundItems[s_xb_all] == 0:
            f.write(s_xb_all)
        
        if foundItems[s_ax_all] == 0:
            f.write(s_ax_all)
        
        if foundItems[s_xx_all] == 0:
            f.write(s_xx_all)
f.close()
