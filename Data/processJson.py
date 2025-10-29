import json
import sys
from collections import defaultdict

with open(sys.argv[1], 'r') as file:
    data = json.load(file)

with open(sys.argv[2], "w+") as f:
    f.write("condition\ttokens\tcriticalwords\tgrammaticality\n")

    for item in data:
        s_ab = " ".join(item["s_ab"]['original_tokens']).strip("*").strip(" ")
        s_ab_crit = ",".join(item["s_ab"]['critical_tokens'])
        s_ab_gram = str(item["s_ab"]['grammatical'])
        f.write("s_ab" + "\t" + s_ab + "\t" + s_ab_crit + "\t" +s_ab_gram + "\n")


        s_xb = " ".join(item["s_xb"]['original_tokens']).strip("*").strip(" ")
        s_xb_crit = ",".join(item["s_xb"]['critical_tokens'])
        s_xb_gram = str(item["s_xb"]['grammatical'])
        f.write("s_xb" + "\t" + s_xb + "\t" + s_xb_crit + "\t" +s_xb_gram + "\n")

        s_ax = " ".join(item["s_ax"]['original_tokens']).strip("*").strip(" ")
        s_ax_crit = ",".join(item["s_ax"]['critical_tokens'])
        s_ax_gram = str(item["s_ax"]['grammatical'])
        f.write("s_ax" + "\t" + s_ax + "\t" + s_ax_crit + "\t" +s_ax_gram + "\n")

        s_xx = " ".join(item["s_xx"]['original_tokens']).strip("*").strip(" ")
        s_xx_crit = ",".join(item["s_xx"]['critical_tokens'])
        s_xx_gram = str(item["s_xx"]['grammatical'])
        f.write("s_xx" + "\t" + s_xx + "\t" + s_xx_crit + "\t" +s_xx_gram + "\n")

f.close()
