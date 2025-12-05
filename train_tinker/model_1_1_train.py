import tinker
from tinker import types
import pandas as pd
from tqdm import tqdm
import random
import os
import json
from sklearn.model_selection import train_test_split
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
import asyncio

from pathlib import Path

name = "model_1_1_3B"
#learning_rate = 0.0004908239409722158 #1B
learning_rate = 0.00035760078614950245 #3B

epochs = 10
batch_size = 32 
weights_path_file = os.path.join(Path(__file__).resolve().parent, f"weight_paths_{name}.json")

model = "meta-llama/Llama-3.2-3B"


train_df = pd.read_csv("Data/splits/train_cleft.tsv", sep="\t")

# remove duplicates
train_list = list(set(train_df["tokens"].to_list()))

# val2_df = pd.read_csv("Data/splits/eval_cleft.tsv", sep="\t")
# # train set contains only s_ab and s_xx conditions
# val2_list = val2_df[(val2_df["condition"] == "s_ab") | (val2_df["condition"] == "s_xx")]["tokens"].to_list()
# # train set doesn't have underscores
# val2_list = [x.replace(" _", "") for x in val2_list]

service_client = tinker.ServiceClient()

training_client = service_client.create_lora_training_client(base_model=model)

#training_client.load_state("tinker://a72b0f91-d935-40fd-aa4d-f9e75f8ea8c6/weights/trees2")

tokenizer = training_client.get_tokenizer() 

def make_batch(texts, max_len=32):
    batch = []
    for txt in texts:
        ids = tokenizer.encode(txt, add_special_tokens=True)[:max_len]

        # target for each token the the next token
        # target for last token is eos token
        target = ids[1:] + [tokenizer.eos_token_id]


        # weight for loss function. don't include prediction for first token in loss
        # (would be predicting the first token only from sos token)
        weights = [0] + [1] * (len(ids) - 1)
        
        # pack into 1 training sample in Tinker's format
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens=ids),
            loss_fn_inputs={"target_tokens": target, "weights": weights},
        )
        batch.append(datum)
    return batch

weight_paths = {}

train_evaluator = NLLEvaluator(data=make_batch(train_list))
# val1_evaluator = NLLEvaluator(data=make_batch(val1_list)) 
# val2_evaluator = NLLEvaluator(data=make_batch(val2_list)) 

random.seed(47)
for epoch in tqdm(range(1, epochs + 1)):
  random.shuffle(train_list)

  for i in range(0, len(train_list), batch_size):
      batch = make_batch(train_list[i:i+batch_size])
      training_client.forward_backward(batch, loss_fn="cross_entropy").result()
      training_client.optim_step(types.AdamParams(learning_rate=learning_rate)).result()

  sampler_path = training_client.save_weights_for_sampler(name=f"{name}_epoch{epoch}").result().path
  full_path = training_client.save_state(name=f"{name}_epoch{epoch}").result().path
  weight_paths[f"{name}_epoch{epoch}"] = {"sampler": sampler_path, "full": full_path}
  with open(weights_path_file, "w", encoding="utf-8") as f:
      json.dump(weight_paths, f, indent=4) 

  train_loss = asyncio.run(train_evaluator(training_client))['nll']
  print(f"Train loss after epoch {epoch}: {train_loss:.4f}")
