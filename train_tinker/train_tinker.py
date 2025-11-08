import tinker
from tinker import types
import pandas as pd
from tqdm import tqdm
import random
import os
import json

from pathlib import Path

epochs = 1000
batch_size = 1000 # 1000 is larger than number of training samples in train_cleft.tsv, so 1 batch = 1 epoch
weights_path_file = os.path.join(Path(__file__).resolve().parent, "weight_paths_howitt.json")

model = "meta-llama/Llama-3.2-3B"


train_df = pd.read_csv("Data/splits/train_cleft.tsv", sep="\t")
train_list = train_df["tokens"].to_list()

service_client = tinker.ServiceClient()

training_client = service_client.create_lora_training_client(base_model=model)
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

for epoch in tqdm(range(epochs)):

  if epoch <= 10 or epoch in [15, 25, 35, 45] or epoch % 10 == 0:
    sampler_path = training_client.save_weights_for_sampler(name=f"howitt{epoch}").result().path
    full_path = training_client.save_state(name=f"howitt{epoch}").result().path
    weight_paths[f"howitt{epoch}"] = {"sampler": sampler_path, "full": full_path}
    with open(weights_path_file, "w", encoding="utf-8") as f:
        json.dump(weight_paths, f, indent=4) 

  random.shuffle(train_list)

  for i in range(0, len(train_list), batch_size):
      batch = make_batch(train_list[i:i+batch_size])
      training_client.forward_backward(batch, loss_fn="cross_entropy").result()
      training_client.optim_step(types.AdamParams(learning_rate=0.0004908239409722158)).result()