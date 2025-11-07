import tinker
from tinker import types
import pandas as pd
from tqdm import tqdm

from pathlib import Path
from datetime import datetime

epochs = 0
batch_size = 20
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

for _ in tqdm(range(epochs)):
  for i in tqdm(range(0, len(train_list), batch_size)):
      batch = make_batch(train_list[i:i+batch_size])
      training_client.forward_backward(batch, loss_fn="cross_entropy").result()
      training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()


# SWITCH TO OTHER SAVE METHOD TO CONTINUE TRAINING AFTER
sampling_path = training_client.save_weights_for_sampler(name=f"epoch_{epochs}").result().path

base_dir = Path(__file__).resolve().parent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
outfile = base_dir / f"weights_path_{timestamp}.txt"
with outfile.open("w", encoding="utf-8") as f:
    f.write(sampling_path)