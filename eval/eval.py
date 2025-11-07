# WARNING: FILE WAS RUN ON COLAB.
# INPUT/OUTPUT FILEPATHS DO NOT NECESSARILY WORK WHEN RUN LOCALLY

import torch
import os
import math
from transformers import pipeline
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm
import pandas as pd
from peft import PeftModel
from datetime import datetime
from google.colab import userdata

cpt = True

hf_token = userdata.get("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map=device,
    token=hf_token
)

if cpt:
  model = PeftModel.from_pretrained(model, "my_adapter")

model.eval()
tokenizer.pad_token=tokenizer.eos_token

def batch_eval(batch_size, eval_file, output_directory):
  df = pd.read_csv(eval_file, sep="\t")

  # create column with underscores removed
  df["r"] = df["tokens"].str.replace(" _", "", regex=False)

  # placeholder for surprisal
  df["surprisal"] = -1.0

  for i in tqdm(range(0, len(df), batch_size)):

    # on the last batch, set batch_size to exactly the number of remaining items
    if len(df) - i < batch_size:
      batch_size = len(df) - i


    texts = df["tokens"][i:i + batch_size].tolist()
    texts_r = df["r"][i:i + batch_size].tolist()

    # tokenize texts with and without underscores together (output has length 2 * batch_size)
    enc = tokenizer(texts + texts_r, return_tensors="pt", padding=True, max_length=32, truncation=True).to(model.device)

    # extract output based on whether it was for text with or without underscores
    input_ids = enc["input_ids"][:batch_size] # (batch_size, seq_len (max of batch))
    input_ids_r = enc["input_ids"][batch_size:] # (batch_size, seq_len (max of batch))
    mask = enc["attention_mask"][:batch_size] # (batch_size, seq_len (max of batch))
    mask_r = enc["attention_mask"][batch_size:] # (batch_size, seq_len (max of batch))

    seq_len = input_ids.size(1)

    with torch.no_grad():
      out = model(input_ids=input_ids_r, attention_mask=mask_r)
      logits = out.logits # (batch_size, seq_len, vocab size)


      # find indicies of underscores and compute the surprisal at those indicies
      # in the text without underscores (adjusted for shifting from removal of underscores)
      for j in range(0, batch_size):
        input_ids_as_list = input_ids[j][mask[j].bool()].tolist()
        input_ids_as_list_r = input_ids_r[j][mask_r[j].bool()].tolist()
        tokens = [tokenizer.decode([k]) for k in input_ids_as_list]
        left_underscore_index = tokens.index(" _")
        right_underscore_index = len(tokens) - 1 - tokens[::-1].index(" _")


        total_surprisal = 0

        for target_index in range(left_underscore_index, right_underscore_index - 1):
          target_token_id = input_ids_as_list_r[target_index]
          #print(f"Target token: {tokenizer.decode([target_token_id])}")
          predicting_logits = logits[j, target_index - 1, :] # (vocab,)
          log_probs = F.log_softmax(predicting_logits, dim=0) # (vocab,)
          logp = log_probs[target_token_id].item()
          nll = -logp
          total_surprisal += nll

        #average_surprisal = total_surprisal / (right_underscore_index - 1 - left_underscore_index)
        df.loc[i+j, "surprisal"] = total_surprisal #average_surprisal

  df = df.drop("r", axis=1)
  name_without_extension, _ = os.path.splitext(eval_file)
  df.to_csv(f"{os.path.join(output_directory, name_without_extension)}_{"cpt" if cpt else "baseline"}.tsv", sep="\t")

def batch_eval_all():
  # batchsize changes results, especially on bfloat16.
  # colab gpu doesn't have enough ram for float32
  # i.e. item 2 of eval_cleft. result for batch_size=1
  # is the same as for batch_size=4, but different for batch_size=3
  #
  # https://github.com/huggingface/transformers/issues/25921
  # https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
  batch_size = 20

  files = ["eval_cleft.tsv", "eval_intro_topic.tsv", "eval_nointro_topic.tsv", "eval_tough.tsv", "eval_wh.tsv"]

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_directory = f"{"cpt" if cpt else "baseline"}_evals_{timestamp}"
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  for file in files:
    batch_eval(batch_size, file, output_directory)


def manual_test():
  # input underscores manually removed
  text = "It is these snacks that Jennifer saw recently"

  # manually computed index of (single) critical word
  # (essentially 1-index position to account for "<|begin_of_text|>" token)
  target_index = 8

  enc = tokenizer(text, return_tensors="pt", padding=True, max_length=32, truncation=True).to(model.device)
  input_ids = enc["input_ids"] # (1, seq_len)
  seq_len = input_ids.size(1)
  print("Sequence length:", seq_len)

  input_ids_as_list = enc["input_ids"][0].tolist()
  tokens = [tokenizer.decode([i]) for i in input_ids_as_list]
  print("Tokens:", tokens)

  with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=enc["attention_mask"])
    logits = out.logits # (1, seq_len, vocab)
    target_token_id = input_ids[0, target_index]
    predicting_logits = logits[0, target_index - 1, :] # (vocab,)
    log_probs = F.log_softmax(predicting_logits, dim=0) # (vocab,)
    logp = log_probs[target_token_id]
    nll = -logp
    print("Text:", text)
    print("Log prob:", logp.item())
    print("Negative log prob:", nll.item())
    print("Single-token perplexity", math.exp(nll.item()))



batch_eval_all()
#manual_test()