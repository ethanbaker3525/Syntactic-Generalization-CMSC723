import argparse
import tinker
from tinker import types
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import os
import json
import asyncio
from pathlib import Path
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator


def main(model_path=None):
    # setup Tinker + base model
    service_client = tinker.ServiceClient()
    base_model = 'meta-llama/Llama-3.2-1B'
    training_client = service_client.create_lora_training_client(
        base_model=base_model
    )
    tokenizer = training_client.get_tokenizer()
    
    model_name = "model_HS"
    weight_output_file = os.path.join(Path(__file__).resolve().parent, f"weights_path_{model_name}.json")

    # if necessary, load state of prior model
    if model_path is not None:
        training_client.load_state(model_path)

    # create batch of examples
    def create_batch(trees: List[str], max_length=256) -> List[types.Datum]:
        batch = []
        for tree in trees:
            tokens = tokenizer.encode(tree, add_special_tokens=True)[:max_length]
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            weights = [1] * len(target_tokens)
            batch.append(types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens)
            ))
        return batch

    # load the data, make splits
    tree_data = pd.read_csv('Data/trees/brown_trees.tsv', sep='\t')
    trees = tree_data['tree'].to_list() # list of syntax trees
    trees_train, temp = train_test_split(trees, test_size=0.2, random_state=47)
    trees_val, trees_test = train_test_split(temp, test_size=0.5, random_state=47)

    val_batch = create_batch(trees_val)
    val_evaluator = NLLEvaluator(data=val_batch) # from Tinker cookbook

    # hyperparameters
    epochs = 3
    batch_size = 32
    learning_rate = 0.0004908239409722158  # from Tinker docs

    def save_model(training_client, epoch):
        sampling_path = training_client.save_weights_for_sampler(name=f"{model_name}_epoch{epoch+1}").result().path
        full_path = training_client.save_state(name=f"{model_name}_epoch{epoch + 1}").result().path
        with open(weight_output_file, "w", encoding="utf-8") as f:
            json.dump({f"{model_name}_epoch{epoch+1}": {"sampling_path": sampling_path, "full_path": full_path}}, f, indent=4)

    # training loop
    random.seed(47)
    min_val_loss = float('inf')
    best_epoch = -1
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_weights = 0.0
        random.shuffle(trees_train)
        with tqdm(total=len(trees_train), desc=f"Epoch {epoch+1}") as pbar:
            for i in range(0, len(trees_train), batch_size):
                batch = create_batch(trees_train[i:i+batch_size])
                fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
                optim_future = training_client.optim_step(types.AdamParams(learning_rate=learning_rate))
                fwdbwd_result = fwdbwd_future.result()
                optim_result = optim_future.result()
        
                logprobs = np.concatenate([output['logprobs'].tolist() for output in fwdbwd_result.loss_fn_outputs])
                weights = np.concatenate([example.loss_fn_inputs['weights'].tolist() for example in batch])
                
                batch_loss_sum=(-np.dot(logprobs, weights))
                epoch_loss += batch_loss_sum
                epoch_weights += weights.sum()
                pbar.update(len(batch))
        
        # output training loss    
        print(f"Training loss after epoch {epoch+1}: {epoch_loss / epoch_weights:.4f}")
        
        # output validation loss
        val_loss = asyncio.run(val_evaluator(training_client))['nll']
        print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")
        
        # if this is the best model (according to test loss), then save it
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            save_model(training_client, epoch)
     
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Syntactic Tree Training')
    parser.add_argument('--model_path', type=str, required=False)
    args = parser.parse_args()

    main(model_path=args.model_path)
    
    
    