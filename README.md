# Syntactic-Generalization-CMSC723
 
Final project for UMD CS: CMSC723 Natural Language Processing; Fall 2025. This project aims to answer the question: **Do transformers with access to hierarchical knowledge make more human-like syntactic generalizations than transformers without this knowledge?**

Created by [Ethan Baker](ebaker35@umd.edu), [Daniel Kiely](dmkiely@umd.edu), [Annika Shankwitz](ashankwi@umd.edu), and [Chuanhai Xu](cxu12349@umd.edu)

## Data
Run the following code to generate the hierarchical syntax tree dataset:
```sh
python Data/serialize.py

# data stored in Data/trees/brown_trees.tsv"
```

## Training
Run the following code to train models on the hierarchical syntax tree dataset:
```sh
# set Tinker API key
export TINKER_API_KEY=your_api_key_here

# train baseline Llama-3.2-1B
python train_tinker/HS_train.py

# train a custom model 
python train_tinker/HS_train.py --model_path /path/to/previous/model

# model saved as train_tinker/weight_paths_model_HS.json. Note that these are NOT the full weight files, instead
# model weights are saved in Tinker storage. Use download_weights.py to download them.
```

Similarly, run the following code to train models on (clefting) filler gap dependencies:
```sh
# set Tinker API key
export TINKER_API_KEY=your_api_key_here

# train baseline Llama-3.2-1B
python train_tinker/FG_train.py

# train a custom model 
python train_tinker/FG_train.py --model_path /path/to/previous/model

# model saved as train_tinker/weight_paths_model_FG.json. Note that these are NOT the full weight files, instead
# model weights are saved in Tinker storage. Use download_weights.py to download them.
```

For both of the above, `/path/to/previous/model` is a string of a path in Tinker storage, i.e. `tinker://4a517769-ccad-47a9-bdf4-1c96342a6c21/weights/model_FG_epoch9`

Tinker provides a "full" path and a "sampler" path for saved models. The above uses the full paths. Sampler paths look the same except `/weights/` is replaced with `/sampler_weights/`.

## Evaluation

Run the following code to evaluate surprisals on models:

```sh
# Download the weights of a model saved on Tinker (this path is a sampler path)
python train_tinker/download_weights.py --model_path /path/to/saved/model

# This downloads an archive.tar file in train_tinker/ as well as extracts it into train_tinker/adapters/my_adapter/

# evaluate baseline model
python eval/eval.py --HF_TOKEN your_huggingface_token

# evaluate continual pre-trained model, using the model weights currently stored in train_tinker/adapters/my_adapter from running download_weights
python eval/eval.py --HF_TOKEN your_huggingface_token --cpt

# output is five .tsv files found in /eval_output
# (if replicating from scratch, would require manually moving these files before running the analysis code below)
```

## Analysis

Run the following code to analyze all models in `./eval/`
```sh
cd analysis
conda env create -f environment.yaml
conda activate syngen
python analysis.py
python plots.py 
```
outputs will be in `./analysis/outputs/`.

## Miscellaneous
Miscellaneous helper code:
```sh
# print the Tinker-recommended learning rate for Llama-3.2-1B
python train_tinker/get_lr.py

# check the format and structure of filler gap training data
python eval/verify_criticalwords.py
```