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

## Evaluation

TODO

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
