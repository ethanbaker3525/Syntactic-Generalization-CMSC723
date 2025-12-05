# Syntactic-Generalization-CMSC723
 
Final project for UMD CS: CMSC723 Natural Language Processing; Fall 2025. This project aims to answer the question: **Do transformers with access to hierarchical knowledge make more human-like syntactic generalizations than transformers without this knowledge?**

Created by [Ethan Baker](ebaker35@umd.edu), [Daniel Kiely](dmkiely@umd.edu), [Annika Shankwitz](ashankwi@umd.edu), and [Chuanhai Xu](cxu12349@umd.edu)

## Data

TODO

## Training

TODO

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
