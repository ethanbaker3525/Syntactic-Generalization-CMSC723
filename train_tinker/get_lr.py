# script provided by Tinker Cookbook to estimate optimal learning rate for a given model

from tinker_cookbook.hyperparam_utils import get_lr
model_name = "meta-llama/Llama-3.2-1B"
recommended_lr = get_lr(model_name)
print(f"Recommended LR: {recommended_lr}")