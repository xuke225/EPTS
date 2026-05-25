import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from utils.pruner import Pruner
from utils.data import get_loaders
import argparse
import gc
import os
def get_llm(model_path, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.config.max_position_embeddings = 2048
    model.seqlen = model.config.max_position_embeddings
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    np.random.seed(1024)
    torch.random.manual_seed(1024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune_method', type=str, default='wanda')
    parser.add_argument('--model', type=str, default='decapoda-research/llama-7b-hf')
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--importance_dir', type=str, default='./cache',help='path for saving saving importance')
    args = parser.parse_args()

    model_path = args.model
    model = get_llm(model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    print(model.seqlen)
    data_loader, _ = get_loaders("c4",nsamples=args.nsamples,seed=1024,seqlen=model.seqlen,tokenizer=tokenizer)
    
    for layer_idx in range(len(model.model.layers)):
        wanda_pruner = Pruner(
                model=model,
                layer=model.model.layers[layer_idx],
                block_idx=layer_idx,
                prune_method=args.prune_method,
                model_type=args.model_type,
                data_loader=data_loader,
                use_cached_wanda=False,
                nsamples=args.nsamples
            )
        wanda_pruner.precompute_wanda_importance_per_layer(importance_dir=args.importance_dir)
        del wanda_pruner
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()