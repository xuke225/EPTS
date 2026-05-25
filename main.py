import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os
from utils.reconstruction import reconstruction_llama
from utils.eval import eval_ppl
from utils.data import get_loaders
import argparse
import torch.nn.functional as F
import time

def get_llm(model_path, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    model.config.max_position_embeddings = 2048
    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device:{device}")
    
    np.random.seed(1024)
    torch.random.manual_seed(1024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune_method', type=str, default='wanda')
    parser.add_argument('--group_l1',type=list,default=[0.0,0.4])
    parser.add_argument('--group_l2',type=list,default=[0.4,0.45])
    parser.add_argument('--group_l3',type=list,default=[0.45,0.5])
    parser.add_argument('--group_m1',type=list,default=[0.5,0.55])
    parser.add_argument('--group_m2',type=list,default=[0.55,0.6])
    parser.add_argument('--group_h1',type=list,default=[0.6,0.65])
    parser.add_argument('--group_h2',type=list,default=[0.65,0.7])
    parser.add_argument('--fusion_level', type=str, default="datasets")
    parser.add_argument('--model', type=str, default='decapoda-research/llama-7b-hf')
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--importance_dir', type=str, default='./cache',help='path for saving wanda importance')
    parser.add_argument('--reconstructed_model_path', type=str, default=None, 
                        help='path for saving the reconstructed model')
    args = parser.parse_args()

    model_path = args.model
    model = get_llm(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    dataset = "c4"
    train_loader, test_loader = get_loaders(
        dataset, nsamples=args.nsamples,seed=1024, seqlen=model.seqlen, tokenizer=tokenizer 
    )
    
    print(f"Model device distribution: {model.hf_device_map}")

    start_time = time.time()
    reconstruction_llama(
        original_model=model,
        data_loader=train_loader,   
        device=device,
        args=args
    )

    print("Reconstruction complete")
    end_time = time.time()
    print("Reconstruction time:{}s".format(end_time-start_time))
    
    reconstructed_model_path = args.reconstructed_model_path
    layer_flags = {}
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, 'low_group_lora_disabled'):
            layer_flags[f'layer_{idx}'] = layer.low_group_lora_disabled
    save_dict = {
        'model_state': model.state_dict(),
        'layer_flags': layer_flags
    }
    torch.save(save_dict, reconstructed_model_path)
    print(f"Reconstructed model saved to: {reconstructed_model_path}")
    print(f"Saved flags for {len(layer_flags)} layers")
    
if __name__ == "__main__":
    main()