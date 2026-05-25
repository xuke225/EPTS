import torch
from utils.ms_hilora import set_LoRA,set_lora_active_group_ranks,merge_active_lora_to_weights
import os
import numpy as np
import argparse
from utils.pruner import Pruner
import os
from utils.reconstruction import generate_llama_uniform_rates
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data import get_loaders
from utils.eval import eval_ppl,eval_zero_shot
from utils.maskmanager import MaskManager
def get_llm(model_path,cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    model.config.max_position_embeddings = 2048
    model.seqlen = model.config.max_position_embeddings 
    return model

def count_nonzero_parameters(model):
    total_params = 0
    nonzero_params = 0
    for name,param in model.model.layers.named_parameters():
        if 'weight' not in name:
            continue
        param_count = param.data.numel()
        total_params += param_count
        nonzero_params += torch.nonzero(param.data).size(0)
    return nonzero_params, total_params
def main():
    device = torch.device("cuda")
    np.random.seed(1024)
    torch.random.manual_seed(1024)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='decapoda-research/llama-7b-hf')
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--importance_dir', type=str, default='./cache',help='path for saving saving importance')
    parser.add_argument('--prune_rate', type=float, default='0.5')
    parser.add_argument('--reconstructed_model_path', type=str, help='path for saving reconstructed model')
    args = parser.parse_args()

    model_path = args.model
    model = get_llm(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    dataset = "c4"
    train_loader, test_loader = get_loaders(
        dataset, seed=1024, seqlen=model.seqlen, tokenizer=tokenizer 
    )
    
    for block_idx in range(len(model.model.layers)):
        current_layer = model.model.layers[block_idx]
        layer_device = next(current_layer.parameters()).device
        set_LoRA(layer=current_layer,device=layer_device,block_idx=block_idx)
    
    checkpoint_path = args.reconstructed_model_path
    if os.path.exists(checkpoint_path):
        print(f"Loading : {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])

            if 'layer_flags' in checkpoint:
                layer_flags = checkpoint['layer_flags']
                for idx, layer in enumerate(model.model.layers):
                    flag_key = f'layer_{idx}'
                    if flag_key in layer_flags:
                        layer.low_group_lora_disabled = layer_flags[flag_key]
                        print(f"Layer {idx}: low_group_lora_disabled = {layer.low_group_lora_disabled}")
        else:
            model.load_state_dict(checkpoint)
            for layer in model.model.layers:
                layer.low_group_lora_disabled = False
    print("====================Pre-pruning Statistics==========================")
    nonzero_before, total_before = count_nonzero_parameters(model)
    print(f"Non-zero parameters: {nonzero_before}/{total_before} ({nonzero_before/total_before:.2%})")
    best_prune_rates = [args.prune_rate] * len(model.model.layers)
    print(f"Using uniform pruning rates: {best_prune_rates}")
    
    for block_idx, prune_rate in enumerate(best_prune_rates):
        current_layer = model.model.layers[block_idx]
        layer_device = next(current_layer.parameters()).device
        prune_rates_dict = generate_llama_uniform_rates(current_layer, prune_rate)
        print(prune_rates_dict)
        if 0 < prune_rate <= 0.4:
            if getattr(current_layer, 'low_group_lora_disabled', False):
                print(f"Layer {block_idx}: LoRA for low pruning rate group is disabled, skipping LoRA")
                set_lora_active_group_ranks(current_layer, [])
            else:
                set_lora_active_group_ranks(current_layer, [0])
        elif 0.4 < prune_rate <= 0.45:
            set_lora_active_group_ranks(current_layer, [0,1])
        elif 0.45 < prune_rate <= 0.5:
            set_lora_active_group_ranks(current_layer, [0,1,2])
        elif 0.5 < prune_rate <= 0.55:
            set_lora_active_group_ranks(current_layer, [0,1,2,3])
        elif 0.55 < prune_rate <= 0.6:
            set_lora_active_group_ranks(current_layer, [0,1,2,3,4])
        elif 0.6 < prune_rate <= 0.65:
            set_lora_active_group_ranks(current_layer, [0,1,2,3,4,5])
        elif 0.65 < prune_rate <= 0.7:
            set_lora_active_group_ranks(current_layer, [0,1,2,3,4,5,6])
        pruner = Pruner(
                model=model,
                layer=model.model.layers[block_idx],
                block_idx=block_idx,
                prune_method='wanda',
                model_type='llama',
                data_loader=train_loader,
                use_cached_wanda=True,
                nsamples=128
            )
        pruner.load_wanda_cache_per_layer(block_idx,importance_dir=args.importance_dir,device=layer_device)
        pruner.apply_pruning(prune_rates_dict,block_idx=block_idx)
        merge_active_lora_to_weights(current_layer)
        pruner.clear_wanda_cache_for_block(block_idx)
        pruner.cleanup()
        MaskManager.clear_block_masks(block_idx)
        del pruner

    print("===========================Post-pruning Statistics==============================")
    nonzero_after, total_after = count_nonzero_parameters(model)
    global_prune_rate = (total_before - nonzero_after)/total_before
    print(f"Non-zero parameters: {nonzero_after}/{total_after} ({nonzero_after/total_after:.2%})") 
    print(f"compression rate:: {global_prune_rate:.2%}")
    device = model.hf_device_map["lm_head"]
    ppl_test = eval_ppl(model,tokenizer,device)
    print(f"wikitext perplexity {ppl_test}")

    # accelerate = False
    # task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
    # num_shot = 0
    # eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)

if __name__ == '__main__':
    main()