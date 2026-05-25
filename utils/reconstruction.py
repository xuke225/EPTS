import torch
import torch.nn as nn
from utils.imagenet_utils import DataSaverHook,StopForwardException
from utils.pruner import Pruner
from utils.ms_hilora import set_LoRA,set_lora_active_group_ranks,set_lora_enabled,set_lora_mask_enabled,set_lora_ranks_requires_grad
import logging
from datetime import datetime
import os
import random
import gc
from utils.maskmanager import MaskManager
class UniformRandomGenerator:
    def __init__(self, value_range, total_samples):
        self.low = min(value_range)
        self.high = max(value_range)
        self.total_samples = total_samples
        self.generated_count = 0

        self.interval_width = (self.high - self.low) / total_samples
        self.intervals = []
        
        for i in range(total_samples):
            start = self.low + i * self.interval_width
            end = start + self.interval_width
            self.intervals.append((start, end))

        random.shuffle(self.intervals)
    
    def next(self):
        if self.generated_count >= self.total_samples:
            raise StopIteration("All samples generated")
        
        start, end = self.intervals[self.generated_count]
        sample = random.uniform(start, end)
        
        if sample <= self.low:
            sample = random.uniform(start + 1e-6, end)  
        
        if sample > self.high:
            sample = self.high 

        self.generated_count += 1
        return round(sample, 3)
def setup_logger():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger('LoRA_Pruning')
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
def generate_llama_uniform_rates(block, target_sparsity, erk_power_scale=1.0):
    block_layers = {
        'q_proj': block.self_attn.q_proj,
        'k_proj': block.self_attn.k_proj,
        'v_proj': block.self_attn.v_proj,
        'o_proj': block.self_attn.o_proj,
        'gate_proj': block.mlp.gate_proj,
        'up_proj': block.mlp.up_proj,
        'down_proj': block.mlp.down_proj
    }
    prune_rates = {}
    for layer_type in block_layers:
        prune_rates[layer_type] = target_sparsity
    
    return prune_rates

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()
def get_batch_outputs(layer, inps, attention_mask, indices, device,position_ids=None):
    if isinstance(indices, slice):
        indices = list(range(indices.start, indices.stop))
    elif isinstance(indices, int):
        indices = [indices]
    
    batch_size = len(indices)

    if batch_size == 1:
        test_input = inps[indices[0]:indices[0]+1].to(device)
        test_mask = attention_mask[indices[0]:indices[0]+1].to(device) if attention_mask is not None else None
        test_position_ids = position_ids[indices[0]:indices[0]+1].to(device) if position_ids is not None else None
        
        output = layer(
            hidden_states=test_input,
            attention_mask=test_mask,
            position_ids=test_position_ids
        )[0]
        return output

    single_outputs = []
    for idx in indices:
        test_input = inps[idx:idx+1].to(device)
        test_mask = attention_mask[idx:idx+1].to(device) if attention_mask is not None else None
        test_position_ids = position_ids[idx:idx+1].to(device) if position_ids is not None else None
        
        single_output = layer(
            hidden_states=test_input,
            attention_mask=test_mask,
            position_ids=test_position_ids
        )[0]
        single_outputs.append(single_output)
    batch_output = torch.cat(single_outputs, dim=0)
    del single_outputs
    return batch_output

def process_batch_data(layer, inps, attention_mask, start_idx, end_idx, device,position_ids):
    indices = list(range(start_idx, end_idx))
    return get_batch_outputs(layer, inps, attention_mask, indices, device,position_ids)

def prepare_calibration_input_llama(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    inps_list = []
    attention_mask_list = []
    position_ids_list = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
            
        def forward(self, inp, **kwargs):
            inps_list.append(inp.detach().cpu())
            attention_mask_list.append(kwargs['attention_mask'].detach().cpu())
            position_ids_list.append(kwargs['position_ids'].detach().cpu())
            raise StopForwardException

    original_first_layer = layers[0]
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None
                if attention_mask is not None:
                    model(input_ids, attention_mask=attention_mask)
                else:
                    model(input_ids)
            else:
                model(batch.to(device))
        except StopForwardException:
            pass 

    layers[0] = original_first_layer

    inps = torch.cat(inps_list)
    if attention_mask_list:
        attention_mask = torch.cat(attention_mask_list)
        if attention_mask.dim() == 2:
            batch_size, seq_len = attention_mask.shape
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        elif attention_mask.dim() == 4 and attention_mask.size(1) == 1:
            pass
        else:
            batch_size = attention_mask.size(0)
            seq_len = attention_mask.size(-1)
            attention_mask = attention_mask.view(batch_size, 1, seq_len, seq_len)
    else:
        attention_mask = None

    if position_ids_list:
        position_ids = torch.cat(position_ids_list)
    else:
        position_ids = None

    model.config.use_cache = use_cache
    
    return inps, attention_mask, position_ids

def reconstruction_llama(original_model,batch_size: int = 1, 
                  epochs: int = 10, GDLOSS: bool = True,data_loader=None,subset_loader=None,device=None,
                  args=None):
    args = args
    logger = setup_logger()
    logger.info("\n" + "="*50)
    logger.info(f"Starting model reconstruction process")
    logger.info(f"Training epochs: {epochs}, Batch size: {batch_size}")
    logger.info("="*50 + "\n")

    layers = original_model.model.layers
    
    with torch.no_grad():
        inps, attention_mask, position_ids = prepare_calibration_input_llama(
            original_model, data_loader, device=device
        )
        del data_loader
        torch.cuda.empty_cache()  
        gc.collect()
    actual_batch_size = inps.size(0)
    total_batch = int(actual_batch_size / batch_size)
    

    pruned_inps = inps.detach().clone()
    dense_inps = inps.detach().clone()
    del inps

    for cur_idx in range(len(layers)):
        current_layer = original_model.model.layers[cur_idx]
        layer_device = next(current_layer.parameters()).device
        set_LoRA(layer=current_layer,device=layer_device,block_idx=cur_idx)
        pruner = Pruner(model=original_model,layer=current_layer, block_idx=cur_idx,model_type='llama',prune_method=args.prune_method, use_cached_wanda=True, data_loader=subset_loader, device=device)

        with torch.no_grad():
            wanda_pruner = Pruner(model=original_model,layer=current_layer, block_idx=cur_idx,model_type='llama',prune_method=args.prune_method, use_cached_wanda=True, data_loader=subset_loader, device=layer_device)
            wanda_pruner.load_wanda_cache_per_layer(block_idx=cur_idx,importance_dir=args.importance_dir,device=layer_device)
            del wanda_pruner

        for name,params in current_layer.named_parameters():
            params.requires_grad = False
            if 'lora' in name:
                params.requires_grad = True
        optimizer = torch.optim.Adam([p for p in current_layer.parameters() if p.requires_grad],lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs*total_batch,eta_min=0.)
        logger.info(f"Reconstructing Block {cur_idx+1}/{len(layers)}")

        loss = nn.MSELoss()
        group_generators = {
            'group_l1': UniformRandomGenerator(args.group_l1, total_samples=epochs*total_batch),
            'group_l2': UniformRandomGenerator(args.group_l2, total_samples=epochs*total_batch),
            'group_l3': UniformRandomGenerator(args.group_l3, total_samples=epochs*total_batch),
            'group_m1': UniformRandomGenerator(args.group_m1, total_samples=epochs*total_batch),
            'group_m2': UniformRandomGenerator(args.group_m2, total_samples=epochs*total_batch),
            'group_h1': UniformRandomGenerator(args.group_h1, total_samples=epochs*total_batch),
            'group_h2': UniformRandomGenerator(args.group_h2, total_samples=epochs*total_batch)
        }
        for e in range(epochs):
            l1_loss_records = []
            l2_loss_records = []
            l3_loss_records = [] 
            m1_loss_records = [] 
            m2_loss_records = [] 
            h1_loss_records = [] 
            h2_loss_records = [] 
            
            for id_batch in range(total_batch):
                start_idx = id_batch * batch_size
                end_idx = min((id_batch + 1) * batch_size, actual_batch_size)
                batch_attention_mask = attention_mask[start_idx:end_idx].to(layer_device) 
                batch_position_ids = position_ids[start_idx:end_idx].to(layer_device)
                cur_inp = pruned_inps[start_idx:end_idx].to(layer_device)
                cur_dense_inps = dense_inps[start_idx:end_idx].to(layer_device)
                
                set_lora_mask_enabled(current_layer, False)
                with torch.no_grad():
                    dense_out = process_batch_data(
                        current_layer, cur_dense_inps, 
                        batch_attention_mask, 0, cur_dense_inps.size(0), layer_device,batch_position_ids
                    )
                set_lora_mask_enabled(current_layer, True)
                optimizer.zero_grad(set_to_none=True)
                act_cached = None
                
                for prunerate_group in range(7):
                    if prunerate_group == 0:
                        group_name = 'group_l1'
                        set_lora_ranks_requires_grad(current_layer,[0], requires_grad=True)
                        set_lora_active_group_ranks(current_layer, [0])
                        target_sparsity = group_generators[group_name].next()
                        prune_rates = generate_llama_uniform_rates(current_layer, target_sparsity)  
                        pruner.apply_pruning(prune_rates,block_idx=cur_idx) 

                        pruned_output = process_batch_data(
                            current_layer, cur_inp, 
                            batch_attention_mask, 0, cur_inp.size(0), layer_device, batch_position_ids
                        )
                        if GDLOSS:
                            act_cached = pruned_output.detach().clone()

                        rec_loss = lp_loss(pruned_output, dense_out)
                        l1_loss_records.append(rec_loss.item())
                        logger.info(f"[Block {cur_idx+1}/{len(layers)}][Epoch {e+1}/{epochs}]"
                                f"[Batch {id_batch+1}/{total_batch}][Sparsity {target_sparsity}] "
                                f"Loss: {rec_loss.item():.4f}")

                        MaskManager.clear_block_masks(cur_idx)
                        del pruned_output
                    elif prunerate_group == 1:    
                        group_name = 'group_l2'   
                        set_lora_ranks_requires_grad(current_layer, [0,1], requires_grad=True) 
                        set_lora_active_group_ranks(current_layer, [0,1])
                        target_sparsity = group_generators[group_name].next()
                        prune_rates = generate_llama_uniform_rates(current_layer, target_sparsity)
                        pruner.apply_pruning(prune_rates,block_idx=cur_idx)
                        pruned_output = process_batch_data(
                            current_layer, cur_inp, 
                            batch_attention_mask, 0, cur_inp.size(0), layer_device, batch_position_ids
                        )
                        if GDLOSS and 'act_cached' in locals():
                            rec_loss = 0.5 * lp_loss(pruned_output, act_cached.detach()) + \
                                    0.5 * lp_loss(pruned_output, dense_out)
                        else:
                            rec_loss = lp_loss(pruned_output, dense_out)

                        l2_loss_records.append(rec_loss.item())
                        logger.info(f"[Block {cur_idx+1}/{len(layers)}][Epoch {e+1}/{epochs}]"
                                f"[Batch {id_batch+1}/{total_batch}][Sparsity {target_sparsity}] "
                                f"Loss: {rec_loss.item():.4f}")
                        MaskManager.clear_block_masks(cur_idx)
                        del pruned_output   
                    elif prunerate_group == 2:
                        group_name = 'group_l3'
                        set_lora_ranks_requires_grad(current_layer, [0,1,2], requires_grad=True)
                        set_lora_active_group_ranks(current_layer, [0,1,2])
                        target_sparsity = group_generators[group_name].next()
                        prune_rates = generate_llama_uniform_rates(current_layer, target_sparsity)
                        pruner.apply_pruning(prune_rates,block_idx=cur_idx)

                        pruned_output = process_batch_data(
                            current_layer, cur_inp, 
                            batch_attention_mask, 0, cur_inp.size(0), layer_device, batch_position_ids
                        )
                        if GDLOSS and 'act_cached' in locals():
                            rec_loss = 0.5 * lp_loss(pruned_output, act_cached.detach()) + \
                                    0.5 * lp_loss(pruned_output, dense_out)
                        else:
                            rec_loss = lp_loss(pruned_output, dense_out)
                        l3_loss_records.append(rec_loss.item())
                        logger.info(f"[Block {cur_idx+1}/{len(layers)}][Epoch {e+1}/{epochs}]"
                                f"[Batch {id_batch+1}/{total_batch}][Sparsity {target_sparsity}] "
                                f"Loss: {rec_loss.item():.4f}")
                        MaskManager.clear_block_masks(cur_idx)
                        del pruned_output  
                    elif prunerate_group == 3:
                        group_name = 'group_m1'
                        set_lora_ranks_requires_grad(current_layer, [0,1,2,3], requires_grad=True)
                        set_lora_active_group_ranks(current_layer, [0,1,2,3])
                        target_sparsity = group_generators[group_name].next()
                        prune_rates = generate_llama_uniform_rates(current_layer, target_sparsity)
                        pruner.apply_pruning(prune_rates,block_idx=cur_idx)

                        pruned_output = process_batch_data(
                            current_layer, cur_inp, 
                            batch_attention_mask, 0, cur_inp.size(0), layer_device, batch_position_ids
                        )
                        if GDLOSS and 'act_cached' in locals():
                            rec_loss = 0.5 * lp_loss(pruned_output, act_cached.detach()) + \
                                    0.5 * lp_loss(pruned_output, dense_out)
                        else:
                            rec_loss = lp_loss(pruned_output, dense_out)
                        m1_loss_records.append(rec_loss.item())
                        logger.info(f"[Block {cur_idx+1}/{len(layers)}][Epoch {e+1}/{epochs}]"
                                f"[Batch {id_batch+1}/{total_batch}][Sparsity {target_sparsity}] "
                                f"Loss: {rec_loss.item():.4f}")
                    
                        MaskManager.clear_block_masks(cur_idx)
                        del pruned_output  
                    elif prunerate_group == 4:
                        group_name = 'group_m2'
                        set_lora_ranks_requires_grad(current_layer, [0,1,2,3,4], requires_grad=True)
                        set_lora_active_group_ranks(current_layer, [0,1,2,3,4])
                        target_sparsity = group_generators[group_name].next()
                        prune_rates = generate_llama_uniform_rates(current_layer, target_sparsity)
                        pruner.apply_pruning(prune_rates,block_idx=cur_idx)

                        pruned_output = process_batch_data(
                            current_layer, cur_inp, 
                            batch_attention_mask, 0, cur_inp.size(0), layer_device, batch_position_ids
                        )
                        if GDLOSS and 'act_cached' in locals():
                            rec_loss = 0.5 * lp_loss(pruned_output, act_cached.detach()) + \
                                    0.5 * lp_loss(pruned_output, dense_out)
                        else:
                            rec_loss = lp_loss(pruned_output, dense_out)
                        m2_loss_records.append(rec_loss.item())
                        logger.info(f"[Block {cur_idx+1}/{len(layers)}][Epoch {e+1}/{epochs}]"
                                f"[Batch {id_batch+1}/{total_batch}][Sparsity {target_sparsity}] "
                                f"Loss: {rec_loss.item():.4f}")
                    
                        MaskManager.clear_block_masks(cur_idx)
                        del pruned_output  
                    elif prunerate_group == 5:
                        group_name = 'group_h1'
                        set_lora_ranks_requires_grad(current_layer, [0,1,2,3,4,5], requires_grad=True)
                        set_lora_active_group_ranks(current_layer, [0,1,2,3,4,5])
                        target_sparsity = group_generators[group_name].next()
                        prune_rates = generate_llama_uniform_rates(current_layer, target_sparsity)
                        pruner.apply_pruning(prune_rates,block_idx=cur_idx)
                        pruned_output = process_batch_data(
                            current_layer, cur_inp, 
                            batch_attention_mask, 0, cur_inp.size(0), layer_device, batch_position_ids
                        )
                        if GDLOSS and 'act_cached' in locals():
                            rec_loss = 0.5 * lp_loss(pruned_output, act_cached.detach()) + \
                                    0.5 * lp_loss(pruned_output, dense_out)
                        else:
                            rec_loss = lp_loss(pruned_output, dense_out)
                        h1_loss_records.append(rec_loss.item())
                        logger.info(f"[Block {cur_idx+1}/{len(layers)}][Epoch {e+1}/{epochs}]"
                                f"[Batch {id_batch+1}/{total_batch}][Sparsity {target_sparsity}] "
                                f"Loss: {rec_loss.item():.4f}")
                        MaskManager.clear_block_masks(cur_idx)
                        del pruned_output  
                    elif prunerate_group == 6:
                        group_name = 'group_h2'
                        set_lora_ranks_requires_grad(current_layer, [0,1,2,3,4,5,6], requires_grad=True)
                        set_lora_active_group_ranks(current_layer, [0,1,2,3,4,5,6])
                        target_sparsity = group_generators[group_name].next()
                        prune_rates = generate_llama_uniform_rates(current_layer, target_sparsity)
                        pruner.apply_pruning(prune_rates,block_idx=cur_idx)
                        pruned_output = process_batch_data(
                            current_layer, cur_inp, 
                            batch_attention_mask, 0, cur_inp.size(0), layer_device, batch_position_ids
                        )
                        if GDLOSS and 'act_cached' in locals():
                            rec_loss = 0.5 * lp_loss(pruned_output, act_cached.detach()) + \
                                    0.5 * lp_loss(pruned_output, dense_out)
                        else:
                            rec_loss = lp_loss(pruned_output, dense_out)
                        h2_loss_records.append(rec_loss.item())
                        logger.info(f"[Block {cur_idx+1}/{len(layers)}][Epoch {e+1}/{epochs}]"
                                f"[Batch {id_batch+1}/{total_batch}][Sparsity {target_sparsity}] "
                                f"Loss: {rec_loss.item():.4f}")
                        MaskManager.clear_block_masks(cur_idx)
                        del pruned_output  
                    if cur_idx < 6:
                        rec_loss.backward(retain_graph=True)
                    else:
                        rec_loss.backward(retain_graph=False)
                optimizer.step()
                scheduler.step()

                del dense_out, cur_inp, cur_dense_inps, batch_attention_mask, batch_position_ids
                if act_cached is not None:
                    del act_cached
                torch.cuda.empty_cache()

            l1_loss = sum(l1_loss_records) / len(l1_loss_records)
            l2_loss = sum(l2_loss_records) / len(l2_loss_records)
            l3_loss = sum(l3_loss_records) / len(l3_loss_records)
            m1_loss = sum(m1_loss_records) / len(m1_loss_records)
            m2_loss = sum(m2_loss_records) / len(m2_loss_records)
            h1_loss = sum(h1_loss_records) / len(h1_loss_records)
            h2_loss = sum(h2_loss_records) / len(h2_loss_records)
            logger.info("[Stage-{}] [Epochs: {}/{}] [low1 loss:{:.4f}] | [low2 loss:{:.4f}] | [low3 loss:{:.4f}] | [middle1 loss:{:.4f}]| [middle2 loss:{:.4f}]  | [high1 loss:{:.4f}]| [high2 loss:{:.4f}]".format(cur_idx+1,e+1,epochs, l1_loss, l2_loss, l3_loss, m1_loss, m2_loss, h1_loss, h2_loss)) 

        is_positive = evaluate_lora_effectiveness(
                original_model, current_layer, dense_inps, attention_mask, position_ids,
                batch_size, layer_device, cur_idx, subset_loader,args
            )
        if not is_positive:
            logger.info(f"Block {cur_idx+1}: Negative optimization for low pruning rate group LoRA, disabling in this block")
            current_layer.low_group_lora_disabled = True
        else:
            current_layer.low_group_lora_disabled = False
        if cur_idx < 31:
            set_lora_mask_enabled(current_layer, False)
            dense_cached_batches = []
            with torch.no_grad():
                for i in range(total_batch):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, actual_batch_size)
                    
                    batch_input = dense_inps[start_idx:end_idx].to(layer_device)
                    
                    batch_attention_mask = attention_mask[start_idx:end_idx].to(layer_device)
                    batch_position_ids = position_ids[start_idx:end_idx].to(layer_device)
                    
                    layer_output = process_batch_data(
                        current_layer, batch_input, 
                        batch_attention_mask, 0, batch_input.size(0), layer_device,batch_position_ids
                    )
                    dense_cached_batches.append(layer_output)
            dense_inps = torch.cat([ x for x in dense_cached_batches])
            del dense_cached_batches
            gc.collect()
            torch.cuda.empty_cache()
            set_lora_mask_enabled(current_layer, True)
            pruned_inps = feature_mixer(
                original_model, current_layer, pruned_inps, batch_size, 
                args.fusion_level, args, block_idx=cur_idx, attention_mask=attention_mask,position_ids=position_ids,data_loader=subset_loader,device=device
            )

            if layer_device != torch.device('cuda:0'):
                with torch.cuda.device('cuda:0'):
                    torch.cuda.empty_cache()
        pruner.cleanup()
        Pruner.clear_wanda_cache_for_block(cur_idx)
        MaskManager.clear_block_masks(cur_idx)
        del pruner
        gc.collect()
        torch.cuda.empty_cache()
def feature_mixer(model, layer, inps, batch_size, level, args, block_idx,attention_mask=None,position_ids=None,device=None, data_loader=None,sparsity_alloc=None):
    layer_device = next(layer.parameters()).device
    inps = inps.to(layer_device)

    if attention_mask is not None:
        attention_mask = attention_mask.to(layer_device)
    if position_ids is not None:
        position_ids = position_ids.to(layer_device)

    logger = logging.getLogger('LoRA_Pruning') if hasattr(logging, 'getLogger') else None
    def log_info(msg):
        if logger: logger.info(msg)
        else: print(msg)
    group_l1_min, group_l1_max = min(args.group_l1), max(args.group_l1)
    group_l2_min, group_l2_max = min(args.group_l2), max(args.group_l2)
    group_l3_min, group_l3_max = min(args.group_l3), max(args.group_l3)
    group_m1_min, group_m1_max = min(args.group_m1), max(args.group_m1)
    group_m2_min, group_m2_max = min(args.group_m2), max(args.group_m2)
    group_h1_min, group_h1_max = min(args.group_h1), max(args.group_h1)
    group_h2_min, group_h2_max = min(args.group_h2), max(args.group_h2)

    low_group_lora_disabled = getattr(layer, 'low_group_lora_disabled', False)
    if level == "datasets":
        log_info("Executing dataset-level feature fusion....")
        fused_batches = []
        total_batch = int(inps.size(0) / batch_size)
        group_generators = {
            'group_l1': UniformRandomGenerator(args.group_l1, total_samples=total_batch),
            'group_l2': UniformRandomGenerator(args.group_l2, total_samples=total_batch),
            'group_l3': UniformRandomGenerator(args.group_l3, total_samples=total_batch),
            'group_m1': UniformRandomGenerator(args.group_m1, total_samples=total_batch),
            'group_m2': UniformRandomGenerator(args.group_m2, total_samples=total_batch),
            'group_h1': UniformRandomGenerator(args.group_h1, total_samples=total_batch),
            'group_h2': UniformRandomGenerator(args.group_h2, total_samples=total_batch)
        }
        prune_rates = [
            group_generators['group_l1'].next(),
            group_generators['group_l2'].next(),
            group_generators['group_l3'].next(),
            group_generators['group_m1'].next(),
            group_generators['group_m2'].next(),
            group_generators['group_h1'].next(),
            group_generators['group_h2'].next()
        ]
        pruner = Pruner(model=model, layer=layer, block_idx=block_idx, model_type='llama',prune_method=args.prune_method, data_loader=data_loader,use_cached_wanda=True,device=layer_device)
        with torch.no_grad():
            for t in range(total_batch):
                cached_inps_prune = list()
                
                start_idx = t * batch_size
                end_idx = min((t + 1) * batch_size, inps.size(0))
                batch_inp = inps[start_idx:end_idx]
                batch_attention_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
                batch_position_ids = position_ids[start_idx:end_idx] if position_ids is not None else None

                for i, prune_rate in enumerate(prune_rates):
                    if i == 0 and low_group_lora_disabled:
                        set_lora_enabled(layer, False)
                        log_info(f"Batch {t+1}: LoRA disabled for low pruning rate group (Pruning rate: {prune_rate})")
                    else:
                        set_lora_enabled(layer, True)
                        
                        if group_l1_min < prune_rate <= group_l1_max:
                            set_lora_active_group_ranks(layer, [0])
                        elif group_l2_min < prune_rate <= group_l2_max:
                            set_lora_active_group_ranks(layer, [0,1])
                        elif group_l3_min < prune_rate <= group_l3_max:    
                            set_lora_active_group_ranks(layer, [0,1,2])
                        elif group_m1_min < prune_rate <= group_m1_max:
                            set_lora_active_group_ranks(layer, [0,1,2,3])
                        elif group_m2_min < prune_rate <= group_m2_max:
                            set_lora_active_group_ranks(layer, [0,1,2,3,4])
                        elif group_h1_min < prune_rate <= group_h1_max:
                            set_lora_active_group_ranks(layer, [0,1,2,3,4,5])
                        elif group_h2_min < prune_rate <= group_h2_max:
                            set_lora_active_group_ranks(layer, [0,1,2,3,4,5,6])
                    
                    er_rates = generate_llama_uniform_rates(layer, prune_rate)
                    pruner.apply_pruning(er_rates, block_idx=block_idx)
                    output = process_batch_data(
                        layer, batch_inp, batch_attention_mask,
                        0, batch_inp.size(0), layer_device,batch_position_ids
                    )
                    cached_inps_prune.append(output)
                    del output
                    torch.cuda.empty_cache()
                weights = [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]  
                fused_output = sum(w * out for w, out in zip(weights, cached_inps_prune))
                fused_batches.append(fused_output)
                del fused_output
                torch.cuda.empty_cache()
            pruned_inps = torch.cat([x for x in fused_batches])   
            MaskManager.clear_block_masks(block_idx) 
            pruner.cleanup()
            del pruner,fused_batches,cached_inps_prune
            gc.collect()
            torch.cuda.empty_cache()
            return pruned_inps 

def evaluate_lora_effectiveness(model, layer, dense_inps, attention_mask, position_ids, 
                                batch_size, layer_device, block_idx, subset_loader, args):
 
    logger = setup_logger()
    logger.info(f"Evaluating LoRA effectiveness of low pruning rate group for Block {block_idx+1}")

    low_sparsity = 0.3
    total_samples = dense_inps.size(0)
    total_batch = int(total_samples / batch_size)
    if total_samples % batch_size != 0:
        total_batch += 1

    set_lora_enabled(layer, False)
    
    total_loss_no_lora = 0.0
    actual_samples_count = 0

    with torch.no_grad():
        prune_rates = generate_llama_uniform_rates(layer, low_sparsity)
        pruner = Pruner(model=model, layer=layer, block_idx=block_idx,
                       model_type='llama', prune_method=args.prune_method, 
                       use_cached_wanda=True, data_loader=subset_loader, device=layer_device)
        pruner.apply_pruning(prune_rates, block_idx=block_idx)
        
        for t in range(total_batch):
            start_idx = t * batch_size
            end_idx = min((t + 1) * batch_size, total_samples)
            if start_idx >= end_idx: break
            
            current_batch_size = end_idx - start_idx

            batch_inp = dense_inps[start_idx:end_idx].to(layer_device)
            batch_mask = attention_mask[start_idx:end_idx].to(layer_device) if attention_mask is not None else None
            batch_pos = position_ids[start_idx:end_idx].to(layer_device) if position_ids is not None else None

            set_lora_mask_enabled(layer, False) 
            dense_out = process_batch_data(
                layer, batch_inp, batch_mask, 0, current_batch_size, layer_device, batch_pos
            )

            set_lora_mask_enabled(layer, True)
            output_no_lora = process_batch_data(
                layer, batch_inp, batch_mask, 0, current_batch_size, layer_device, batch_pos
            )

            batch_loss = lp_loss(output_no_lora, dense_out).item()
            total_loss_no_lora += batch_loss * current_batch_size
            actual_samples_count += current_batch_size

            del dense_out, output_no_lora, batch_inp, batch_mask, batch_pos
            torch.cuda.empty_cache() 

        pruner.restore_original_weights(block_idx=block_idx)
        MaskManager.clear_block_masks(block_idx)
        pruner.cleanup()
        del pruner

    avg_loss_no_lora = total_loss_no_lora / actual_samples_count if actual_samples_count > 0 else float('inf')

    set_lora_enabled(layer, True)
    set_lora_active_group_ranks(layer, [0]) 
    
    total_loss_with_lora = 0.0
    actual_samples_count = 0 

    with torch.no_grad():
        prune_rates = generate_llama_uniform_rates(layer, low_sparsity)
        pruner = Pruner(model=model, layer=layer, block_idx=block_idx,
                       model_type='llama', prune_method=args.prune_method, 
                       use_cached_wanda=True, data_loader=subset_loader, device=layer_device)
        pruner.apply_pruning(prune_rates, block_idx=block_idx)
        
        for t in range(total_batch):
            start_idx = t * batch_size
            end_idx = min((t + 1) * batch_size, total_samples)
            if start_idx >= end_idx: break
            
            current_batch_size = end_idx - start_idx
            
            batch_inp = dense_inps[start_idx:end_idx].to(layer_device)
            batch_mask = attention_mask[start_idx:end_idx].to(layer_device) if attention_mask is not None else None
            batch_pos = position_ids[start_idx:end_idx].to(layer_device) if position_ids is not None else None

            set_lora_mask_enabled(layer, False)
            set_lora_enabled(layer, False)
            dense_out = process_batch_data(
                layer, batch_inp, batch_mask, 0, current_batch_size, layer_device, batch_pos
            )

            set_lora_mask_enabled(layer, True)
            set_lora_enabled(layer, True) 
            output_with_lora = process_batch_data(
                layer, batch_inp, batch_mask, 0, current_batch_size, layer_device, batch_pos
            )

            batch_loss = lp_loss(output_with_lora, dense_out).item()
            total_loss_with_lora += batch_loss * current_batch_size
            actual_samples_count += current_batch_size

            del dense_out, output_with_lora, batch_inp, batch_mask, batch_pos

        pruner.restore_original_weights(block_idx=block_idx)
        MaskManager.clear_block_masks(block_idx)
        pruner.cleanup()
        del pruner

    avg_loss_with_lora = total_loss_with_lora / actual_samples_count if actual_samples_count > 0 else float('inf')

    improvement = avg_loss_no_lora - avg_loss_with_lora
    is_positive = improvement > 0
    
    logger.info(f"Block {block_idx+1} LoRA evaluation results:")
    logger.info(f"Avg loss without LoRA: {avg_loss_no_lora:.6f}")
    logger.info(f"Avg loss with LoRA: {avg_loss_with_lora:.6f}")
    logger.info(f"Improvement: {improvement:.6f}")
    logger.info(f"Is LoRA positive: {is_positive}")

    set_lora_enabled(layer, True) 

    return is_positive