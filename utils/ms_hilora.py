import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTDecoderLayer,OPTAttention
from utils.maskmanager import MaskManager
class LoRABase(nn.Module):
    def __init__(self, block_idx=None, layer_name=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_group_indexes = [] 
        self.block_idx = block_idx  
        self.layer_name = layer_name  
        self.lora_mask_enabled = True
        self.lora_enabled = True
    def get_pruning_mask(self, device=None):
        if self.block_idx is not None and self.layer_name is not None:
            return MaskManager.get_mask(self.block_idx, self.layer_name, device)
        return None
    def set_active_group_ranks(self, target_indexes):
        if target_indexes is None:
            self.active_group_indexes = []
        else:
            self.active_group_indexes = target_indexes
    def get_active_indexes(self):
        return self.active_group_indexes
    def set_lora_mask_enabled(self, enabled=True):
        self.lora_mask_enabled = enabled
    def set_lora_enabled(self, enabled=True):
        self.lora_enabled = enabled
    
class _LoRA_fc(LoRABase):
    def __init__(self, fc: nn.Module, prune_rate: float, s: int = 1,block_idx=None, layer_name=None, use_lora=True):
        super().__init__(block_idx=block_idx, layer_name=layer_name)
        self.fc = fc
        self.weight = fc.weight
        self.bias = fc.bias
        self.in_features = fc.in_features
        self.out_features = fc.out_features   
        self.prune_rate = prune_rate
        self.use_lora = use_lora
        self.merged = False
        self.s = s

        self.initial_ranks = [8,8,8,8,8,8,8]
        self.fc_lora_down_list = nn.ParameterList()
        self.fc_lora_up_list = nn.ParameterList()
        
        for r in self.initial_ranks:
            down_param = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.bfloat16))
            nn.init.zeros_(down_param)
            self.fc_lora_down_list.append(down_param)

            up_param = nn.Parameter(torch.zeros(r, self.in_features, dtype=torch.bfloat16))
            nn.init.kaiming_uniform_(up_param, a=math.sqrt(5))
            self.fc_lora_up_list.append(up_param)
        
        self.weight.requires_grad = False
    
    def merge_active_lora_to_weight(self):
        target_indexes = self.get_active_indexes()
        
        if not self.use_lora:
            print(f"No LoRA groups, no need to merge, only pruning required")
            pruning_mask = self.get_pruning_mask(device=self.weight.device)
            if pruning_mask is not None:
                mask = pruning_mask.to(self.weight.device)
                if mask.dtype != self.weight.dtype:
                    mask = mask.to(self.weight.dtype)
                self.weight.data.mul_(mask)
            self.merged = True
            return
        
        with torch.no_grad():
            delta_W = torch.zeros_like(self.weight)
            
            for idx in target_indexes:
                if idx < len(self.fc_lora_down_list):
                    lora_update = self.fc_lora_down_list[idx] @ self.fc_lora_up_list[idx]
                    if lora_update.dtype != self.weight.dtype:
                        lora_update = lora_update.to(self.weight.dtype)
                    delta_W = delta_W + lora_update

            pruning_mask = self.get_pruning_mask(device=self.weight.device)
            if pruning_mask is not None:
                mask = pruning_mask.to(self.weight.device)
                if mask.dtype != self.weight.dtype:
                    mask = mask.to(self.weight.dtype)
                self.weight.data.add_(delta_W)
                self.weight.data.mul_(mask)
            else:
                self.weight.data.add_(delta_W)
                
            print(f"Active LoRA groups {target_indexes} merged into original weights")
            self.merged = True

    def set_rank_requires_grad(self, target_indexes, requires_grad=True):   
        if not self.use_lora:
            return
        for idx in range(len(self.initial_ranks)):
            if idx in target_indexes:
                self.fc_lora_down_list[idx].requires_grad = requires_grad
                self.fc_lora_up_list[idx].requires_grad = requires_grad
            else:
                self.fc_lora_down_list[idx].requires_grad = False
                self.fc_lora_up_list[idx].requires_grad = False
    
    def forward(self, x):
        if self.merged:
            return F.linear(x, self.weight, self.bias)

        if x.device != self.weight.device:
            print(f"Device mismatch: x is on {x.device}, weight is on {self.weight.device}")
            x = x.to(self.weight.device)
        if x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        weight = self.weight

        if not self.lora_mask_enabled:
            print("Applying neither LoRA nor mask, getting dense output")
            return F.linear(x, self.weight, self.bias)

        if not self.lora_enabled:
            print("Applying no LoRA, getting pruned output")
            pruning_mask = self.get_pruning_mask(device=x.device)
            print(pruning_mask.shape)
            if pruning_mask is not None:
                mask = pruning_mask.to(weight.device)
                if mask.dtype != weight.dtype:
                    mask = mask.to(weight.dtype)
                mask_weight = weight * mask 

                if mask_weight.dtype != weight.dtype:
                    mask_weight = mask_weight.to(weight.dtype)

                return F.linear(x, mask_weight, self.bias)

        delta_W = torch.zeros_like(weight)

        for idx in self.get_active_indexes():
            lora_update = self.fc_lora_down_list[idx] @ self.fc_lora_up_list[idx]
            if lora_update.dtype != weight.dtype:
                lora_update = lora_update.to(weight.dtype)
            delta_W = delta_W + lora_update

        pruning_mask = self.get_pruning_mask(device=x.device)
        print(pruning_mask.shape)
        if pruning_mask is not None:
            mask = pruning_mask.to(weight.device)
            if mask.dtype != weight.dtype:
                mask = mask.to(weight.dtype)
            delta_W = delta_W * mask
            new_weight = weight * mask + self.s * delta_W

        if new_weight.dtype != weight.dtype:
            new_weight = new_weight.to(weight.dtype)

        return F.linear(x, new_weight, self.bias)
   
def set_LoRA(layer=None,s=1,prune_rate=0.0,device=None,block_idx=None):
    if isinstance(layer,LlamaDecoderLayer):
        for name , _ in layer.named_children():
            if isinstance(_,LlamaAttention):
                current_device = _.q_proj.weight.device if device is None else device
                original_q_proj = _.q_proj
                new_q_proj = _LoRA_fc(original_q_proj, prune_rate,s,block_idx=block_idx, layer_name=f"q_proj").to(current_device)
                _.q_proj = new_q_proj
                
                current_device = _.k_proj.weight.device if device is None else device
                original_k_proj = _.k_proj
                new_k_proj = _LoRA_fc(original_k_proj, prune_rate,s,block_idx=block_idx, layer_name=f"k_proj").to(current_device)
                _.k_proj = new_k_proj

                current_device = _.v_proj.weight.device if device is None else device
                original_v_proj = _.v_proj
                new_v_proj = _LoRA_fc(original_v_proj, prune_rate,s,block_idx=block_idx, layer_name=f"v_proj").to(current_device)
                _.v_proj = new_v_proj

                current_device = _.o_proj.weight.device if device is None else device
                original_o_proj = _.o_proj
                new_o_proj = _LoRA_fc(original_o_proj, prune_rate,s,block_idx=block_idx, layer_name=f"o_proj").to(current_device)
                _.o_proj = new_o_proj
            elif isinstance(_,LlamaMLP):
                current_device = _.gate_proj.weight.device if device is None else device
                original_gate_proj = _.gate_proj
                new_gate_proj = _LoRA_fc(original_gate_proj, prune_rate,s,block_idx=block_idx, layer_name=f"gate_proj").to(current_device)
                _.gate_proj = new_gate_proj

                current_device = _.up_proj.weight.device if device is None else device
                original_up_proj = _.up_proj
                new_up_proj = _LoRA_fc(original_up_proj, prune_rate,s,block_idx=block_idx, layer_name=f"up_proj").to(current_device)
                _.up_proj = new_up_proj

                current_device = _.down_proj.weight.device if device is None else device
                original_down_proj = _.down_proj
                new_down_proj = _LoRA_fc(original_down_proj, prune_rate,s,block_idx=block_idx, layer_name=f"down_proj").to(current_device)
                _.down_proj = new_down_proj
    if isinstance(layer, OPTDecoderLayer):
        for name, _ in layer.named_children():
            if isinstance(_, OPTAttention):
                current_device = _.q_proj.weight.device if device is None else device
                original_q_proj = _.q_proj
                new_q_proj = _LoRA_fc(original_q_proj, prune_rate, s).to(current_device)
                _.q_proj = new_q_proj

                current_device = _.k_proj.weight.device if device is None else device
                original_k_proj = _.k_proj
                new_k_proj = _LoRA_fc(original_k_proj, prune_rate, s).to(current_device)
                _.k_proj = new_k_proj

                current_device = _.v_proj.weight.device if device is None else device
                original_v_proj = _.v_proj
                new_v_proj = _LoRA_fc(original_v_proj, prune_rate, s).to(current_device)
                _.v_proj = new_v_proj

                current_device = _.out_proj.weight.device if device is None else device
                original_out_proj = _.out_proj
                new_out_proj = _LoRA_fc(original_out_proj, prune_rate, s).to(current_device)
                _.out_proj = new_out_proj
        
        if hasattr(layer, 'fc1'):
            current_device = layer.fc1.weight.device if device is None else device
            fc1 = layer.fc1
            new_fc1 = _LoRA_fc(fc1, prune_rate, s).to(current_device)
            layer.fc1 = new_fc1

        if hasattr(layer, 'fc2'):
            current_device = layer.fc2.weight.device if device is None else device
            fc2 = layer.fc2
            new_fc2 = _LoRA_fc(fc2, prune_rate, s).to(current_device)
            layer.fc2 = new_fc2

def set_lora_ranks_requires_grad(module, target_indexes, requires_grad=True):
    for m in module.modules():
        if isinstance(m, (_LoRA_fc)):
            print(f"Find {m.__class__.__name__}")
            m.set_rank_requires_grad(target_indexes, requires_grad)
def set_lora_active_group_ranks(module, target_indexes):
    for m in module.modules():
        if isinstance(m, (LoRABase,)):
            m.set_active_group_ranks(target_indexes)
def set_lora_mask_enabled(module, enabled=True):
    for m in module.modules():
        if hasattr(m, 'set_lora_mask_enabled'):
            m.set_lora_mask_enabled(enabled)

def set_lora_enabled(module, enabled=True):
    for m in module.modules():
        if hasattr(m, 'set_lora_enabled'):
            m.set_lora_enabled(enabled)

def merge_active_lora_to_weights(model):
    for module_name, module in model.named_modules():
        if hasattr(module, 'merge_active_lora_to_weight'):
            print(f"Merging module: {module_name}")
            module.merge_active_lora_to_weight()