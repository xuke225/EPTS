import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import os
import gc
from utils.maskmanager import MaskManager
class WrappedLayer:
    def __init__(self, layer):
        self.layer = layer
        self.scaler_row = None  
    
    def add_batch(self, inp):
        if len(inp) > 0:
            inp = inp[0].detach() if isinstance(inp, tuple) else inp.detach()
            if self.scaler_row is None:
                self.scaler_row = (inp ** 2).mean(dim=0)
            else:
                self.scaler_row = self.scaler_row + (inp ** 2).mean(dim=0)
    
    def finalize(self, total_samples):
        if self.scaler_row is not None:
            self.scaler_row /= total_samples

class Pruner:
    class_hessian_cache = {}
    class_wanda_importance_cache = {}

    @staticmethod
    def save_wanda_cache_per_layer(block_idx, file_path_prefix="./cache/wanda_importance"):
        try:
            os.makedirs(os.path.dirname(file_path_prefix), exist_ok=True)
            block_importance = {}
            saved_count = 0

            for (cache_block_idx, layer_name), importance in Pruner.class_wanda_importance_cache.items():
                if cache_block_idx == block_idx:
                    block_importance[layer_name] = importance
                    saved_count += 1
            
            if block_importance:
                block_file_path = f"{file_path_prefix}_block_{block_idx}.pth"
                torch.save(block_importance, block_file_path)
                print(f"Saved complete importance for block {block_idx} to: {block_file_path}, containing {saved_count} layers")
            
            print(f"Importance for block {block_idx} has been saved as a single file")
            return True
        except Exception as e:
            print(f"Failed to save Wanda importance cache per block: {e}")
            return False

    @staticmethod
    def load_wanda_cache_per_layer(block_idx, importance_dir=None, device=None):
        import os
        try:
            load_dir = importance_dir if importance_dir is not None else "./cache"

            file_name = f"llama_7b_wanda_importance_block_{block_idx}.pth"
            block_file_path = os.path.join(load_dir, file_name)

            block_importance = torch.load(block_file_path, map_location=device)
            
            loaded_count = 0
            for layer_name, importance in block_importance.items():
                Pruner.class_wanda_importance_cache[(block_idx, layer_name)] = importance
                loaded_count += 1
            
            print(f"Loaded importance for {loaded_count} layers of block {block_idx} from {block_file_path}")
            return loaded_count > 0
        except Exception as e:
            print(f"Failed to load Wanda importance cache per block (block {block_idx}): {e}")
            return False

    @staticmethod
    def save_hessian_cache(file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            torch.save(Pruner.class_hessian_cache, file_path)
            print(f"Hessian cache saved to: {file_path}")
        except Exception as e:
            print(f"Failed to save Hessian cache: {e}")
    @staticmethod
    def load_hessian_cache(file_path, device='cuda'):
        try:
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}, skipping load")
                return
            cache_data = torch.load(file_path, map_location=device)
            Pruner.class_hessian_cache.clear()
            Pruner.class_hessian_cache.update(cache_data)
            print(f"Hessian cache loaded from {file_path} to device {device}")
        except Exception as e:
            print(f"Failed to load Hessian cache: {e}")
    
    @staticmethod
    def clear_wanda_cache_for_block(block_idx):
        keys_to_remove = []
        for key in Pruner.class_wanda_importance_cache.keys():
            if key[0] == block_idx:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del Pruner.class_wanda_importance_cache[key]
        print(f"Cleared Wanda importance cache for block {block_idx}, total {len(keys_to_remove)} items")

    def __init__(self, model, layer,block_idx,prune_method='l1', model_type=None,use_cached_hessian=None,data_loader=None,use_cached_wanda=None,nsamples=None,device='cuda',prune_direction='row'):
        self.model = model
        self.layer = layer
        self.prune_method = prune_method
        self.device = device
        self.data_loader = data_loader
        self.masks = {}
        self.hessian_cache = {}
        self.original_weights = {}
        self.prunable_layers = []
        self.block_idx = block_idx
        self.model_type = model_type
        self.use_cached_hessian = use_cached_hessian
        self.use_cached_wanda = use_cached_wanda
        self.nsamples = nsamples
        self.prune_direction = prune_direction
        self.wanda_importance = {}  
        self.wrapped_layers = {}  
        self._register_current_block_only()

    def _register_current_block_only(self):
        if self.model_type == 'llama':
            self._register_llama_current_block()
        elif self.model_type == 'opt':
            self._register_opt_current_block()
        elif self.model_type == 'vit':
            self._register_vit_current_block()
        elif self.model_type in ['resnet_low', 'resnet_high']:
            self._register_resnet_current_block()
        elif self.model_type == 'mobilenetv2':
            self._register_mobilenet_current_block()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def _register_llama_current_block(self):
        print(f"Registering prunable layers for LLaMA block {self.block_idx}")
        
        layers = {
            'q_proj': self.layer.self_attn.q_proj,
            'k_proj': self.layer.self_attn.k_proj,
            'v_proj': self.layer.self_attn.v_proj,
            'o_proj': self.layer.self_attn.o_proj,
            'gate_proj': self.layer.mlp.gate_proj,
            'up_proj': self.layer.mlp.up_proj,
            'down_proj': self.layer.mlp.down_proj
        }
        
        self.prunable_layers.append((self.block_idx, layers))

        self.original_weights[self.block_idx] = {}
        for layer_name, layer in layers.items():
            weight = layer.weight.detach().clone()
            if weight.device.type == 'cuda':
                weight = weight.cpu()
            
            self.original_weights[self.block_idx][f'{layer_name}_weight'] = weight

            if layer.bias is not None:
                bias = layer.bias.detach().clone()
                if bias.device.type == 'cuda':
                    bias = bias.cpu()
                self.original_weights[self.block_idx][f'{layer_name}_bias'] = bias
        
        print(f"Block {self.block_idx} registration complete, total {len(layers)} layers")
    def _register_opt_current_block(self):
        print(f"Registering prunable layers for opt block {self.block_idx}")
        
        layers = {
            'q_proj': self.layer.self_attn.q_proj,
            'k_proj': self.layer.self_attn.k_proj,
            'v_proj': self.layer.self_attn.v_proj,
            'out_proj': self.layer.self_attn.out_proj,
            'fc1': self.layer.fc1,
            'fc2': self.layer.fc2
        }
        
        self.prunable_layers.append((self.block_idx, layers))

        self.original_weights[self.block_idx] = {}
        for layer_name, layer in layers.items():
            weight = layer.weight.detach().clone()
            if weight.device.type == 'cuda':
                weight = weight.cpu()
            
            self.original_weights[self.block_idx][f'{layer_name}_weight'] = weight

            if layer.bias is not None:
                bias = layer.bias.detach().clone()
                if bias.device.type == 'cuda':
                    bias = bias.cpu()
                self.original_weights[self.block_idx][f'{layer_name}_bias'] = bias
        
        print(f"Block {self.block_idx} registration complete, total {len(layers)} layers")
    def _register_vit_current_block(self):
        print(f"Registering prunable layers for ViT block {self.block_idx}")
        
        layers = {
            'qkv': self.layer.attn.qkv,
            'proj': self.layer.attn.proj,
            'fc1': self.layer.mlp.fc1,
            'fc2': self.layer.mlp.fc2
        }
        
        self.prunable_layers.append((self.block_idx, layers))

        self.original_weights[self.block_idx] = {}
        for layer_name, layer in layers.items():
            weight = layer.weight.detach().clone()
            if weight.device.type == 'cuda':
                weight = weight.cpu()
            self.original_weights[self.block_idx][f'{layer_name}_weight'] = weight
            
            if layer.bias is not None:
                bias = layer.bias.detach().clone()
                if bias.device.type == 'cuda':
                    bias = bias.cpu()
                self.original_weights[self.block_idx][f'{layer_name}_bias'] = bias

    def _register_resnet_current_block(self):
        print(f"Registering prunable layers for ResNet block {self.block_idx}")

        if self._is_basic_block(self.layer):
            layers = {
                'conv1': self.layer.conv1,
                'conv2': self.layer.conv2
            }
            if self.layer.downsample is not None and len(self.layer.downsample) > 0:
                if isinstance(self.layer.downsample[0], nn.Conv2d):
                    layers['shortcut_conv'] = self.layer.downsample[0]
        elif self._is_bottleneck_block(self.layer):
            layers = {
                'conv1': self.layer.conv1,
                'conv2': self.layer.conv2,
                'conv3': self.layer.conv3
            }
            if self.layer.downsample is not None and len(self.layer.downsample) > 0:
                if isinstance(self.layer.downsample[0], nn.Conv2d):
                    layers['shortcut_conv'] = self.layer.downsample[0]
        else:
            raise ValueError(f"Unknown block type: {type(self.layer)}")
        
        self.prunable_layers.append((self.block_idx, layers))

        self.original_weights[self.block_idx] = {}
        for layer_name, layer in layers.items():
            weight = layer.weight.detach().clone()
            if weight.device.type == 'cuda':
                weight = weight.cpu()
            self.original_weights[self.block_idx][f'{layer_name}_weight'] = weight
            
            if layer.bias is not None:
                bias = layer.bias.detach().clone()
                if bias.device.type == 'cuda':
                    bias = bias.cpu()
                self.original_weights[self.block_idx][f'{layer_name}_bias'] = bias

    def _register_mobilenet_current_block(self):
        print(f"Registering prunable layers for MobileNet block {self.block_idx}")
        
        layers = self._extract_inverted_residual_layers(self.layer)
        if layers:
            self.prunable_layers.append((self.block_idx, layers))

            self.original_weights[self.block_idx] = {}
            for layer_name, layer_obj in layers.items():
                weight = layer_obj.weight.detach().clone()
                if weight.device.type == 'cuda':
                    weight = weight.cpu()
                self.original_weights[self.block_idx][f'{layer_name}_weight'] = weight
                
                if layer_obj.bias is not None:
                    bias = layer_obj.bias.detach().clone()
                    if bias.device.type == 'cuda':
                        bias = bias.cpu()
                    self.original_weights[self.block_idx][f'{layer_name}_bias'] = bias

    def _is_basic_block(self, block):
        return hasattr(block, 'conv1') and hasattr(block, 'conv2') and not hasattr(block, 'conv3')
    
    def _is_bottleneck_block(self, block):
        return hasattr(block, 'conv1') and hasattr(block, 'conv2') and hasattr(block, 'conv3')

    def _extract_inverted_residual_layers(self, block):
        layers = {}
        conv_count = 0
        conv_modules = []
        
        def collect_convs(module):
            if isinstance(module, nn.Conv2d):
                nonlocal conv_count
                conv_count += 1
                conv_modules.append(module)
            for child in module.children():
                collect_convs(child)
        
        collect_convs(block)
        
        if conv_count == 2:
            if conv_modules[0].groups > 1:
                layers['depthwise'] = conv_modules[0]
                layers['projection'] = conv_modules[1]
            elif conv_modules[1].groups > 1:
                layers['depthwise'] = conv_modules[1]
                layers['projection'] = conv_modules[0]
            else:
                layers['conv1'] = conv_modules[0]
                layers['conv2'] = conv_modules[1]
                
        elif conv_count >= 3:
            dw_conv = None
            for conv in conv_modules:
                if conv.groups > 1:
                    dw_conv = conv
                    break
            
            if dw_conv is None:
                layers['expansion'] = conv_modules[0]
                layers['depthwise'] = conv_modules[1]
                layers['projection'] = conv_modules[2]
            else:
                dw_index = conv_modules.index(dw_conv)
                if dw_index > 0:
                    layers['expansion'] = conv_modules[dw_index - 1]
                layers['depthwise'] = dw_conv
                if dw_index < len(conv_modules) - 1:
                    layers['projection'] = conv_modules[dw_index + 1]
        
        elif conv_count == 1:
            layers['conv'] = conv_modules[0]
        
        return layers

    def restore_original_weights(self, block_idx=None):
        with torch.no_grad():
            if block_idx is not None:
                target_idx = block_idx
                if (self.model_type in ['resnet_low', 'resnet_high'] and 
                    isinstance(block_idx, int)):
                    target_idx = self._find_resnet_index(block_idx)
                for idx, layers in self.prunable_layers:
                    if idx == target_idx:
                        self._restore_single_block(idx, layers)
                        return
                print(f"Block index {target_idx} not found")
            else:
                for idx, layers in self.prunable_layers:
                    self._restore_single_block(idx, layers)
    
    def _find_resnet_index(self, block_idx):
        if self.model_type not in ['resnet_low', 'resnet_high']:
            return block_idx
            
        count = 0
        for idx, _ in self.prunable_layers:
            if isinstance(idx, tuple):
                if count == block_idx:
                    return idx
                count += 1
        return block_idx
    
    def _restore_single_block(self, block_idx, layers):
        if block_idx not in self.original_weights:
            print(f"Saved weights not found for block {block_idx}")
            return
            
        for layer_name, layer_obj in layers.items():
            weight_key = f'{layer_name}_weight'
            bias_key = f'{layer_name}_bias'
            
            if weight_key not in self.original_weights[block_idx]:
                print(f"Saved weights not found for layer {layer_name} in block {block_idx}")
                continue
            saved_weight = self.original_weights[block_idx][weight_key]
            if saved_weight.device != layer_obj.weight.device:
                saved_weight = saved_weight.to(layer_obj.weight.device)

            layer_obj.weight.data.copy_(saved_weight)

            if bias_key in self.original_weights[block_idx] and self.original_weights[block_idx][bias_key] is not None:
                saved_bias = self.original_weights[block_idx][bias_key]
                if saved_bias.device != layer_obj.bias.device:
                    saved_bias = saved_bias.to(layer_obj.bias.device)
                
                if layer_obj.bias is not None:
                    layer_obj.bias.data.copy_(saved_bias)

    def compute_hessian_diag(self, max_batches=10):
        cache_key = f"{self.model_type}_{self.block_idx}"
        
        if self.use_cached_hessian and cache_key in Pruner.class_hessian_cache:
            self.hessian_cache = Pruner.class_hessian_cache[cache_key]
            print(f"Using cached Hessian diagonal approximation (block {self.block_idx})")
            return
        
        print(f"Computing Hessian diagonal approximation for block {self.block_idx}...")
        self.model.train()

        current_layers = None
        for idx, layers in self.prunable_layers:
            if idx == self.block_idx:
                current_layers = layers
                break
        
        if current_layers is None:
            raise ValueError(f"Layers corresponding to block index {self.block_idx} not found")

        self.hessian_cache = {self.block_idx: {}}

        if self.model_type == 'vit':
            layer_types = ['qkv', 'proj','fc1', 'fc2']
        elif self.model_type in ['resnet_low', 'resnet_high']:
            layer_types = list(current_layers.keys())
        elif self.model_type == 'mobilenetv2':
            layer_types = ['expansion', 'depthwise', 'projection']
        else:
            layer_types = list(current_layers.keys())

        for layer_name in layer_types:
            if layer_name in current_layers:
                layer = current_layers[layer_name]
                if hasattr(layer, 'weight') and layer.weight is not None:
                    self.hessian_cache[self.block_idx][layer_name] = torch.zeros_like(layer.weight.data)
        
        num_batches = 0
        for i, (inputs, labels) in enumerate(tqdm(self.data_loader, desc="Hessian")):
            if i >= max_batches:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            for layer_name in self.hessian_cache[self.block_idx]:
                layer = current_layers[layer_name]
                if layer.weight.grad is not None:
                    self.hessian_cache[self.block_idx][layer_name] += layer.weight.grad.data.pow(2)
            
            num_batches += 1

        for layer_name in self.hessian_cache[self.block_idx]:
            self.hessian_cache[self.block_idx][layer_name] /= max(1, num_batches)
            self.hessian_cache[self.block_idx][layer_name] += 1e-6

        if self.use_cached_hessian:
            Pruner.class_hessian_cache[cache_key] = self.hessian_cache
            print(f"Saved Hessian diagonal approximation for block {self.block_idx} to cache")

    def prepare_calibration_input(self, dataloader, device):
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        
        if self.model_type == 'llama':
            layers = self.model.model.layers
        elif self.model_type == 'opt':
            layers = self.model.model.decoder.layers

        dtype = next(iter(self.model.parameters())).dtype

        if hasattr(self.model.config, 'max_position_embeddings'):
            seqlen = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'seqlen'):
            seqlen = self.model.config.seqlen
        else:
            seqlen = 2048 
        
        hidden_size = self.model.config.hidden_size
        
        inps = torch.zeros((self.nsamples, seqlen, hidden_size), dtype=dtype, device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None}
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.self_attn = module.self_attn
            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs.get('attention_mask', None)
                cache['position_ids'] = kwargs.get('position_ids', None)
                raise ValueError

        original_first_layer = layers[0]
        layers[0] = Catcher(layers[0])

        for batch in dataloader:
            try:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)
                self.model(inputs)
            except ValueError:
                pass
            if cache['i'] >= self.nsamples:
                break

        layers[0] = original_first_layer
        self.model.config.use_cache = use_cache

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']

        return inps, outs, attention_mask, position_ids

    def precompute_wanda_importance_per_layer(self, max_batches=None, importance_dir=None):
        if max_batches is None:
            max_batches = self.nsamples if self.nsamples else 32

        print(f"Precomputing complete Wanda importance for block {self.block_idx} and saving as a single file......")
        self.model.eval()

        current_layers = None
        for idx, layers in self.prunable_layers:
            if idx == self.block_idx:
                current_layers = layers
                break
        
        if current_layers is None:
            raise ValueError(f"Layers corresponding to block index {self.block_idx} not found")

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = self.prepare_calibration_input(
                self.data_loader, self.device
            )
        
        print(f"Calibration input shape: {inps.shape}")

        if self.model_type == 'llama':
            model_layers = self.model.model.layers
        elif self.model_type == 'opt':
            model_layers = self.model.model.decoder.layers

        wrapped_layers = {}
        for layer_name, layer in current_layers.items():
            if hasattr(layer, 'weight') and layer.weight is not None:
                wrapped_layers[layer_name] = WrappedLayer(layer)
                print(f"Initializing wrapped layer: block {self.block_idx}, layer {layer_name}")
        
        print(f"Initialized {len(wrapped_layers)} wrapped layers in total")

        for i in range(len(model_layers)):
            layer = model_layers[i]

            if f"model.layers.{i}" in getattr(self.model, 'hf_device_map', {}):
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            
            handles = []
            
            for layer_name, wrapped_layer in wrapped_layers.items():
                layer_obj = current_layers[layer_name]

                layer_matched = False
                try:
                    if any(layer_obj is child for child in layer.children()):
                        layer_matched = True
                    elif hasattr(layer, layer_name) and getattr(layer, layer_name) is layer_obj:
                        layer_matched = True
                    elif hasattr(layer, 'self_attn') and any(layer_obj is child for child in layer.self_attn.children()):
                        layer_matched = True
                    elif hasattr(layer, 'mlp') and any(layer_obj is child for child in layer.mlp.children()):
                        layer_matched = True
                except Exception as e:
                    print(f"Layer matching check failed: {e}")
                    continue
                
                if layer_matched:
                    def add_batch_hook(l_name):
                        def hook(module, inp, out):
                            wrapped_layers[l_name].add_batch(inp)
                        return hook
                    
                    try:
                        handle = layer_obj.register_forward_hook(add_batch_hook(layer_name))
                        handles.append(handle)
                    except Exception as e:
                        print(f"Failed to register hook: block {self.block_idx} layer {layer_name}, error: {e}")

            for j in range(min(max_batches, inps.shape[0])):
                with torch.no_grad():
                    try:
                        forward_kwargs = {}
                        if attention_mask is not None:
                            forward_kwargs['attention_mask'] = attention_mask
                        if position_ids is not None:
                            forward_kwargs['position_ids'] = position_ids
                        layer_output = layer(
                            inps[j].unsqueeze(0),
                            **forward_kwargs
                        )
                        if isinstance(layer_output, tuple):
                            outs[j] = layer_output[0]
                        else:
                            outs[j] = layer_output
                    except Exception as e:
                        print(f"Forward pass failed: {e}")
                        break

            for h in handles:
                h.remove()

            inps, outs = outs, inps

        block_importance = {}

        for layer_name, wrapped_layer in wrapped_layers.items():
            wrapped_layer.finalize(min(max_batches, inps.shape[0]))
            
            if wrapped_layer.scaler_row is not None:
                layer_obj = current_layers[layer_name]
                weight_abs = torch.abs(layer_obj.weight.data)

                activation_scaler = wrapped_layer.scaler_row
                
                if len(activation_scaler.shape) > 1:
                    activation_rms = torch.sqrt(activation_scaler.mean(dim=0))
                else:
                    activation_rms = torch.sqrt(activation_scaler)

                if len(activation_rms.shape) == 1:
                    if weight_abs.shape[1] == activation_rms.shape[0]:
                        activation_rms = activation_rms.reshape((1, -1))
                    else:
                        activation_rms = activation_rms.mean().reshape((1, 1))
                
                if weight_abs.shape[1] == activation_rms.shape[1]:
                    importance = weight_abs * activation_rms
                else:
                    importance = weight_abs * activation_rms.mean()

                block_importance[layer_name] = importance
                print(f"Computed importance for block {self.block_idx} layer {layer_name}, shape: {importance.shape}")
            else:
                print(f"Warning: No activation data collected for block {self.block_idx} layer {layer_name}")

        if block_importance:
            os.makedirs(importance_dir, exist_ok=True)
            
            file_name = file_name = f"llama_7b_wanda_importance_block_{self.block_idx}.pth"
            block_file_path = os.path.join(importance_dir, file_name)
            torch.save(block_importance, block_file_path)
            print(f"Saved complete importance for block {self.block_idx} to: {block_file_path}, containing {len(block_importance)} layers")
        
        print(f"Precomputation of importance for block {self.block_idx} complete, saved as a single file")


    

    def compute_importance(self, weight, layer_name, block_idx):
        if block_idx != self.block_idx:
            print(f"Warning: Requested block {block_idx} does not match current block {self.block_idx}")
            return torch.abs(weight)
        
        if self.prune_method == 'l1':
            return torch.abs(weight)
        elif self.prune_method == 'wanda':
            cache_key = (block_idx, layer_name)
            if cache_key in Pruner.class_wanda_importance_cache:
                print("Loaded Wanda importance from class cache")
                return Pruner.class_wanda_importance_cache[cache_key]
            else:
                print(f"Importance for block {block_idx} layer {layer_name} not found in class cache, trying to load from file")
                layer_file_path = f"./cache/wanda_importance_block_{block_idx}_{layer_name}.pth"
                if os.path.exists(layer_file_path):
                    try:
                        importance = torch.load(layer_file_path, map_location=weight.device)
                        Pruner.class_wanda_importance_cache[cache_key] = importance
                        print(f"Loaded importance for block {block_idx} layer {layer_name} from file")
                        return importance
                    except Exception as e:
                        print(f"Failed to load importance from file: {e}")

                print(f"File also does not exist, trying on-the-fly computation of importance for block {block_idx} layer {layer_name}")
                try:
                    with torch.no_grad():
                        importance = torch.abs(weight)
                        Pruner.class_wanda_importance_cache[cache_key] = importance
                        torch.save(importance, layer_file_path)
                        print(f"Computed on-the-fly and saved importance for block {block_idx} layer {layer_name}")
                    return importance
                except Exception as e:
                    print(f"On-the-fly computation failed, using absolute value as fallback: {e}")
                    return torch.abs(weight)

        elif self.prune_method == 'first_order':
            grad = self._get_layer_grad(block_idx, layer_name)
            return torch.abs(weight * grad)
        elif self.prune_method == 'second_order_OBD':
            if block_idx in self.hessian_cache and layer_name in self.hessian_cache[block_idx]:
                H_diag = self.hessian_cache[block_idx][layer_name]
                return 0.5 * weight.pow(2) * H_diag
            else:
                print(f"Warning: Hessian diagonal approximation not found, using L1 as fallback")
                return torch.abs(weight)
        elif self.prune_method == 'second_order_OBS':
            if block_idx in self.hessian_cache and layer_name in self.hessian_cache[block_idx]:
                H_inv = self.hessian_cache[block_idx][layer_name]
                return (weight.view(-1) @ H_inv @ weight.view(-1)).view_as(weight)
            else:
                print(f"Warning: Hessian inverse approximation not found, using L1 as fallback")
                return torch.abs(weight)
        else:
            raise ValueError(f"Unsupported pruning method: {self.prune_method}")
        
    def _get_layer_grad(self, block_idx, layer_type):
        for layers in self.prunable_layers:
            if layers[0] == block_idx:
                return layers[1][layer_type].weight.grad.data
        raise ValueError(f"Layer {block_idx} {layer_type} not found")
    def apply_pruning(self, prune_rates, block_idx=None):
        print(f"Applying {self.prune_method} pruning...")
        
        if 'second_order' in self.prune_method:
            if 'OBD' in self.prune_method:
                self.compute_hessian_diag()
        elif self.prune_method == 'first_order':
            self._compute_gradients()
        self.masks = {}

        if block_idx is not None:
            target_idx = block_idx
            if (self.model_type in ['resnet_low', 'resnet_high'] and 
                isinstance(block_idx, int)):
                target_idx = self._find_resnet_index(block_idx)

            for idx, layers in self.prunable_layers:
                if idx == target_idx:
                    self._prune_single_block(idx, layers, prune_rates)
                    return
            print(f"Block index {target_idx} not found for pruning")
        else:
            for idx, layers in self.prunable_layers:
                self._prune_single_block(idx, layers, prune_rates)
    
    def _prune_single_block(self, block_idx, layers, prune_rates):
        print(f"\nProcessing block {block_idx}...")
        self.masks[block_idx] = {}
        
        if isinstance(prune_rates, dict):
            if self.model_type == 'vit':
                for layer_name in ['qkv', 'fc1', 'fc2','fc']:
                    if layer_name not in prune_rates:
                        raise ValueError(f"prune_rates dictionary is missing pruning rate for layer '{layer_name}") 
            elif self.model_type == 'opt':
                for layer_name in ['q_proj','k_proj','v_proj','out_proj','fc1','fc2']:
                    if layer_name not in prune_rates:
                        raise ValueError(f"prune_rates dictionary is missing pruning rate for layer '{layer_name}")
            elif self.model_type == 'llama':
                for layer_name in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
                    if layer_name not in prune_rates:
                        raise ValueError(f"prune_rates dictionary is missing pruning rate for layer '{layer_name}")   
            elif 'resnet' in self.model_type:
                for layer_name in layers.keys():
                    if layer_name not in prune_rates:
                        raise ValueError(f"prune_rates dictionary is missing pruning rate for layer '{layer_name}")
            elif 'mobilenet' in self.model_type:
                for layer_name in layers.keys():
                    if layer_name not in prune_rates:
                        raise ValueError(f"prune_rates dictionary is missing pruning rate for layer '{layer_name}")
            
            block_config = prune_rates  
        else:
            raise ValueError("prune_rates must be a dictionary containing pruning rates for layer names")

        for layer_name, layer in layers.items():
            if not hasattr(layer, 'weight') or layer.weight is None:
                continue
                
            weight = layer.weight.data
            layer_device = weight.device

            current_rate = block_config.get(layer_name, 0.0)
            
            print(f"Processing layer {layer_name}: shape={weight.shape}, prune rate={current_rate}")

            importance = self.compute_importance(weight, layer_name, block_idx)
            mask = self._generate_mask(importance, current_rate,prune_direction=self.prune_direction)
            del importance
            if mask.device != layer_device:
                print(f"Mask device {mask.device} does not match weight device {layer_device}, synchronizing devices")
                mask = mask.to(layer_device)

            self.masks[block_idx][layer_name] = mask
            MaskManager.set_mask(block_idx, layer_name, mask)
                
    def _compute_gradients(self):
        print("Computing first-order gradients...")
        self.model.train()

        inputs, labels = next(iter(self.data_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
    
    def _generate_mask(self, importance, prune_rate, device=None, prune_direction=None):
        if prune_rate <= 0:
            return torch.ones_like(importance, device=device)

        if prune_direction == 'row':
            return self._generate_mask_row_wise(importance, prune_rate, device)
        elif prune_direction == 'column':
            return self._generate_mask_column_wise(importance, prune_rate, device)
        elif prune_direction == 'global':
            return self._generate_mask_global(importance, prune_rate)
        else:
            raise ValueError(f"Unsupported pruning direction: {prune_direction}, options are 'row', 'column', 'global'")

    def _generate_mask_row_wise(self, importance, prune_rate, device=None):
        original_shape = importance.shape
        if len(importance.shape) == 2:  
            num_rows, num_cols = importance.shape
            is_conv = False
        elif len(importance.shape) == 4:  
            out_channels, in_channels, H, W = importance.shape
            num_rows = out_channels
            num_cols = in_channels * H * W
            importance = importance.view(out_channels, -1) 
            is_conv = True
        else:
            print(f"Warning: Unsupported weight shape {importance.shape}, using global pruning")
            return self._generate_mask_global(importance, prune_rate, device)
        
        num_keep_per_row = max(1, int(num_cols * (1 - prune_rate)))
        
        print(f"Row-wise pruning: Matrix shape [{num_rows}, {num_cols}], keeping {num_keep_per_row}/{num_cols} parameters per row")

        _, topk_indices = torch.topk(importance, k=num_keep_per_row, dim=1, largest=True)

        mask = torch.zeros_like(importance, device=importance.device)
        mask.scatter_(1, topk_indices, 1.0)

        if is_conv:
            mask = mask.view(original_shape)
        
        total_params = mask.numel()
        zero_ratio = (mask == 0).float().mean().item()
        print(f"Row-wise pruning result: Total parameters {total_params}, zero parameter ratio {zero_ratio:.4f}")
        
        if device is not None and mask.device != device:
            mask = mask.to(device)
        return mask

    def _generate_mask_column_wise(self, importance, prune_rate, device=None):
        original_shape = importance.shape
        print(f"逐列剪枝 - 原始形状: {original_shape}")
    
        if len(importance.shape) == 2:  
            out_features, in_features = importance.shape
            num_rows = out_features
            num_cols = in_features
            is_conv = False
        elif len(importance.shape) == 4:  
            out_channels, in_channels, H, W = importance.shape
            num_rows = out_channels
            num_cols = in_channels * H * W  
            importance_flat = importance.view(out_channels, -1)  
            is_conv = True
        else:
            print(f"Warning: Unsupported weight shape {importance.shape}, using global pruning")
            return self._generate_mask_global(importance, prune_rate, device)

        num_keep_per_column = max(1, int(num_rows * (1 - prune_rate)))
        
        print(f"Column-wise pruning: Matrix shape [{num_rows}, {num_cols}], keeping {num_keep_per_column}/{num_rows} parameters per column")

        if is_conv:
            mask_flat = torch.zeros_like(importance_flat)

            for j in range(num_cols):
                column_importance = importance_flat[:, j] 

                _, topk_indices = torch.topk(column_importance, num_keep_per_column, largest=True)

                mask_flat[topk_indices, j] = 1.0

            mask = mask_flat.view(original_shape)
        else:
            mask = torch.zeros_like(importance)

            for j in range(num_cols):
                column_importance = importance[:, j]  

                _, topk_indices = torch.topk(column_importance, num_keep_per_column, largest=True)

                mask[topk_indices, j] = 1.0

        total_params = mask.numel()
        zero_ratio = (mask == 0).float().mean().item()
        print(f"Column-wise pruning result: Total parameters {total_params}, zero parameter ratio {zero_ratio:.4f}, output shape: {mask.shape}")
        
        if device is not None:
            mask = mask.to(device)
        return mask

    def _generate_mask_global(self, importance, prune_rate):
        if prune_rate <= 0:
            return torch.ones_like(importance, device=self.device)
        
        flat_imp = importance.view(-1)
        num_params = flat_imp.numel()
        num_keep = max(1, int(num_params * (1 - prune_rate)))
        
        print(f"Total params: {num_params}, Keeping: {num_keep}")
        
        if num_keep < num_params:
            topk_vals, topk_indices = torch.topk(flat_imp, k=num_keep, largest=True)

            mask_flat = torch.zeros_like(flat_imp)
            mask_flat[topk_indices] = 1.0
            mask = mask_flat.view(importance.shape)
            
            zero_ratio = (mask == 0).float().mean().item()
            print(f"Generated mask zero ratio: {zero_ratio:.4f}")
            
            return mask.to(self.device)
        else:
            return torch.ones_like(importance, device=self.device)
    
    def _generate_mask_global_slow(self, importance, prune_rate):
        if prune_rate <= 0:
            return torch.ones_like(importance, device=self.device)
        
        flat_imp = importance.view(-1)
        num_params = flat_imp.numel()
        num_keep = max(1, int(num_params * (1 - prune_rate)))
        k = num_params - num_keep

        print(f"Total params: {num_params}, Keeping: {num_keep}, Pruning: {k}")

        if k > 0:
            threshold = torch.kthvalue(flat_imp, k).values
            mask = (importance >= threshold).float()

            zero_ratio = (mask == 0).float().mean().item()
            print(f"Generated mask zero ratio: {zero_ratio:.4f}")
            
            return mask.to(self.device)
        else:
            return torch.ones_like(importance, device=self.device)
    
    def cleanup(self):
        self.masks.clear()
        self.hessian_cache.clear()
        self.original_weights.clear()
        self.prunable_layers.clear()
        gc.collect()
        torch.cuda.empty_cache()