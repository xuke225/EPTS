class MaskManager:
    _instance = None
    _masks = {}  
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MaskManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def set_mask(cls, block_idx, layer_name, mask):
        key = (block_idx, layer_name)
        cls._masks[key] = mask.detach().clone()
    
    @classmethod
    def get_mask(cls, block_idx, layer_name, device=None):
        key = (block_idx, layer_name)
        if key in cls._masks:
            mask = cls._masks[key]
            if device is not None and mask.device != device:
                mask = mask.to(device)
            return mask
        return None
    
    @classmethod
    def clear_mask(cls, block_idx, layer_name):
        key = (block_idx, layer_name)
        if key in cls._masks:
            del cls._masks[key]
    
    @classmethod
    def clear_block_masks(cls, block_idx):
        keys_to_remove = [key for key in cls._masks if key[0] == block_idx]
        for key in keys_to_remove:
            del cls._masks[key]
    
    @classmethod
    def clear_all(cls):
        cls._masks.clear()
