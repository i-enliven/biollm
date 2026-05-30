import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for native NVIDIA Transformer Engine (TE) support on DGX OS / Blackwell
HAS_TE = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, NVFP4BlockScaling
    HAS_TE = True
except (ImportError, ModuleNotFoundError):
    pass

class E2M1Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, block_size=32):
        orig_shape = x.shape
        flat_x = x.view(-1)
        num_elements = flat_x.numel()
        
        if num_elements == 0:
            return x
            
        # Check if padding is needed
        pad_len = (block_size - (num_elements % block_size)) % block_size
        if pad_len > 0:
            flat_x = torch.cat([flat_x, torch.zeros(pad_len, device=x.device, dtype=x.dtype)])
            
        blocked_x = flat_x.view(-1, block_size)
        
        # Find max absolute value per block
        block_max = torch.max(torch.abs(blocked_x), dim=-1, keepdim=True)[0]
        
        # Compute scale factor (max value in E2M1 is 6.0)
        scale_factor = block_max / 6.0
        scale_factor = torch.clamp(scale_factor, min=1e-12)
        
        # Scale input
        scaled_x = blocked_x / scale_factor
        
        # E2M1 representable values
        e2m1_values = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 
                                     0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], 
                                   device=x.device, dtype=x.dtype)
        
        # Nearest neighbor mapping via broadcasting
        diffs = torch.abs(scaled_x.unsqueeze(-1) - e2m1_values)
        indices = torch.argmin(diffs, dim=-1)
        quantized_scaled_x = e2m1_values[indices]
        
        # Scale back
        quantized_blocked_x = quantized_scaled_x * scale_factor
        
        # Flatten and remove padding
        quantized_flat_x = quantized_blocked_x.view(-1)
        if pad_len > 0:
            quantized_flat_x = quantized_flat_x[:-pad_len]
            
        return quantized_flat_x.view(orig_shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        return grad_output, None

def quantize_to_fp4_e2m1(x, block_size=32):
    """
    Quantizes a tensor to OCP FP4 E2M1 format using block-wise microscaling.
    Uses Straight-Through Estimation (STE) in the backward pass.
    """
    return E2M1Quantizer.apply(x, block_size)

if HAS_TE:
    class NVFP4Linear(te.Linear):
        """
        Hardware-native NVIDIA Blackwell FP4/FP8 linear layer wrapper.
        """
        def __init__(self, in_features: int, out_features: int, bias: bool = True, block_size: int = 32):
            # te.Linear handles native execution on Blackwell hardware
            super().__init__(in_features, out_features, bias=bias)
else:
    class NVFP4Linear(nn.Module):
        """
        Linear layer simulating NVIDIA Blackwell NVFP4 (E2M1) quantization for weights and activations.
        """
        def __init__(self, in_features: int, out_features: int, bias: bool = True, block_size: int = 32):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.block_size = block_size
            
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter('bias', None)
                
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            quantized_weight = quantize_to_fp4_e2m1(self.weight, self.block_size)
            quantized_x = quantize_to_fp4_e2m1(x, self.block_size)
            return F.linear(quantized_x, quantized_weight, self.bias)
