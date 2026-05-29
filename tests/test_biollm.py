import pytest
import torch
import torch.nn as nn
from biollm import (
    quantize_to_fp4_e2m1,
    NVFP4Linear,
    SparkBrainBlock,
    HippocampalMemory,
    LocalBrainTrainer,
)

def test_quantize_to_fp4_e2m1():
    # Create random tensor
    x = torch.randn(10, 64)
    x.requires_grad_(True)
    
    # Quantize
    block_size = 32
    quantized_x = quantize_to_fp4_e2m1(x, block_size=block_size)
    
    # Shape check
    assert quantized_x.shape == x.shape
    
    # Check representable values
    # For each block of size 32, if we divide the block values by (block_max / 6.0),
    # the scaled values should be exactly in the E2M1 set.
    flat_q = quantized_x.view(-1)
    flat_x = x.view(-1)
    
    # Reconstruct blocks
    blocked_q = flat_q.view(-1, block_size)
    blocked_x = flat_x.view(-1, block_size)
    
    block_max = torch.max(torch.abs(blocked_x), dim=-1, keepdim=True)[0]
    scale_factor = torch.clamp(block_max / 6.0, min=1e-12)
    
    scaled_q = blocked_q / scale_factor
    
    # E2M1 valid set
    valid_values = {-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    
    # Let's check that all values are very close to one of the valid values (due to floating point precision)
    for val in scaled_q.flatten().tolist():
        # Find minimum distance to any valid value
        min_dist = min(abs(val - v) for v in valid_values)
        assert min_dist < 1e-4, f"Value {val} is not in the E2M1 set!"
        
    # Check gradient flow (STE)
    loss = quantized_x.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.all(x.grad == 1.0) # Derivative of x.sum() with respect to x is 1.0

def test_nvfp4_linear():
    layer = NVFP4Linear(in_features=16, out_features=8, bias=True)
    x = torch.randn(4, 16, requires_grad=True)
    
    out = layer(x)
    assert out.shape == (4, 8)
    
    # Backprop
    out.sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None

def test_brain_block():
    block = SparkBrainBlock(d_model=32, num_experts=4, expert_dim=16)
    x = torch.randn(2, 8, 32, requires_grad=True)
    
    out = block(x)
    assert out.shape == (2, 8, 32)
    
    # Backprop should work without in-place modification errors
    out.sum().backward()
    assert x.grad is not None
    assert block.raw_inhibitory_weights.grad is not None

def test_hippocampal_memory():
    memory = HippocampalMemory(d_model=16, memory_size=10, decay_rate=0.8)
    
    # Initial read should return zeros
    queries = torch.randn(2, 4, 16)
    out = memory.read(queries)
    assert torch.all(out == 0.0)
    
    # Write to memory
    keys = torch.randn(3, 16)
    values = torch.randn(3, 16)
    memory.write(keys, values)
    
    # Read should return non-zero representations
    out = memory.read(queries)
    assert out.shape == (2, 4, 16)
    assert not torch.all(out == 0.0)
    
    # Memory strength check (written elements are 1.0, rest are 0.0)
    assert torch.all(memory.strengths[:3] == 1.0)
    assert torch.all(memory.strengths[3:] == 0.0)
    
    # Write more to cause decay and pointer wrapping (without overwriting index 0 yet)
    more_keys = torch.randn(5, 16)
    more_values = torch.randn(5, 16)
    memory.write(more_keys, more_values)
    
    # The first 3 should decay
    assert memory.strengths[0] == pytest.approx(0.8)
    assert memory.strengths[1] == pytest.approx(0.8)
    assert memory.strengths[2] == pytest.approx(0.8)
    # The newly written ones should be 1.0
    assert torch.all(memory.strengths[3:8] == 1.0)
    # The rest should still be 0.0
    assert torch.all(memory.strengths[8:] == 0.0)

def test_local_trainer():
    layers = [
        SparkBrainBlock(d_model=16, num_experts=4, expert_dim=8),
        SparkBrainBlock(d_model=16, num_experts=4, expert_dim=8)
    ]
    trainer = LocalBrainTrainer(layers, lr=1e-3, local_hebbian_rate=0.01)
    
    inputs = torch.randn(2, 4, 16)
    targets = torch.randn(2, 4, 16)
    
    # Record initial weights of excitatory router in layer 0
    init_weight = layers[0].excitatory_router.weight.clone()
    init_inhib = layers[0].raw_inhibitory_weights.clone()
    
    losses = trainer.local_train_step(inputs, targets)
    
    assert len(losses) == 2
    assert all(isinstance(l, float) for l in losses)
    
    # Verify weights changed after training step
    assert not torch.equal(layers[0].excitatory_router.weight, init_weight)
    assert not torch.equal(layers[0].raw_inhibitory_weights, init_inhib)
