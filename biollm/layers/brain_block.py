import torch
import torch.nn as nn
import torch.nn.functional as F
from biollm.layers.nvfp4 import NVFP4Linear

class SparkBrainBlock(nn.Module):
    """
    Brain-inspired layer utilizing dynamic E/I routing over 
    simulated NVFP4 (E2M1) structures.
    """
    def __init__(self, d_model: int, num_experts: int = 8, expert_dim: int = 256):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        
        # Layer Normalization to stabilize internal activation scale
        self.ln = nn.LayerNorm(d_model)
        
        # Sparse Excitatory Router (quantized)
        self.excitatory_router = NVFP4Linear(d_model, num_experts)
        
        # Lateral Inhibition Matrix (The Brain's localized braking system)
        self.raw_inhibitory_weights = nn.Parameter(torch.ones(num_experts, num_experts) * 0.15)
        
        # NVFP4-quantized Expert Layers
        self.experts = nn.ModuleList([
            nn.Sequential(
                NVFP4Linear(d_model, expert_dim),
                nn.ReLU(),
                NVFP4Linear(expert_dim, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq_len, d_model]
        batch, seq_len, d_model = x.shape
        
        # Pre-LN stabilizes activations across deep layers
        normalized_x = self.ln(x)
        tokens = normalized_x.view(-1, d_model)
        
        # Step 1: Fire Initial Excitatory pathways 
        excitation = F.relu(self.excitatory_router(tokens))
        
        # Step 2: Enforce Lateral Inhibition (E/I Settling loop)
        # Avoid in-place modification of autograd variables (fill_diagonal_) by using a mask
        raw_inhib = F.relu(self.raw_inhibitory_weights)
        diagonal_mask = torch.eye(self.num_experts, device=self.raw_inhibitory_weights.device)
        inhibitory_matrix = raw_inhib * (1.0 - diagonal_mask)
        
        inhibition = torch.zeros_like(excitation)
        
        for _ in range(3): # Local recurrent settling loop
            active_state = F.relu(excitation - inhibition)
            inhibition = torch.matmul(active_state, inhibitory_matrix) * 0.5
            
        final_routing_scores = F.relu(excitation - inhibition)
        
        # Step 3: Hard Top-1 Selection
        top_scores, top_indices = torch.topk(final_routing_scores, k=1, dim=-1)
        
        # Step 4: Dispatch to the NVFP4-quantized Experts
        output_tokens = torch.zeros_like(tokens)
        for expert_idx in range(self.num_experts):
            mask = (top_indices.squeeze(-1) == expert_idx)
            if not mask.any():
                continue
                
            selected_tokens = tokens[mask]
            # Run calculations inside the NVFP4 expert layer
            expert_out = self.experts[expert_idx](selected_tokens)
            
            # Apply dynamic neural amplitude scaling
            scale = final_routing_scores[mask, expert_idx].unsqueeze(-1)
            output_tokens[mask] = expert_out * scale
            
        # Return residual connection to preserve signal and gradient flow
        return x + output_tokens.view(batch, seq_len, d_model)
