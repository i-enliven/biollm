import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LocalBrainTrainer:
    """
    Manages localized, backpropagation-free training for the Brain-Inspired LLM.
    Updates are calculated layer-by-layer using predictive coding objectives.
    """
    def __init__(self, model_layers: List[nn.Module], lr: float = 1e-3, local_hebbian_rate: float = 0.01, optimizer_type: str = "adamw"):
        self.layers = model_layers
        self.lr = lr
        self.hebbian_rate = local_hebbian_rate
        self.optimizer_type = optimizer_type
        
        # Track local optimizer states independently for each layer
        if self.optimizer_type.lower() == "sgd":
            self.optimizers = [
                torch.optim.SGD(layer.parameters(), lr=self.lr, momentum=0.9) for layer in self.layers
            ]
        else:
            self.optimizers = [
                torch.optim.AdamW(layer.parameters(), lr=self.lr) for layer in self.layers
            ]

    def local_train_step(self, batch_inputs: torch.Tensor, target_tokens: torch.Tensor) -> List[float]:
        """
        Executes a forward-local learning step. 
        Each layer minimizes prediction error relative to target semantic tokens,
        encourages sparsity, and avoids representation collapse.
        Returns a list of local loss values for each layer.
        """
        current_hidden_state = batch_inputs.detach().clone()
        layer_losses = []
        
        # Enforce gradient isolation across layers
        for idx, layer in enumerate(self.layers):
            optimizer = self.optimizers[idx]
            optimizer.zero_grad()
            
            # Detach input to break the global backpropagation chain
            layer_input = current_hidden_state.detach()
            layer_input.requires_grad_(True)
            
            # Forward pass through the NVFP4 E/I layer
            layer_output = layer(layer_input)
            
            # Calculate Local Self-Supervised Loss
            # 1. Cosine similarity alignment with the semantic target context
            # We cast target_tokens to layer_output's device and dtype
            target_semantic = target_tokens.to(device=layer_output.device, dtype=layer_output.dtype)
            
            target_alignment = F.cosine_similarity(
                layer_output.view(-1, layer_output.size(-1)), 
                target_semantic.view(-1, target_semantic.size(-1)), 
                dim=-1
            ).mean()
            
            # Calculate the update delta (new information added by experts)
            delta = layer_output - layer_input
            
            # 2. Sparsity penalty (L1 norm) to encourage sparse firing on the updates
            sparsity_penalty = torch.mean(torch.abs(delta))
            
            # 3. Variance loss (VICReg-style) on the updates to prevent representation collapse
            # Adding epsilon for numerical stability
            std_y = torch.sqrt(torch.var(delta, dim=0) + 1e-4)
            variance_loss = torch.mean(F.relu(1.0 - std_y))
            
            # Combine losses: maximize alignment, minimize dense activity, prevent collapse
            local_loss = -target_alignment + (0.1 * sparsity_penalty) + (0.5 * variance_loss)
            
            # Trigger isolated backward pass
            local_loss.backward()
            
            # Local Anti-Hebbian updates for the lateral inhibition matrix
            with torch.no_grad():
                tokens = layer_input.view(-1, layer_input.size(-1))
                # Re-compute excitation for router updates
                excitation = F.relu(layer.excitatory_router(tokens))
                
                # Outer product of co-activations
                co_activation = torch.matmul(excitation.T, excitation) / (tokens.size(0) + 1e-8)
                
                # Apply anti-Hebbian update: more co-activation -> more inhibition
                layer.raw_inhibitory_weights.data += self.hebbian_rate * co_activation
                
                # Non-differentiable diagonal zero-out for self-inhibition avoidance
                layer.raw_inhibitory_weights.data.fill_diagonal_(0.0)
                
            # Apply gradient clipping to avoid numerical explosions
            torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
            
            # Step the local optimizer
            optimizer.step()
            
            # Store loss value for monitoring
            layer_losses.append(local_loss.item())
            
            # Pass output to the next layer (detached to release computational graph)
            current_hidden_state = layer_output.detach()
            
        return layer_losses
