import os
import time
import argparse
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from biollm import SparkBrainBlock, HippocampalMemory, LocalBrainTrainer, get_dataloader

class BackgroundSaver:
    def __init__(self):
        self.thread = None

    def save(self, checkpoint_data, path):
        if self.thread is not None and self.thread.is_alive():
            print("\n -> Warning: Previous checkpoint save is still in progress. Skipping this save step to avoid disk contention.")
            return False
            
        self.thread = threading.Thread(
            target=self._run_save, 
            args=(checkpoint_data, path),
            daemon=True
        )
        self.thread.start()
        return True

    def _run_save(self, checkpoint_data, path):
        try:
            temp_path = path + ".tmp"
            torch.save(checkpoint_data, temp_path)
            if os.path.exists(temp_path):
                os.replace(temp_path, path)
                print(f"\n -> [Background Save] Checkpoint saved successfully to {path}")
        except Exception as e:
            print(f"\n -> [Background Save] Error saving checkpoint: {e}")

class LMHead(nn.Module):
    """
    Language Model head to project internal representations back to vocabulary log probabilities.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def run_demo(
    max_steps: int = 1000, 
    resume: bool = False,
    d_model: int = 256,
    expert_dim: int = 1024,
    num_experts: int = 8,
    num_layers: int = 3,
    batch_size: int = 8,
    seq_len: int = 64,
    optimizer_type: str = "adamw"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"Starting BioLLM Local Training Demo on device: {device}")
    print(f"==================================================")
    
    # Hyperparameters
    vocab_size = 256  # Byte-level vocabulary
    
    # 1. Load Checkpoint first to auto-detect architecture configuration if resuming
    checkpoint_path = "biollm_checkpoint.pt"
    checkpoint = None
    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Auto-detect and override hyperparameters
        d_model = checkpoint["embedding"]["weight"].shape[1]
        num_layers = len(checkpoint["layers"])
        num_experts = checkpoint["layers"][0]["excitatory_router.weight"].shape[0]
        expert_dim = checkpoint["layers"][0]["experts.0.0.weight"].shape[0]
        
        print(f"Resuming with auto-detected configuration:")
        print(f" - d_model: {d_model}")
        print(f" - expert_dim: {expert_dim}")
        print(f" - num_experts: {num_experts}")
        print(f" - num_layers: {num_layers}")
    
    # 2. Initialize Embeddings & Head
    # We use a fixed embedding to map byte tokens to semantic space
    embedding = nn.Embedding(vocab_size, d_model).to(device).bfloat16()
    # Freeze embedding parameters to demonstrate purely local layer-wise learning
    for p in embedding.parameters():
        p.requires_grad = False
        
    lm_head = LMHead(d_model, vocab_size).to(device).bfloat16()
    if optimizer_type.lower() == "sgd":
        lm_optimizer = torch.optim.SGD(lm_head.parameters(), lr=1e-3, momentum=0.9)
    else:
        lm_optimizer = torch.optim.AdamW(lm_head.parameters(), lr=1e-3)
    
    # 3. Initialize BioLLM Layers
    layers = [
        SparkBrainBlock(d_model=d_model, num_experts=num_experts, expert_dim=expert_dim).to(device).bfloat16()
        for _ in range(num_layers)
    ]
    
    # 4. Initialize Hippocampal Memory
    memory = HippocampalMemory(d_model=d_model, memory_size=512, decay_rate=0.98).to(device).bfloat16()
    
    # 5. Initialize Local Trainer
    trainer = LocalBrainTrainer(layers, lr=5e-4, local_hebbian_rate=0.005, optimizer_type=optimizer_type)
    
    # 6. Apply state dicts if checkpoint is loaded
    if checkpoint is not None:
        embedding.load_state_dict(checkpoint["embedding"])
        for idx, layer in enumerate(layers):
            layer.load_state_dict(checkpoint["layers"][idx])
        lm_head.load_state_dict(checkpoint["lm_head"])
        memory.keys.copy_(checkpoint["memory_keys"])
        memory.values.copy_(checkpoint["memory_values"])
        memory.strengths.copy_(checkpoint["memory_strengths"])
        print("Resumed training state successfully.")
    
    # 6. Initialize Background Saver for non-blocking checkpoints
    bg_saver = BackgroundSaver()
    
    # 7. Initialize Transformer Engine (TE) Native FP4 Context
    from biollm.layers.nvfp4 import HAS_TE
    if HAS_TE:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, NVFP4BlockScaling
        print("Transformer Engine detected. Enabling native Blackwell NVFP4 (E2M1) training...")
        recipe = NVFP4BlockScaling()
        def native_fp4_autocast():
            return te.autocast(enabled=True, recipe=recipe)
    else:
        import contextlib
        print("Transformer Engine not detected. Falling back to simulated FP4 training...")
        def native_fp4_autocast():
            return contextlib.nullcontext()
            
    # 8. Initialize Streaming DataLoader
    print("Initializing streamed dataloader for HuggingFaceFW/fineweb-edu...")
    dataloader = get_dataloader(batch_size=batch_size, seq_len=seq_len)
    data_iter = iter(dataloader)
    
    print("\nStarting Training Loop (Local Backprop-Free Predictive Coding)...")
    print(f"{'Step':<6} | {'Mean Layer Loss':<16} | {'LM Head Loss':<12} | {'Speed (tok/s)':<14}")
    print("-" * 60)
    
    start_time = time.time()
    try:
        for step in range(1, max_steps + 1):
            try:
                x, y = next(data_iter)
            except StopIteration:
                print("Dataset stream ended.")
                break
                
            x = x.to(device)
            y = y.to(device)
            
            # Mapping inputs and targets to semantic representations
            with torch.no_grad():
                input_embeddings = embedding(x)  # [batch, seq_len, d_model]
                target_embeddings = embedding(y) # [batch, seq_len, d_model]
                
            # Read from Hippocampal episodic memory to retrieve context
            retrieved_context = memory.read(input_embeddings, temperature=0.5)
            
            # Integrate episodic memory context
            layer_input = input_embeddings + retrieved_context
            
            # Run local trainer step and LM head updates under native FP4 autocast context
            with native_fp4_autocast():
                # Run local trainer step (backprop isolated to each block)
                local_losses = trainer.local_train_step(layer_input, target_embeddings)
                
                # Run forward pass without gradients to compute final output and update LM head
                with torch.no_grad():
                    hidden = layer_input.clone()
                    for layer in layers:
                        hidden = layer(hidden)
                    final_output = hidden
                    
                # Update LM Head (local classification update)
                lm_optimizer.zero_grad()
                # Normalize representations before linear projection to avoid logit saturation
                norm_final_output = (final_output - final_output.mean(dim=-1, keepdim=True)) / (final_output.std(dim=-1, keepdim=True) + 1e-5)
                logits = lm_head(norm_final_output) # [batch, seq_len, vocab_size]
                lm_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                lm_loss.backward()
                
                # Apply gradient clipping to LM Head to prevent logit magnitude explosion
                torch.nn.utils.clip_grad_norm_(lm_head.parameters(), max_norm=1.0)
                lm_optimizer.step()
            
            # Write step context to Hippocampal memory (associate inputs with final semantic outputs)
            # Flatten batch and sequence dimensions for memory storage
            flat_keys = input_embeddings.view(-1, d_model)
            flat_vals = final_output.view(-1, d_model)
            # Write a fraction to keep it fast
            memory.write(flat_keys[:16], flat_vals[:16])
            
            # Speed logging
            elapsed = time.time() - start_time
            tokens_processed = step * batch_size * seq_len
            tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            
            if step % 10 == 0 or step == 1:
                mean_layer_loss = sum(local_losses) / len(local_losses)
                print(f"{step:<6} | {mean_layer_loss:<16.4f} | {lm_loss.item():<12.4f} | {tokens_per_sec:.1f}")
                
            # Save intermediate checkpoints to allow interruption and resuming
            if step % 100 == 0:
                checkpoint = {
                    "embedding": embedding.state_dict(),
                    "layers": [layer.state_dict() for layer in layers],
                    "lm_head": lm_head.state_dict(),
                    "memory_keys": memory.keys,
                    "memory_values": memory.values,
                    "memory_strengths": memory.strengths,
                }
                checkpoint_path = "biollm_checkpoint.pt"
                bg_saver.save(checkpoint, checkpoint_path)
                print(f" -> [Step {step}] Intermediate checkpoint write triggered in background...")
                
            # Periodic garbage collection and cache clearing to prevent memory leak accumulation
            if step % 50 == 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C). Cleaning up background threads...")
        if bg_saver.thread is not None and bg_saver.thread.is_alive():
            print("Waiting for active background checkpoint saving to complete...")
            bg_saver.thread.join()
        print("Exited cleanly.")
        return
        
    print("-" * 80)
    
    # Wait for any active background saving to complete before final save
    if bg_saver.thread is not None and bg_saver.thread.is_alive():
        print("Waiting for background checkpoint save to complete before final write...")
        bg_saver.thread.join()
        
    # Save the trained weights and memory states to a file
    checkpoint = {
        "embedding": embedding.state_dict(),
        "layers": [layer.state_dict() for layer in layers],
        "lm_head": lm_head.state_dict(),
        "memory_keys": memory.keys,
        "memory_values": memory.values,
        "memory_strengths": memory.strengths,
    }
    checkpoint_path = "biollm_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved trained weights and episodic memory states to: {checkpoint_path}")
    
    print(f"Finished BioLLM training demo. Total time: {time.time() - start_time:.2f} seconds.")
    print(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioLLM Local Training Demo")
    parser.add_argument(
        "--steps", 
        type=int, 
        default=1000, 
        help="Number of training steps (default: 1000)"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume training from standard checkpoint if present"
    )
    parser.add_argument(
        "--d_model", 
        type=int, 
        default=256, 
        help="Hidden dimension size of the model (default: 256)"
    )
    parser.add_argument(
        "--expert_dim", 
        type=int, 
        default=1024, 
        help="Bottleneck dimension for expert layers (default: 1024)"
    )
    parser.add_argument(
        "--experts", 
        type=int, 
        default=8, 
        help="Number of experts per layer block (default: 8)"
    )
    parser.add_argument(
        "--layers", 
        type=int, 
        default=3, 
        help="Number of layer blocks in the model (default: 3)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for training (default: 8)"
    )
    parser.add_argument(
        "--seq_len", 
        type=int, 
        default=64, 
        help="Sequence length context window (default: 64)"
    )
    parser.add_argument(
        "--optimizer", 
        type=str, 
        choices=["adamw", "sgd"], 
        default="adamw", 
        help="Optimizer to use for training (default: adamw)"
    )
    args = parser.parse_args()
    run_demo(
        max_steps=args.steps, 
        resume=args.resume,
        d_model=args.d_model,
        expert_dim=args.expert_dim,
        num_experts=args.experts,
        num_layers=args.layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        optimizer_type=args.optimizer
    )


