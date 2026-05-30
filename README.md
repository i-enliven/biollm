# BioLLM: A Brain-Inspired LLM Architecture with Localized Learning & NVFP4 Simulation

BioLLM is a PyTorch-based neural network architecture that mirrors biological brain mechanisms. It replaces traditional dense Transformers and global backpropagation with neuromorphic sparsity, local predictive coding, and hippocampal episodic memory, simulated using NVIDIA Blackwell's NVFP4 (E2M1) microscaling precision.

---

## Architecture Blueprint

BioLLM replaces standard transformer designs with biological analogues:

```
            Excitatory-Inhibitory (E/I) Routing Block
           
                        [Input Tokens]
                              |
                              v
                    +--------------------+
                    | Excitatory Router  |  <-- Simulated NVFP4 Quantization
                    +---------+----------+
                              | excitation
                              v
                   /======================\
                  ||  E/I Settling Loop   || <-- Recurrent lateral inhibition
                  ||                      ||     (3 iterations)
                  ||  active - inhibition ||
                   \==========+===========/
                              |
                              v final scores
                    +--------------------+
                    |  Hard Top-1 Router |
                    +---------+----------+
                              |
                     +--------+--------+
                     |                 |
                     v (Expert 0)      v (Expert N)
                 +-------+         +-------+
                 | NVFP4 |  ...    | NVFP4 | <-- Segmented Experts
                 +-------+         +-------+
```

### 1. Neuromorphic Sparsity & Lateral Inhibition
Instead of dense attention, the architecture routes inputs through Excitatory-Inhibitory (E/I) routing blocks. A lateral inhibition matrix implements a localized competitive settling process (anti-Hebbian learning) to select active experts, encouraging specialized activation and preserving compute bandwidth.

### 2. Localized Training (Backpropagation-Free)
To avoid the memory footprint of storing activations across a deep network, BioLLM employs **Predictive Coding / Contrastive Local Learning**. 

```
           Input Tokens
                 |
                 v
        +------------------+
        | Embedding Layer  |
        +--------+---------+
                 |
                 v (gradient isolated)
        +------------------+
        | SparkBrainBlock  | <---+ Local Loss:
        |     Layer 0      |     | - Maximize Cosine Similarity to Target
        +--------+---------+     | - Minimize L1 Activation Sparsity
                 |               | - Prevent Representation Collapse (VICReg Variance)
                 v (gradient isolated)
        +------------------+
        | SparkBrainBlock  | <---+
        |     Layer 1      |
        +--------+---------+
                 |
                 v
            [LM Head] ---------> Cross-Entropy Loss to predict next token
```

Each layer optimizes its weights independently using a local self-supervised loss:
- **Target Alignment**: Maximizes cosine similarity between the layer's output and the target semantic context.
- **Sparsity Penalty**: L1 regularization to keep activations sparse.
- **Variance Regularization**: Prevents representation collapse (VICReg variance constraint) by enforcing that the activations have non-zero variance across features.

### 3. Hippocampal Episodic Memory
The architecture integrates a plastic **HippocampalMemory** module. It acts as an episodic key-value cache that updates dynamically with a ring buffer, applies exponential decay (forgetting rate), and allows soft attention-based retrieval of past token contexts.

### 4. High-Fidelity NVFP4 (E2M1) Simulation
Supports simulated block-wise microscaled E2M1 FP4 quantization on weights and activations, using Straight-Through Estimators (STE) to enable gradient flow during training:
- **E2M1 format representable values**: $\pm 0.0, \pm 0.5, \pm 1.0, \pm 1.5, \pm 2.0, \pm 3.0, \pm 4.0, \pm 6.0$.
- Block-wise scaling (default block size of 32) matches NVIDIA Blackwell hardware constraints.

---

## Codebase Structure

The codebase is organized into clean, reusable Python modules:

- [`biollm/layers/nvfp4.py`](file:///home/ienliven/Projects/BioLLM/biollm/layers/nvfp4.py): Block-wise microscaled E2M1 simulation and `NVFP4Linear` layer.
- [`biollm/layers/brain_block.py`](file:///home/ienliven/Projects/BioLLM/biollm/layers/brain_block.py): `SparkBrainBlock` implementation with E/I routing and lateral inhibition.
- [`biollm/layers/hippocampus.py`](file:///home/ienliven/Projects/BioLLM/biollm/layers/hippocampus.py): Plastic key-value episodic memory buffer.
- [`biollm/training/local_trainer.py`](file:///home/ienliven/Projects/BioLLM/biollm/training/local_trainer.py): Layer-wise gradient-isolated predictive coding trainer.
- [`biollm/utils/dataloader.py`](file:///home/ienliven/Projects/BioLLM/biollm/utils/dataloader.py): Byte-level streaming dataloader for `HuggingFaceFW/fineweb-edu`.

---

## Running the Code

### Installation
Ensure you have `uv` installed, then synchronize the environment:

```bash
uv sync
```

### Running the Demo
The demo streams text data from `HuggingFaceFW/fineweb-edu`, tokenizes it at the byte level, and trains a multi-layer BioLLM network using backprop-free local learning and hippocampal episodic retrieval.

Run the default demo with custom model dimensions, batch size, sequence length, and optimizer options. To run within 128 GB memory limits:
- Use `--optimizer sgd` to cut optimizer state memory in half (saves ~25 GB RAM).

```bash
NVTE_NVFP4_DISABLE_RHT=1 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1 NVTE_BACKWARD_OVERRIDE=dequantized PYTHONPATH=. uv run python3 run_demo.py \
  --steps 10000 \
  --d_model 4096 \
  --expert_dim 12288 \
  --experts 8 \
  --layers 16 \
  --batch_size 64 \
  --seq_len 128 \
  --optimizer adamw
```

Customize parameters or resume training from an existing checkpoint:
```bash
NVTE_NVFP4_DISABLE_RHT=1 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=1 NVTE_BACKWARD_OVERRIDE=dequantized PYTHONPATH=. uv run python3 run_demo.py \
  --steps 10000 \
  --d_model 4096 \
  --expert_dim 12288 \
  --experts 8 \
  --layers 16 \
  --batch_size 64 \
  --seq_len 128 \
  --optimizer adamw \
  --resume
```

### Running inference
Run inference on a trained checkpoint. To compile the model blocks and speed up execution, append the `--compile` flag:
```bash
PYTHONPATH=. uv run python3 run_inference.py --prompt "Biological networks are" --tokens 100 --compile
```

### Running Tests
Verify the installation by running the test suite:

```bash
PYTHONPATH=. uv run pytest
```

---

## References

1. **Microscaling Formats (MX) Specification**: Open Compute Project (OCP) standard defining the E2M1 format.
2. **Predictive Coding & Forward-Forward Algorithm**: Hinton, G. (2022). "The Forward-Forward Algorithm: Some Preliminary Investigations".
3. **VICReg: Variance-Invariance-Covariance Regularization**: Bardes, A., Ponce, J., & LeCun, Y. (2021). Self-supervised learning method preventing representation collapse.
4. **NVIDIA Blackwell Architecture**: Support for hardware-accelerated NVFP4 instructions.

