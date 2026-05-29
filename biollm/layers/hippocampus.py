import torch
import torch.nn as nn
import torch.nn.functional as F

class HippocampalMemory(nn.Module):
    """
    Hippocampal-inspired plastic episodic memory buffer.
    Maintains a key-value memory bank that decays over time.
    """
    def __init__(self, d_model: int, memory_size: int = 128, decay_rate: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        
        # Buffer to store keys, values, and strengths
        self.register_buffer("keys", torch.zeros(memory_size, d_model))
        self.register_buffer("values", torch.zeros(memory_size, d_model))
        self.register_buffer("strengths", torch.zeros(memory_size))
        
        # Pointer for ring buffer writing
        self.write_ptr = 0

    def write(self, new_keys: torch.Tensor, new_values: torch.Tensor):
        """
        Writes new key-value pairs into the episodic memory.
        new_keys: [num_items, d_model]
        new_values: [num_items, d_model]
        """
        # Ensure we don't track gradients through the memory buffer
        new_keys = new_keys.detach()
        new_values = new_values.detach()
        
        num_items = new_keys.size(0)
        if num_items == 0:
            return
            
        # Apply exponential decay to existing memory strengths
        self.strengths = self.strengths * self.decay_rate
        
        # Write to memory bank using a ring buffer
        for i in range(num_items):
            ptr = (self.write_ptr + i) % self.memory_size
            self.keys[ptr] = new_keys[i]
            self.values[ptr] = new_values[i]
            self.strengths[ptr] = 1.0 # Reset strength to 1.0 for new memory
            
        self.write_ptr = (self.write_ptr + num_items) % self.memory_size

    def read(self, queries: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """
        Retrieves representations from memory using soft attention based on similarity.
        queries: [batch, seq_len, d_model]
        returns: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = queries.shape
        flat_queries = queries.view(-1, d_model)
        
        # If memory is completely empty, return zeros
        if torch.sum(self.strengths) == 0:
            return torch.zeros_like(queries)
            
        # Normalize keys and queries for cosine similarity
        norm_queries = F.normalize(flat_queries, p=2, dim=-1)
        norm_keys = F.normalize(self.keys, p=2, dim=-1)
        
        # Compute similarity: [N, memory_size]
        sim = torch.matmul(norm_queries, norm_keys.T)
        
        # Scale similarity by memory strengths
        weighted_sim = sim * self.strengths.unsqueeze(0)
        
        # Apply soft attention over memory slots
        attn_weights = F.softmax(weighted_sim / temperature, dim=-1)
        
        # Retrieve weighted sum of values
        retrieved = torch.matmul(attn_weights, self.values)
        
        return retrieved.view(batch, seq_len, d_model)
