import os
# Configure HuggingFace cache directory to be in the workspace by default to avoid permission errors
if "HF_HOME" not in os.environ:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["HF_HOME"] = os.path.abspath(os.path.join(current_dir, "../../.hf_cache"))

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

class FineWebByteDataset(IterableDataset):
    """
    An IterableDataset that streams HuggingFaceFW/fineweb-edu,
    tokenizes text at the byte level (vocab size = 256),
    and yields input-target sequence pairs of length seq_len.
    """
    def __init__(self, seq_len: int = 128, split: str = "train", sample_name: str = "sample-10BT"):
        super().__init__()
        self.seq_len = seq_len
        # Load dataset in streaming mode to avoid downloading massive files
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            name=sample_name, 
            split=split, 
            streaming=True
        )
        
    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            text = sample["text"]
            # Convert text to UTF-8 bytes to treat each byte as a token ID (0-255)
            tokens = list(text.encode("utf-8", errors="ignore"))
            buffer.extend(tokens)
            
            # Yield full sequences
            while len(buffer) >= self.seq_len + 1:
                seq_in = buffer[:self.seq_len]
                seq_target = buffer[1:self.seq_len + 1]
                
                yield (
                    torch.tensor(seq_in, dtype=torch.long), 
                    torch.tensor(seq_target, dtype=torch.long)
                )
                # Keep sliding window / overlap or discard consumed tokens
                buffer = buffer[self.seq_len:]

def get_dataloader(batch_size: int = 4, seq_len: int = 128) -> DataLoader:
    """
    Helper function to get a PyTorch DataLoader for streaming fineweb-edu.
    """
    dataset = FineWebByteDataset(seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size)
