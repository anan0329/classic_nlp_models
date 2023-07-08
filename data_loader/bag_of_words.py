import torch
from torch.utils.data import Dataset

class BagOfWordDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
def collate_fn(batch):
    inputs = [torch.tensor(elem[0]) for elem in batch]
    targets = torch.tensor([elem[1] for elem in batch], dtype=torch.long)
    
    offsets = [0] + [inp.shape[0] for inp in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    
    inputs = torch.cat(inputs)
    return inputs, offsets, targets

