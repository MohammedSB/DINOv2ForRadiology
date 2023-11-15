import torch
import numpy as np
from dinov2.data.utils import get_fewshot_in_nih
from dinov2.data.datasets import NIHChestXray

class FewShotDatasetWrapper(torch.utils.data.Subset):
    def __init__(self, dataset, shots=16):
        dataset_len = dataset.__len__()
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            if type(dataset.datasets[0]) == NIHChestXray:
                indices = get_fewshot_in_nih(dataset, shots=shots)
            else:
                indices = np.arange(0, dataset_len, dataset_len/shots).round().astype("int")
        elif type(dataset) == NIHChestXray:
            indices = get_fewshot_in_nih(dataset, shots=shots)
        else:
            indices = np.arange(0, dataset_len, dataset_len/shots).round().astype("int")
        self.subset = torch.utils.data.Subset(dataset, indices)
            
    def __getitem__(self, index):
        return self.subset.__getitem__(index)

    def __len__(self):
        return self.subset.__len__()
    
class SystemicSamplerWrapper(torch.utils.data.Subset):
    def __init__(self, dataset, num_samples=5000):
        dataset_len = dataset.__len__()
        indices = np.arange(0, dataset_len, dataset_len/num_samples).round().astype("int")
        self.subset = torch.utils.data.Subset(dataset, indices)
            
    def __getitem__(self, index):
        return self.subset.__getitem__(index)

    def __len__(self):
        return self.subset.__len__()