from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

class MNISTDigitDataset(Dataset):
    """
    Class that inherits from the 'torch.utils.data.Dataset' class and serves as
    a wrapper for paired data and label arrays used to feed inputs into a model.
    """
    def __init__(self,
                npData: np.ndarray,
                npLabels: np.ndarray,
                transform: Optional[Callable] = None
    ) -> None:
        """
        Constructor.
        Args:
            self: Instance that called the function.
            npData (np.ndarray): NumPy array of images with shape (N, H, W).
            npLabels (np.ndarray): NumPy array or list of labels, length N.
            transform (Callable, optional): Optional transform to apply to each image.

        Returns:
            None: Constructor object.
        """
        if (npData.shape[0] != npLabels.shape[0]):
            raise ValueError(f"Mismatch in number of samples: npData has {npData.shape[0]} samples, but npLabels has {npLabels.shape[0]}")

        self.data = npData
        self.labels = npLabels
        self.transform = transform

    def __len__(self) -> int:
        """
        Associated 'length' attribute for object is defined as amount of data and label objects.
        """
        return int(self.data.shape[0])
    
    def __getitem__(self, 
                    idx: Union[int, slice]
    ) -> Union[Tuple[torch.Tensor, int], List[Tuple[torch.Tensor, int]]]:
        """
        Operator for obj[key] accesses, also deals with slicing.
        Args:
            self: Caller instance.
            idx (Union[int, slice]): Index of item or slice.

        Returns:
            image (torch.Tensor): 2D tensor representing the greyscale image.
            label (int): Ground-truth class label.
        """
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        if not (0 <= idx < len(self)):
            raise IndexError("Index out of range!")
        
        image = self.data[idx]
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
        
        return image, label
