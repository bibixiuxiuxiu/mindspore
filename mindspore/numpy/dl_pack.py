import torch.utils.dlpack as dlpack
import torch
import numpy as np
class tensor_dlpack:
    def __init__(self,x):
        if type(x) is np.ndarray:
            self.x =torch.tensor(x)
        else:
            self.x = x

    def from_dlpack(self):
        return dlpack.to_dlpack(self.x)

    def to_dlpack(self):
        return dlpack.from_dlpack(self.x)
