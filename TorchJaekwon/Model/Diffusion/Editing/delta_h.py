import torch
from typing import Dict, Any

class DeltaHBase(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def pre_forward_hook(
        self,
        additional_data_dict: Dict[str, Any]
    ) -> None:
        raise NotImplementedError()

    def forward(
        self, 
        h: torch.Tensor, 
        t_emb: torch.Tensor,
        t: torch.Tensor):
        raise NotImplementedError()