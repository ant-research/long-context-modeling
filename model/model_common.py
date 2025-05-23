from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ModelOutput:
    ar_loss: Optional[torch.FloatTensor] = None,  # next token prediction loss
    ae_loss: Optional[torch.FloatTensor] = None,  # mlm loss
    ncp_loss: Optional[torch.FloatTensor] = None, # next chunk prediction loss
    causal_matrix: Optional[torch.FloatTensor] = None, 
    logits: Optional[torch.FloatTensor] = None,
    total_loss: Optional[torch.FloatTensor] = None