import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba

class Bimamba_outer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.forward_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)    
        self.backward_mamba = Mamba(d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.output_proj = nn.Linear(2*self.d_model, self.d_model)     
    
    def forward(self, hidden_input):
        forward_output = self.forward_mamba(hidden_input)
        backward_output = self.backward_mamba(hidden_input.flip([1]))
        res = torch.cat((forward_output, backward_output.flip([1])), dim=-1)
        res = self.output_proj(res)
        return res
