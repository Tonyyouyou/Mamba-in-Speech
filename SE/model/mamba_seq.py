import math, pdb
from functools import partial
import json
import os
import torch.nn.functional as F
from collections import namedtuple
from einops.layers.torch import Rearrange
from .torchaudio_conformer import _sequence_mask

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .mamba.mamba_ssm.models.mixer_seq_simple import create_block, _init_weights
# from dataclasses import field



class Mamba_(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layer: int,
        inptarget: str,
        ssm_cfg = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg = None,
        fused_add_norm = True,
        residual_in_fp32 =True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        if inptarget in ['ComplexcIRM']: input_dim = input_dim * 2
        self.input_layer = nn.Sequential(
                            nn.Linear(input_dim, d_model, bias=False),
                            nn.LayerNorm(d_model),
                            nn.ReLU()
                            )
        self.inptarget = inptarget
        # nn.Linear(input_dim, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        if inptarget == 'MagcIRM': input_dim = input_dim * 2
        self.output_layer = nn.Linear(d_model, input_dim, bias=True)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, x, lengths, inference_params=None):
        seq_mask = _sequence_mask(lengths)
        hidden_states = self.input_layer(x)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        # pdb.set_trace()
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # pdb.set_trace()
        if self.inptarget in ['MagIRM', 'MagPSM', 'MagSMM']:
            output = F.sigmoid(self.output_layer(hidden_states))
        if self.inptarget in ['MagMag']:
            output = F.relu(self.output_layer(hidden_states))
        if self.inptarget in ['MagcIRM', 'ComplexcIRM']:
            output = self.output_layer(x)

        return output[seq_mask], lengths
        # return hidden_states

class MambaDCLayer(nn.Module):
    def __init__(self, 
                d_model, 
                d_state = 16,
                d_conv = 4,
                expand = 2,
                ):
        super(MambaDCLayer, self).__init__()

        def get_parameter():
            return nn.Parameter(torch.tensor(0.0))
        
        self.MambaLayer = nn.Sequential(
                            nn.LayerNorm(d_model),
                            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                            )
        self.conv_layer = nn.Sequential(
                            nn.LayerNorm(d_model),
                            Rearrange('b n c -> b c n'),
                            nn.Conv1d(d_model, d_model, kernel_size=31, groups=d_model, padding='same'),
                            Rearrange('b c n -> b n c')
                            )
        
    def forward(self, x, mask=None, lengths=None):
        
        attn_output = x + self.MambaLayer(x)
        out = self.conv_layer(attn_output)
        return out + attn_output
        # return attn_output*self.skip_scale2 + self.conv_layer(attn_output)



class MambaDC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layer: int,
        inptarget: str,
    ) -> None:
        super(MambaDC, self).__init__()

        if inptarget in ['ComplexcIRM']: input_dim = input_dim * 2
        self.input_layer= nn.Sequential(
                            nn.Linear(input_dim, d_model, bias=False),
                            nn.LayerNorm(d_model),
                            nn.ReLU()
                            )
        self.inptarget = inptarget
        self.encoder_layers = nn.ModuleList([MambaDCLayer(d_model=d_model) for _ in range(n_layer)])
        if inptarget == 'MagcIRM': input_dim = input_dim * 2
        self.output_layer = torch.nn.Linear(d_model, input_dim, bias=True)
    

    def forward(self, x, lengths):
        seq_mask = _sequence_mask(lengths)
        x = self.input_layer(x)
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, seq_mask, lengths)

        if self.inptarget in ['MagIRM', 'MagPSM', 'MagSMM']:
            output = F.sigmoid(self.output_layer(x))
        if self.inptarget in ['MagMag']:
            output = F.relu(self.output_layer(x))
        if self.inptarget in ['MagcIRM', 'ComplexcIRM']:
            output = self.output_layer(x)
        return output[seq_mask], seq_mask

        


class BiMamba_(Mamba_):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layer: int,
        inptarget: str,
        ssm_cfg = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg = None,
        fused_add_norm = True,
        residual_in_fp32 =True,
        device=None,
        dtype=None,
        fuse_mode=None,
        type=None,
    ) -> None:
        super().__init__(input_dim,
                         d_model,
                         n_layer*2,
                         inptarget,
                         ssm_cfg,
                         norm_epsilon,
                         rms_norm,
                         initializer_cfg,
                         fused_add_norm,
                         residual_in_fp32,
                         device,
                         dtype,
                         )
        # assert n_layer%2 == 0
        self.fuse_mode = fuse_mode
        if fuse_mode == 'Concat':
            self.fuse_layers = nn.ModuleList(
                [
                nn.Linear(d_model*2, d_model, bias=False)
                for i in range(n_layer)
                ]
            )
        # if fuse_mode == 'Add':
        #     self.fuse_layers = nn.ModuleList(
        #         [
        #         nn.Linear(d_model, d_model, bias=False)
        #         for i in range(n_layer)
        #         ]
        #     )
        
    def forward(self, x, lengths, inference_params=None):
        seq_mask = _sequence_mask(lengths)
        hidden_states = self.input_layer(x)
        residual = None
        for i in range(len(self.layers)//2):
            hidden_states_f, residual_f = self.layers[i*2](
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states_b, residual_b = self.layers[i*2+1](
                hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
            )
            if self.fuse_mode == 'Add':
                # hidden_states = self.fuse_layers[i](hidden_states_f + hidden_states_b.flip([1]))
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
            if self.fuse_mode == 'Concat':
                hidden_states = self.fuse_layers[i](torch.cat((hidden_states_f, hidden_states_b.flip([1])), dim=-1))
            residual = residual_f + residual_b.flip([1])

        # for layer in self.layers:
        #     hidden_states, residual = layer(
        #         hidden_states, residual, inference_params=inference_params
        #     )
        # pdb.set_trace()
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # pdb.set_trace()
        if self.inptarget in ['MagIRM', 'MagPSM', 'MagSMM']:
            output = F.sigmoid(self.output_layer(hidden_states))
        if self.inptarget in ['MagMag']:
            output = F.relu(self.output_layer(hidden_states))
        if self.inptarget in ['MagcIRM', 'ComplexcIRM']:
            output = self.output_layer(hidden_states)

        return output[seq_mask], lengths
    

class BiMambaDC(BiMamba_):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layer: int,
        inptarget: str,
        ssm_cfg = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg = None,
        fused_add_norm = True,
        residual_in_fp32 =True,
        device=None,
        dtype=None,
        fuse_mode=None,
        type=None,
    ) -> None:
        super().__init__(
            input_dim,
            d_model,
            n_layer,
            inptarget,
            ssm_cfg,
            norm_epsilon,
            rms_norm,
            initializer_cfg,
            fused_add_norm,
            residual_in_fp32,
            device,
            dtype,
            fuse_mode,
            type
            )
        self.conv_layers = nn.ModuleList([
                            nn.Sequential(
                            nn.LayerNorm(d_model),
                            Rearrange('b n c -> b c n'),
                            nn.Conv1d(d_model, d_model, kernel_size=31, groups=d_model, padding='same'),
                            Rearrange('b c n -> b n c')
                            )
                            for _ in range(n_layer)
                            ])
    
    def forward(self, x, lengths, inference_params=None):
        seq_mask = _sequence_mask(lengths)
        hidden_states = self.input_layer(x)
        # pdb.set_trace()
        residual = None
        for i in range(len(self.layers)//2):
            hidden_states_f, residual_f = self.layers[i*2](
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states_b, residual_b = self.layers[i*2+1](
                hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
            )
            if self.fuse_mode == 'Add':
                # hidden_states = self.fuse_layers[i](hidden_states_f + hidden_states_b.flip([1]))
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                hidden_states = hidden_states + self.conv_layers[i](hidden_states)
            if self.fuse_mode == 'Concat':
                hidden_states = self.fuse_layers[i](torch.cat((hidden_states_f, hidden_states_b.flip([1])), dim=-1))
            residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # pdb.set_trace()
        if self.inptarget in ['MagIRM', 'MagPSM', 'MagSMM']:
            output = F.sigmoid(self.output_layer(hidden_states))
        if self.inptarget in ['MagMag']:
            output = F.relu(self.output_layer(hidden_states))
        if self.inptarget in ['MagcIRM', 'ComplexcIRM']:
            output = self.output_layer(hidden_states)

        return output[seq_mask], lengths
    
class BiMamba_V2(Mamba_):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layer: int,
        inptarget: str,
        ssm_cfg = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg = None,
        fused_add_norm = True,
        residual_in_fp32 =True,
        device=None,
        dtype=None,
        fuse_mode=None,
        type=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_dim,
                         d_model,
                         n_layer,
                         inptarget,
                         ssm_cfg,
                         norm_epsilon,
                         rms_norm,
                         initializer_cfg,
                         fused_add_norm,
                         residual_in_fp32,
                         device,
                         dtype,
                         )
        # assert n_layer%2 == 0
        self.fuse_mode = fuse_mode
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=True,
                    fuse_mode=fuse_mode,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )
        
        
    def forward(self, x, lengths, inference_params=None):
        seq_mask = _sequence_mask(lengths)
        hidden_states = self.input_layer(x)
        residual = None
        for i in range(len(self.layers)//2):
            hidden_states_f, residual_f = self.layers[i*2](
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states_b, residual_b = self.layers[i*2+1](
                hidden_states.flip([1]), None if residual == None else residual.flip([1]), inference_params=inference_params
            )
            if self.fuse_mode == 'Add':
                # hidden_states = self.fuse_layers[i](hidden_states_f + hidden_states_b.flip([1]))
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
            if self.fuse_mode == 'Concat':
                hidden_states = self.fuse_layers[i](torch.cat((hidden_states_f, hidden_states_b.flip([1])), dim=-1))
            residual = residual_f + residual_b.flip([1])

        # for layer in self.layers:
        #     hidden_states, residual = layer(
        #         hidden_states, residual, inference_params=inference_params
        #     )
        # pdb.set_trace()

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # pdb.set_trace()
        if self.inptarget in ['MagIRM', 'MagPSM', 'MagSMM']:
            output = F.sigmoid(self.output_layer(hidden_states))
        if self.inptarget in ['MagMag']:
            output = F.relu(self.output_layer(hidden_states))
        if self.inptarget in ['MagcIRM', 'ComplexcIRM']:
            output = self.output_layer(hidden_states)

        return output[seq_mask], lengths
    