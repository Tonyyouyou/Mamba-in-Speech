# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging

import torch

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer_mamba import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.bimamba_simple import BiMamba
from mamba_ssm.modules.outer_bimamba import Bimamba_outer

class Encoder(torch.nn.Module):
    """Conformer encoder module.

    Args:
        idim (int): Input dimension.
        d_model (int): Dimension of mamba.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Encoder positional encoding layer type.
        mamba_type (str): Encoder mamba layer type.
        activation_type (str): Encoder activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
        stochastic_depth_rate (float): Maximum probability to skip the encoder layer.
        intermediate_layers (Union[List[int], None]): indices of intermediate CTC layer.
            indices start from 1.
            if not None, intermediate outputs are returned (which changes return type
            signature.)

    """

    def __init__(
        self,
        idim,
        d_model=256,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        mamba_type="mamba",
        activation_type="swish",
        use_cnn_module=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        stochastic_depth_rate=0.0,
        intermediate_layers=None,
        ctc_softmax=None,
        conditioning_layer_dim=None,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        Amatrix_type=None,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert mamba_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            pos_enc_class = LegacyRelPositionalEncoding
            assert mamba_type == "legacy_rel_selfattn"
        elif pos_enc_layer_type == "mamba":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "bimamba":
            pos_enc_class = PositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, d_model),
                torch.nn.LayerNorm(d_model),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(d_model, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                d_model,
                dropout_rate,
                pos_enc_class(d_model, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, d_model)
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, d_model, padding_idx=padding_idx),
                pos_enc_class(d_model, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(d_model, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(d_model, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # mamba module definition
        if mamba_type == "mamba":
            encoder_selfattn_layer = Mamba
            encoder_selfattn_layer_args = (
                d_model,
                d_state,
                d_conv,
                expand,
                dt_rank,
                dt_min,
                dt_max,
                dt_init,
                dt_scale,
                dt_init_floor,
                conv_bias,
                bias,
                use_fast_path,
                layer_idx,
                device,
                dtype,
            )
        elif mamba_type == "bimamba":
            encoder_selfattn_layer = BiMamba
            encoder_selfattn_layer_args = (
                d_model,
                d_state,
                d_conv,
                expand,
                dt_rank,
                dt_min,
                dt_max,
                dt_init,
                dt_scale,
                dt_init_floor,
                conv_bias,
                bias,
                use_fast_path,
                layer_idx,
                device,
                dtype,
            )
        elif mamba_type == 'bimamba_outer':
            print('using bimamba outer')
            encoder_selfattn_layer = Bimamba_outer
            encoder_selfattn_layer_args = (
                d_model,
                d_state,
                d_conv,
                expand,
                device,
                dtype
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + mamba_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                d_model,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                d_model,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                d_model,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (d_model, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                d_model,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                mamba_type,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(d_model)

        self.intermediate_layers = intermediate_layers
        self.use_conditioning = True if ctc_softmax is not None else False
        if self.use_conditioning:
            self.ctc_softmax = ctc_softmax
            self.conditioning_layer = torch.nn.Linear(
                conditioning_layer_dim, d_model
            )

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, 1, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, d_model).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    # intermediate branches also require normalization.
                    encoder_output = xs
                    if isinstance(encoder_output, tuple):
                        encoder_output = encoder_output[0]

                    if self.normalize_before:
                        encoder_output = self.after_norm(encoder_output)

                    intermediate_outputs.append(encoder_output)

                    if self.use_conditioning:
                        intermediate_result = self.ctc_softmax(encoder_output)

                        if isinstance(xs, tuple):
                            x, pos_emb = xs[0], xs[1]
                            x = x + self.conditioning_layer(intermediate_result)
                            xs = (x, pos_emb)
                        else:
                            xs = xs + self.conditioning_layer(intermediate_result)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks
