import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from .module import CogVideoActionBlock

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_utils import ModelMixin

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.cache_utils import CacheMixin

@maybe_allow_in_graph
class State_expert(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(self, 
                 action_dim,
                 time_dim,
                 num_layers,
                 action_lens,
                 action_hidden_size: int = 512,
                 num_attention_heads: int = 48,
                 attention_head_dim: int = 64,
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.action_lens = action_lens
        self.action_hidden_size = action_hidden_size
        self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                self.action_lens,
                time_cond=False,
                time_dim = time_dim,
            )
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoActionBlock(
                    dim=action_hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_dim,
                    dropout=0.0,
                    attention_bias=True,
                    norm_elementwise_affine=True,
                )
                for _ in range(num_layers)
            ]
        )

@maybe_allow_in_graph
class ActionEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(self, action_dim: int, width: int, action_lens: int, time_dim: int, time_cond: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.width = width
        self.linear_1 = nn.Linear(action_dim, width)
        if time_cond:
            self.linear_2 = nn.Linear(width + time_dim, width)
        else:
            self.linear_2 = nn.Linear(width, width)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(width, width)
        self.time_cond = time_cond
        self.action_lens = action_lens

        pos_embedding = self._get_positional_embeddings(action_lens)
        self.register_buffer("pos_embedding", pos_embedding, persistent=True)

    def _get_positional_embeddings(
        self, action_lens: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        pos = torch.ones(1)*action_lens
        pos_embedding = get_1d_sincos_pos_embed_from_grid(
            self.width,
            pos,
            output_type="pt",
        )

        return pos_embedding

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Width]
        emb = self.linear_1(action)
        emb = emb + self.pos_embedding
        if self.time_cond:
            # repeat time embedding for seq_len
            # [Batch_Size, Seq_Len, Width]
            time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, output_type="np"):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`torch.Tensor`): 1D tensor of positions with shape `(M,)`

    Returns:
        `torch.Tensor`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    if output_type == "np":
        return get_1d_sincos_pos_embed_from_grid_np(embed_dim=embed_dim, pos=pos)
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_1d_sincos_pos_embed_from_grid_np(embed_dim, pos):
    """
    This function generates 1D positional embeddings from a grid.

    Args:
        embed_dim (`int`): The embedding dimension `D`
        pos (`numpy.ndarray`): 1D tensor of positions with shape `(M,)`

    Returns:
        `numpy.ndarray`: Sinusoidal positional embeddings of shape `(M, D)`.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb