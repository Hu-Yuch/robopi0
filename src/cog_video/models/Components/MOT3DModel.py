from diffusers import CogVideoXTransformer3DModel
import torch
from torch import nn
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from typing import Any, Dict, Optional, Tuple, Union
import torch.nn.functional as F
from .Action_agent import State_expert, ActionEncoder
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.attention import Attention, FeedForward

from .module import CogVideoActionLayerNorm, SinusoidalPosEmb, CogVideoXPatchEmbed
#from models.Components.Attention import

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


            
def prepare_mask(mode_list: list, text_seq_length: int, video_seq_length: int, action_seq_length: int):
    bs = len(mode_list)
    seq_length = text_seq_length + video_seq_length + action_seq_length
    attention_mask = torch.ones(bs,seq_length,seq_length)
    for i in range(bs):
        if mode_list[i] =='a2v':
            attention_mask[i,text_seq_length:text_seq_length+video_seq_length, :text_seq_length] = 0
            attention_mask[i,text_seq_length+video_seq_length:, :text_seq_length] = 0
            attention_mask = attention_mask.bool()
        elif mode_list[i] =='t2v':
            attention_mask[i,:text_seq_length+video_seq_length, text_seq_length+video_seq_length:] = 0
            attention_mask = attention_mask.bool()
        elif mode_list[i] =='iv2a':
            attention_mask[i,:text_seq_length+video_seq_length, text_seq_length+video_seq_length:] = 0
            attention_mask = attention_mask.bool()
        #elif mode_list[i] == 'm2v':
        #    attention_mask[i,:text_seq_length+video_seq_length, text_seq_length+video_seq_length:] = 0
        #    attention_mask = attention_mask.bool()
        else:
            attention_mask = attention_mask.bool()
    return attention_mask


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_action_emb_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        add_action: bool = False,
        action_dim: int = 512,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if add_action:
            self.norm1_action = CogVideoActionLayerNorm(time_action_emb_dim, action_dim, norm_elementwise_affine, norm_eps, bias=True)
            # TODO: replace with True MOT, need to modify the Attention module
            self.attn1_action = Attention(
            query_dim=action_dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
            )  

            self.norm2_action = CogVideoActionLayerNorm(time_action_emb_dim, action_dim, norm_elementwise_affine, norm_eps, bias=True)
            self.ff_action = FeedForward(
                action_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        t_action_emb: torch.Tensor,
        action_hidden_states: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1) if encoder_hidden_states is not None else 0

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
                hidden_states, encoder_hidden_states, temb
        )

        if action_hidden_states is not None:
            norm_action_hidden_states, gate_msa_a = self.norm1_action(action_hidden_states, t_action_emb)
        else:
            norm_action_hidden_states = None

        # attention
        attn_hidden_states, attn_encoder_hidden_states, attn_action_hidden_states = self.MixtureAttnProcessor(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attn_mask,
            action_hidden_states=norm_action_hidden_states,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        if attn_encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        else:
            encoder_hidden_states = None

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )
        # feed-forward
        if norm_encoder_hidden_states is not None:
            norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        else:
            norm_hidden_states = norm_hidden_states
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        else:
            encoder_hidden_states = None

        # action process
        
        if action_hidden_states is not None:
            action_hidden_states = action_hidden_states + gate_msa_a * attn_action_hidden_states
            norm_action_hidden_states, gate_ff_action = self.norm2_action(action_hidden_states, t_action_emb)
            ff_output_action = self.ff_action(norm_action_hidden_states)
            action_hidden_states = action_hidden_states + gate_ff_action * ff_output_action


        return hidden_states, encoder_hidden_states, action_hidden_states
    
    def MixtureAttnProcessor(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1) if encoder_hidden_states is not None else 0
        img_seq_length = hidden_states.size(1)
        action_seq_length = action_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape
        sequence_length = sequence_length + action_seq_length

        if attention_mask is not None:
            attention_mask = self.attn1.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.attn1.heads, -1, attention_mask.shape[-1])

        query = self.attn1.to_q(hidden_states)
        key = self.attn1.to_k(hidden_states)
        value = self.attn1.to_v(hidden_states)

        action_query = self.attn1_action.to_q(action_hidden_states)
        action_key = self.attn1_action.to_k(action_hidden_states)
        action_value = self.attn1_action.to_v(action_hidden_states)

        query = torch.cat([query, action_query], dim=1)
        key = torch.cat([key, action_key], dim=1)
        value = torch.cat([value, action_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn1.heads

        query = query.view(batch_size, -1, self.attn1.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn1.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn1.heads, head_dim).transpose(1, 2)

        if self.attn1.norm_q is not None:
            query = self.attn1.norm_q(query)
        if self.attn1.norm_k is not None:
            key = self.attn1.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:text_seq_length+img_seq_length] = apply_rotary_emb(query[:, :, text_seq_length: text_seq_length+img_seq_length], image_rotary_emb)
            if not self.attn1.is_cross_attention:
                key[:, :, text_seq_length:text_seq_length+img_seq_length] = apply_rotary_emb(key[:, :, text_seq_length: text_seq_length+img_seq_length], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.attn1.heads * head_dim)
         
        if text_seq_length > 0:
            encoder_hidden_states, hidden_states, action_states = hidden_states.split(
                [text_seq_length, img_seq_length, action_seq_length], dim=1
            )
            # linear proj
            encoder_hidden_states = self.attn1.to_out[0](encoder_hidden_states)
            # dropout
            encoder_hidden_states = self.attn1.to_out[1](encoder_hidden_states)
            # linear proj
            hidden_states = self.attn1.to_out[0](hidden_states)
            # dropout
            hidden_states = self.attn1.to_out[1](hidden_states)
            # linear proj
            action_states = self.attn1_action.to_out[0](action_states)
            # dropout
            action_states = self.attn1_action.to_out[1](action_states)

            return hidden_states, encoder_hidden_states, action_states
        
        else:
            hidden_states, action_states = hidden_states.split(
                [img_seq_length, action_seq_length], dim=1
            )

            hidden_states = self.attn1.to_out[0](hidden_states)
            # dropout
            hidden_states = self.attn1.to_out[1](hidden_states)
            # linear proj
            action_states = self.attn1_action.to_out[0](action_states)
            # dropout
            action_states = self.attn1_action.to_out[1](action_states)

             return hidden_states, None, action_states

class CogVideoXMOT3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]


    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 32,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        add_action: bool = False,
        action_dim: int = 0,
        action_hidden_size: int = 0,
        action_lens: int = 0,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        print("init with custom CogVideoXTransformer3DModel with action")

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )
        self.add_action = add_action
        self.action_dim = action_dim
        self.action_hidden_size = action_hidden_size
        self.action_lens = action_lens
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # if add_action:
        #     # input dim: (B,a_num,a_dim)
        #     self.action_encoder = ActionEncoder(
        #         self.action_dim*2,
        #         self.action_hidden_size,
        #         self.action_lens,
        #         time_cond=False,
        #         time_dim = time_embed_dim,
        #     )
        #     self.action_decoder = nn.Linear(
        #         self.action_hidden_size,
        #         self.action_dim,
        #     )

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        if add_action:
            self.action_time_embedding = SinusoidalPosEmb(
                    self.action_hidden_size, 100.0
                )

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    time_action_emb_dim=512,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    add_action=add_action,
                    action_dim=action_hidden_size,
                    #action_lens=action_lens,
                    #a_expert_dim=a_expert_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)
        # if add_action:
        #     self.norm_final_action = nn.LayerNorm(action_hidden_size, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        # if add_action:
        #     self.norm_out_action = AdaLayerNorm(
        #         embedding_dim=time_embed_dim,
        #         output_dim=2 * action_hidden_size,
        #         norm_elementwise_affine=norm_elementwise_affine,
        #         norm_eps=norm_eps,
        #         chunk_dim=1,
        #     )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False

    def init_action_modules(self):
        pass

    #def _set_gradient_checkpointing(self, module, value=False):
    #    self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        action_timestep: Union[int, float, torch.LongTensor],
        action_states: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        action_timesteps = action_timestep
        t_action_emb = self.action_time_embedding(action_timesteps).to(dtype=hidden_states.dtype)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        # print("hidden_states.shape", hidden_states.shape)
        # print("encoder_hidden_states.shape", encoder_hidden_states.shape)
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # print("hidden_states.shape", hidden_states.shape)
        if not encoder_hidden_states is None:
            text_seq_length = encoder_hidden_states.shape[1]
        else:
            text_seq_length = 0
        encoder_hidden_states = None
        hidden_states = hidden_states[:, text_seq_length:]
        video_seq_length = hidden_states.shape[1]
        action_seq_length = action_states.shape[1]
        action_hidden_states = action_states

        #if action_states is not None:
        #    if self.action_encoder.time_cond:
        #       action_states = self.action_encoder(action_states)
        #    else:
        #       action_states = self.action_encoder(action_states, t_action_emb)
        #    action_seq_length = action_states.shape[1]
        #else:
        #    action_states = None

        attention_mask = prepare_mask(attention_kwargs['attention_mask'], text_seq_length, video_seq_length, action_seq_length)
        attention_mask = attention_mask.to(hidden_states.device)
        attn_mask = attention_mask

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, _ , action_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    t_action_emb,
                    action_hidden_states,
                    image_rotary_emb,
                    attn_mask,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, _, action_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    t_action_emb = t_action_emb,
                    action_hidden_states=action_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    attn_mask=attn_mask,
                )

        # if not self.config.use_rotary_positional_embeddings:
        #     # CogVideoX-2B
        #     hidden_states = self.norm_final(hidden_states)
        # else:
        #     # CogVideoX-5B
        #     hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        #     hidden_states = self.norm_final(hidden_states)
        #     hidden_states = hidden_states[:, text_seq_length:]
        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        # if action_hidden_states is not None:
        #     action_hidden_states = self.norm_final_action(action_hidden_states)
        #     action_hidden_states = self.norm_out_action(action_hidden_states,temb=emb)
        #     action_hidden_states = self.action_decoder(action_hidden_states)


        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        # print("forward success output/action", output.shape, action_hidden_states.shape if action_hidden_states is not None else None)
        if not return_dict:
            return (output)
        return Transformer2DModelOutput(sample=output)




        