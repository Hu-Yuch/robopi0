"""
Wrapper around the joint model (mixtures). Siglip from PaliGemma, action-time encoder, proprio encoder, action decoder. Flow matching training

Generates causal masking for the mixtures

Potentially customized to add/remove mixtures, e.g., remove proprio or add another vision module

"""

import logging
from typing import Optional, Tuple

import hydra
import torch
from torch import nn

from src.model.kv_cache import KVCache
from src.model.vla.modules import (
    ActionEncoder,
    SinusoidalPosEmb,
)
from src.utils.decorator import NoSyncBase
from src.utils.monitor import log_execution_time

from torch.linalg import inv
import torchvision.transforms.functional as TF
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True

SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
ZOE_MEAN, ZOE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

from transformers import (
    AutoTokenizer,
    ZoeDepthConfig,
    ZoeDepthForDepthEstimation,
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainingArguments,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
"""
Launcher for all experiments.

"""
import einops
import logging
import math
import os
import random
import sys
from src.model.vla.processing import VLAProcessor
import hydra
import numpy as np
import pretty_errors
import torch
import open3d as o3d
from PIL import Image
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from src.agent.dataset_real import Dataset_Real
from torch.linalg import inv
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import transformers
from transformers import (
    AutoTokenizer,
    ZoeDepthConfig,
    ZoeDepthForDepthEstimation,
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainingArguments,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from transformers import AutoTokenizer
from src.model.vla.processing import VLAProcessor
from src.cog_video.datasets import BridgeDatasetWithResize

log = logging.getLogger(__name__)

def process_zoe(pixel_values, pad_mode="reflect", output_size=(384, 512)):
    """https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/zoedepth/image_processing_zoedepth.py"""
    # h, w = images.shape[-2:]
    # pad
    ph, pw = 31, 31  # int((h / 2)**0.5 * 3), int((w / 2)**0.5 * 3) # 32, 31
    images = F.pad(pixel_values, (pw, pw, ph, ph), mode=pad_mode)
    # resize
    size = (384, 384)  # get_resize_output_image_size
    images = F.interpolate(images, size=size, mode="bicubic", align_corners=True)
    # zoe: padding -> resize -> nomalize. we follow `nomalize -> padding -> resize` from siglip
    images = TF.normalize(images, mean=ZOE_MEAN, std=ZOE_STD)
    return images, ph, pw

class Ego3DPositionEmbeddingMLP(nn.Module):
    """Absolute pos embedding, learned.
    https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/models/nerf.py#L4
    """

    def __init__(self, in_channels=3, num_pos_feats=768, n_freqs=8, logscale=True):
        super(Ego3DPositionEmbeddingMLP, self).__init__()
        self.n_freqs = n_freqs
        self.freq_out_channels = in_channels * (2 * n_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)
        
        center = torch.tensor([0., 0., 2.]).repeat(in_channels // 3)
        self.register_buffer("freq_bands", freq_bands, persistent=False)
        self.register_buffer("center", center, persistent=False)

        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """init with small weights to maintain stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.01)

    @torch.no_grad()
    def frequency_encoding(self, xyz):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        x \in [-2, 2]
        y \in [-2, 2]
        z \in [0., 4]
        Inputs:
            x: (b n m)
        Outputs:
            out: (b n o)
        """
        xyz_n = ((xyz - self.center) / 2.0).to(self.freq_bands.dtype)
        xyz_feq = xyz_n.unsqueeze(-1) * self.freq_bands  # (b n m 1)
        sin_xyz, cos_xyz = torch.sin(xyz_feq), torch.cos(xyz_feq)  # (b n m nf)
        encoding = torch.cat([xyz_n.unsqueeze(-1), sin_xyz, cos_xyz], -1).reshape(*xyz.shape[:2], -1)
        return encoding

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, N, F)."""
        freq_encoding = self.frequency_encoding(xyz)
        position_embedding = self.position_embedding_head(freq_encoding)
        return position_embedding

class PiZero3D(nn.Module, NoSyncBase):
    @log_execution_time(log)
    def __init__(self, cfg, use_ddp: bool = False):
        super().__init__()
        self.cfg = cfg
        self.use_ddp = use_ddp  # used in NoSyncBase
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.use_lm_head = cfg.get("use_lm_head", False)

        self.max_image_text_tokens = cfg.max_image_text_tokens
        self.num_proprio_tokens = cfg.cond_steps
        self.num_action_tokens = cfg.horizon_steps
        self.total_num_tokens = (
            self.max_image_text_tokens
            + self.num_proprio_tokens
            + self.num_action_tokens
        )

        self.image_text_hidden_size = cfg.mixture.vlm.hidden_size
        self.proprio_hidden_size = cfg.mixture.proprio.hidden_size
        self.action_hidden_size = cfg.mixture.action.hidden_size

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.proprio_dim = cfg.proprio_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.flow_sig_min = cfg.get("flow_sig_min", 0.001)

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision
        self.vision_tower = hydra.utils.instantiate(cfg.vision)
        self.multi_modal_projector = hydra.utils.instantiate(cfg.vision_projector)

        # Mixtures
        self.joint_model = hydra.utils.instantiate(cfg.joint)

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        if cfg.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=False,
            )
            self.time_embedding = SinusoidalPosEmb(
                cfg.time_hidden_size, cfg.time_max_period
            )
        else:  # matching pi0
            self.action_encoder = ActionEncoder(
                self.action_dim,
                self.action_hidden_size,
                time_cond=True,
            )
            self.time_embedding = SinusoidalPosEmb(
                self.action_hidden_size, cfg.time_max_period
            )
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = nn.Linear(
            self.action_hidden_size,
            self.action_dim,
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

        torch_dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32

        self.camera = np.load('/cephfs/shared/hyc/data/calvin/task_ABC_D/camera.npy', allow_pickle=True).item()

        # vision_zoe_config = ZoeDepthConfig.from_pretrained(
        #     cfg.vision_zoe_path, torch_dtype=torch_dtype, local_files_only=True
        # )
        # self.vision_zoe_model = ZoeDepthForDepthEstimation.from_pretrained(  # zoe does not support Flash Attention 2.0 yet.
        #     cfg.vision_zoe_path,
        #     config=vision_zoe_config,
        #     torch_dtype=torch_dtype,
        #     local_files_only=True,
        # )
        # self.vision_zoe_model = self.vision_zoe_model.eval()

        # for param in self.vision_zoe_model.parameters():
        #     param.requires_grad = False
        self.use_3D = cfg.get("use_3d", True)
        if self.use_3D:
            self.position_embedding_3d = Ego3DPositionEmbeddingMLP(
                cfg.ego3d_patch_reso**2 * 3, num_pos_feats=cfg.vision.config.hidden_size, n_freqs=cfg.n_freqs
            )
        # # register buffer
        self.patch_size, self.reso, self.image_size = cfg.vision.config.patch_size, cfg.ego3d_patch_reso, cfg.vision.config.image_size
        # y, x = torch.meshgrid(torch.arange(0, image_size, patch_size // reso), torch.arange(0, image_size, patch_size // reso), indexing="ij")  # (h//sp w//sp)
        # y, x = y + patch_size / reso / 2, x + patch_size / reso / 2
        # uv_h = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)  # (3 hw)
        # self.register_buffer("uv_h", uv_h, persistent=False)

    def deproject(self, depth_img, device, homogeneous=False, sanity_check=False):
        """
        Deprojects a pixel point to 3D coordinates
        Args
            point: tuple (u, v); pixel coordinates of point to deproject
            depth_img: np.array; depth image used as reference to generate 3D coordinates
            homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                        else returns the world coordinates (x, y, z) position
        Output
            (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
        """
        device = device
        b, c, h, w = depth_img.shape
        new_h, new_w = 16*self.reso, 16*self.reso

        new_depth = F.interpolate(depth_img, size=(new_h, new_w), mode="area")

        u, v = np.meshgrid(np.arange(new_h), np.arange(new_w))
        u, v = u.ravel(), v.ravel()

        # Unproject to world coordinates
        T_world_cam = np.linalg.inv(np.array(self.camera['static_cam_viewMatrix']).reshape((4, 4)).T)
        T_world_cam = torch.from_numpy(T_world_cam).to(device).unsqueeze(0).repeat(b,1,1)
        z = new_depth[:, 0, v, u]
        u = torch.from_numpy(u).to(device).repeat(b,1)
        v = torch.from_numpy(v).to(device).repeat(b,1)
        foc = torch.tensor(self.camera['static_cam_height']).to(device) / (2 * torch.tan(torch.tensor(self.camera['static_cam_fov']).to(device) / 2))
        x = (u*h/new_h - torch.tensor(self.camera['static_cam_width']).to(device) // 2) * z / foc
        y = -(v*w/new_w - torch.tensor(self.camera['static_cam_height']).to(device) // 2) * z / foc
        z = -z
        ones = torch.ones_like(z).to(device)

        cam_pos = torch.stack([x, y, z, ones], axis=1)
        T_world_cam = T_world_cam.to(cam_pos.dtype)
        world_pos = T_world_cam @ cam_pos

        if not homogeneous:
            world_pos = world_pos[:,:3]

        return world_pos

    def backproject_patch(self, K: torch.Tensor, depth: torch.Tensor, patch_size=14, reso=2) -> torch.Tensor:
        """
        Backproject depth map to 3D points in camera coordinate.
        Args:
            K: camera intrinsic matrix (b 3 3)
            depth: depth map (b 1 h w)
            patch_size: patch size for siglip
            reso: reso^2 -> sample points in each patch
        patch sz = 14  ......          
        ┌────────┬────────┐       
        │ ─    ─ │ ─    ─ │       
        │ points │        ├─ ─ ─ 
        │ ─    ─ │ ─    ─ │       
        ├────────┼────────┤       
        │ ─    ─ │ ─    ─ │       
        │        │        │       
        │ ─    ─ │ ─    ─ │       
        └────────┴────────┘       
        reso=2───►points=4
            │                    
            │                        
        """
        b, c, h, w = depth.shape
        hp, wp = h // patch_size, w // patch_size
        sub_hp = sub_wp = reso
        patch_depth = F.interpolate(depth, size=(hp * reso, wp * reso), mode="area").reshape(b, c, -1)
        p_cam = (inv(K.float()) @ self.uv_h.float()) * patch_depth  # (b 3 3) @ (3 hw) -> (b 3 hw) * (b 1 hw) -> (b 3 hw)
        patch_p_cam = p_cam.reshape(b, 3, hp, sub_hp, wp, sub_wp).permute(0, 2, 4, 3, 5, 1).reshape(b, hp * wp, -1)
        return patch_p_cam

    @property
    def action_expert_parameters(self):
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.joint_model.mixtures["action"].parameters())
        )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
        if self.use_3D:
            return (
                list(self.vision_tower.parameters())
                + list(self.multi_modal_projector.parameters())
                + list(self.position_embedding_3d.parameters())
                + self.trainable_gemma_parameters
            )
        else:
            return (
                list(self.vision_tower.parameters())
                + list(self.multi_modal_projector.parameters())
                + self.trainable_gemma_parameters
            )

    @property
    def lora_trainable_vlm_parameters(self):
        params = []
        for name, param in self.vision_tower.named_parameters():
            if "lora_" in name:
                params.append(param)
        for name, param in self.multi_modal_projector.named_parameters():
            if "lora_" in name:
                params.append(param)
        params.extend(self.trainable_lora_gemma_parameters)
        return params

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    @log_execution_time(log)
    def load_pretrained_weights(self):
        """vision, projector, lm from paligemma"""
        import glob
        import os

        from safetensors import safe_open

        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k and 'spatial_embed_tokens' not in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        log.info("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "vision_tower" in k:
                new_key = k.replace("vision_tower.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        log.info("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        log.info("Loaded pre-trained weights for projector")

        # load lm --- do not change any lora weights
        joint_model_state_dict = self.joint_model.state_dict()
        lora_keys = []
        for key in (
            joint_model_state_dict.keys()
        ):  # avoid RuntimeError: OrderedDict mutated during iteration
            if "lora_" in key:
                lora_keys.append(key)
        for key in lora_keys:
            del joint_model_state_dict[key]
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "mixtures.vlm.")
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        log.info("Loaded pre-trained weights for lm part of the joint model")

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        """no need to train vlm parameters after attention of last layer"""
        last_hidden_layer_index = self.joint_model.num_hidden_layers - 1
        if (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_proj" in name
            or f"{last_hidden_layer_index}.self_attn.v_proj" in name
        ):  # final norm is not initialized
            return True
        return False

    def freeze_non_lora_weights_in_vlm(self):
        """Keep all bias frozen"""
        for name, param in self.vision_tower.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in vision tower")

        for name, param in self.multi_modal_projector.named_parameters():
            param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in projector")

        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if not self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False
        log.info("Froze non-lora weights in lm part of the joint model")

    def freeze_unused_weights(self):
        """text embedding and part of last layer of vlm, including lora"""
        self.embed_tokens.weight.requires_grad = False
        for name, param in self.joint_model.mixtures["vlm"].named_parameters():
            if self._check_gemma_unused_parameter_by_name(name):
                param.requires_grad = False

    def freeze_all_weights(self):
        for _, param in self.named_parameters():
            param.requires_grad = False

    def tie_action_proprio_weights(self):
        """technically more than just tying weights"""
        self.joint_model.mixtures["proprio"] = self.joint_model.mixtures["action"]

    def build_text_cache(self):
        return KVCache()

    # ---------- Input preparation ----------#

    def build_causal_mask_and_position_ids(
        self, attention_mask: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.FloatTensor]:
        """
        block attention --- padding for unused text tokens

                 img/text img/text img/text (padding) proprio action action
        img/text    x        x        x
        img/text    x        x        x
        img/text    x        x        x
        (padding)
        proprio     x        x        x                 x
        action      x        x        x                 x       x      x
        action      x        x        x                 x       x      x
        """
        bsz = attention_mask.size(0)
        proprio_start = self.max_image_text_tokens
        proprio_end = self.max_image_text_tokens + self.num_proprio_tokens
        action_start = proprio_end
        image_text_token_cnts = torch.sum(attention_mask, dim=1)
        causal_mask = torch.full(
            (bsz, self.total_num_tokens, self.total_num_tokens),
            torch.finfo(dtype).min,
            dtype=dtype,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, cnt in enumerate(image_text_token_cnts):
            causal_mask[idx, :cnt, :cnt] = 0  # image/text attend to itself
            causal_mask[idx, proprio_start:, :cnt] = (
                0  # proprio/action attend to image/text
            )
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
            0  # proprio attend to itself
        )
        causal_mask[:, action_start:, proprio_start:] = (
            0  # action attend to itself and proprio
        )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks --- start at 1
        vlm_position_ids = torch.arange(1, self.max_image_text_tokens + 1).repeat(
            bsz, 1
        )
        proprio_position_ids = torch.arange(1, self.num_proprio_tokens + 1).repeat(
            bsz, 1
        )
        action_position_ids = torch.arange(
            self.num_proprio_tokens + 1,
            self.num_proprio_tokens + self.num_action_tokens + 1,
        ).repeat(bsz, 1)
        # since proprio and action share the same mixture weights, makes sense to use [1 (proprio), 2 (action), 3 (action), ...] instead of [1 (proprio), 1 (action), 2 (action), ...]
        return causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids

    def split_full_mask_into_submasks(
        self, causal_mask: torch.FloatTensor, 
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        proprio_action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return image_text_proprio_mask, action_mask, 

    def build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        dtype, device = attention_mask.dtype, attention_mask.device

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask any token, because we're in the prefill phase
            # assume no padding
            causal_mask = torch.full((bsz, q_len, q_len), 0, dtype=dtype, device=device)
        else:
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # this only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), 0, dtype=dtype, device=device
            )

        # add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # use the last location
            position_ids = attention_mask.cumsum(-1)[:, -1:]
        else:
            # create position_ids based on the size of the attention_mask
            # for padded tokens, use number 1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_(
                (attention_mask == 0), 1
            )
        return causal_mask, position_ids

    # ---------- Inference ----------#

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        depth: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)

        # image features from siglip and projector
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        selected_image_feature = self.vision_tower(pixel_values)

        # 预处理像素值
        pixel_values = (pixel_values + 1) / 2
        
        # 使用 torch.no_grad() 包装 ZoeDepth 模型的前向传播
        # with torch.no_grad():
        #     # 处理 ZoeDepth 输入
        #     zoe_pixel_values, ph, pw = process_zoe(pixel_values, pad_mode="reflect")
            
        #     # 获取深度图
        #     depth_output = self.vision_zoe_model(pixel_values=zoe_pixel_values)
        #     depth = depth_output.predicted_depth
            
        #     # 调整深度图大小
        #     pvh, pvw = pixel_values.shape[-2:]
        #     depth = F.interpolate(
        #         depth.unsqueeze(1),
        #         size=(pvh+2*ph, pvw+2*pw),
        #         mode="bicubic",
        #         align_corners=True,
        #     )[..., ph:-ph, pw:-pw]
            
        #     # 使用张量而不是标量
        #     patch_size = torch.tensor(self.cfg.vision.config.patch_size, device=depth.device)
        #     reso = torch.tensor(self.cfg.ego3d_patch_reso, device=depth.device)
            
        #     # 生成点云
        #     xyz = self.backproject_patch(
        #         intrinsic, depth, patch_size=patch_size, reso=reso
        #     )
        #print('depth_shape:',depth.shape)
        #xyz = [self.deproject(depth[i:i+1], pixel_values.device) for i in range(input_ids.shape[0])]
        #xyz = torch.cat(xyz, dim=0)
        if self.use_3D:
            xyz = self.deproject(depth, pixel_values.device)
            sub_hp = sub_wp = self.reso
            b = depth.shape[0]
            hp, wp = 16, 16
            xyz = xyz.reshape(b, 3, hp, sub_hp, wp, sub_wp).permute(0, 2, 4, 3, 5, 1).reshape(b, hp * wp, -1)
            
            # 3D 位置编码
            pos_embed_3d = self.position_embedding_3d(xyz)
            selected_image_feature = selected_image_feature + pos_embed_3d
        
        image_features = self.multi_modal_projector(selected_image_feature)

        # normalize the image features
        _, _, embed_dim = image_features.shape
        bsz, seq_len = input_ids.shape
        scaled_image_features = image_features / (self.image_text_hidden_size**0.5)

        # put embedding together - image, text, padding
        final_embedding = torch.full(
            (bsz, seq_len, embed_dim), self.pad_token_id, dtype=dtype, device=device
        )

        # [Batch_Size, Seq_Len]
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.image_token_index
        final_embedding[text_mask] = inputs_embeds[text_mask]
        for i in range(bsz):
            image_indices = image_mask[i].nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_indices)
            final_embedding[i, image_indices] = scaled_image_features[
                i, :num_image_tokens
            ]
        return final_embedding

    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        depth: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # forward pass thru the vlm and proprio, cache the kv
        _, kv_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
            },
            kv_caches=kv_caches,
            return_caches=True,
        )

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- using kv caches of vlm and proprio
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=action_mask,
                position_ids_all={"action": action_position_ids},
                embeds_all={"action": action_embeds},
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="append_non_active",  # use caches from other mixtures, i.e., vlm and proprio
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_action_naive(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        kv_caches = self.joint_model.build_mixture_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth)

        # encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # sample pure action noise
        action = torch.randn(
            (bsz, self.horizon_steps, self.action_dim), device=device, dtype=dtype
        )

        # forward euler integration --- run vlm in each step, which is unnecessary
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=device, dtype=dtype)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            time_cond = self.time_embedding(t)
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            if self.action_expert_adaptive_mode:
                action_embeds = self.action_encoder(action)
            else:
                action_embeds = self.action_encoder(action, time_cond)
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all={
                    "vlm": vlm_position_ids,
                    "proprio": proprio_position_ids,
                    "action": action_position_ids,
                },
                embeds_all={
                    "vlm": inputs_embeds.clone(),  # clone needed due to modified in-place
                    "proprio": proprio_embeds.clone(),
                    "action": action_embeds,
                },
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode="no_append",  # no new tokens
            )["action"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    def infer_text(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        q_len = input_ids.size(1)

        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth)

        # build causal mask and position ids for text
        (
            causal_mask,
            position_ids,
        ) = self.build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache
        )

        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={"vlm": kv_cache},
            cache_mode="append",  # new tokens for the active mixture
            final_layer_post_attn_skip_names=[],  # do not skip vlm last layer
        )["vlm"]
        logits = self.lm_head(hidden_states)
        output = {
            "logits": logits,
        }
        if kv_cache is not None:
            output["kv_cache"] = kv_cache
        return output

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward_language(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        depth: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        labels: torch.LongTensor = None,
        loss_mask: torch.BoolTensor = None,  # 新增loss_mask参数
    ) -> torch.FloatTensor:
        """自回归训练PaLiGemma语言模型
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            pixel_values: 图像输入 [batch_size, 3, H, W] 
            depth: 深度图 [batch_size, 1, H, W]
            attention_mask: 注意力mask [batch_size, seq_len]
            labels: 目标token ids [batch_size, seq_len]
            loss_mask: 指定计算loss的位置 [batch_size, seq_len]，True表示计算该位置的loss
        Returns:
            loss: 语言模型损失
        """
        # 构建因果mask和位置编码
        q_len = input_ids.size(1)
        causal_mask, position_ids = self.build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache=None
        )

        # 获取文本和图像的嵌入
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth)

        # 前向传播获取隐状态
        hidden_states = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={"vlm": position_ids},
            embeds_all={"vlm": inputs_embeds},
            kv_caches={},  # 训练时不使用kv cache
            cache_mode="no_append",
            final_layer_post_attn_skip_names=[],  # 不跳过vlm最后一层
        )["vlm"]

        # 获取logits
        logits = self.lm_head(hidden_states)

        # 计算loss
        loss = None
        if labels is not None:
            # 将logits重塑为(batch_size * seq_len, vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            # 将labels重塑为(batch_size * seq_len)
            shift_labels = labels[..., 1:].contiguous()
            
            # 处理loss_mask
            if loss_mask is not None:
                # 移位loss_mask以匹配shift_logits和shift_labels
                shift_loss_mask = loss_mask[..., 1:].contiguous()
                
                # 计算交叉熵损失
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                # 首先计算每个位置的loss
                loss = loss_fct(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1)
                )
                
                # 应用mask，只保留需要计算loss的位置
                loss = loss.view(shift_labels.size())  # 恢复batch维度
                loss = loss * shift_loss_mask  # 应用mask
                
                # 计算平均loss，只考虑mask为True的位置
                num_valid = shift_loss_mask.sum()
                if num_valid > 0:  # 避免除零错误
                    loss = loss.sum() / num_valid
                else:
                    loss = loss.sum() * 0.0  # 如果没有有效位置，返回0
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1)
                )

        return loss if loss is not None else logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.ByteTensor,
        depth: torch.FloatTensor,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)
        # text tokens + image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values, depth)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        time_cond = self.time_embedding(t)
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        if self.action_expert_adaptive_mode:
            action_embeds = self.action_encoder(psi_t)
        else:
            action_embeds = self.action_encoder(psi_t, time_cond)
        action_embeds = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
                "action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            time_cond=time_cond,
            kv_caches={},  # no caching during training
        )["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)


class PiZero3DInference(PiZero3D):
    def __init__(self, cfg, use_ddp: bool = False):
        super().__init__(cfg)
        self.multistep = 0
        self.current_step = 0
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
        self.processor = VLAProcessor(
            self.tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
        self.dtype = None
        self.deviced = None

    def reset(self):
        self.current_step = 0

    def preprocess_batch(self, obs, goal, split_mask: bool, sample_fm_time: bool):
        images = obs["rgb_obs"]['rgb_static'].to(torch.uint8).to('cpu')
        images = images.squeeze(0)
        proprios = obs['robot_obs']
        # texts = [
        #     text.decode("utf-8") for text in batch["task"]["language_instruction"]
        # ]
        # print("images", images.shape)
        # print("propios", proprios.shape)
        # print("actions", actions.shape)
        texts = goal
        depth = obs["depth_obs"]['depth_static']
        #images = einops.rearrange(
        #    images, "B T H W C -> B (T C) H W"
        #)  # remove cond_steps dimension
        model_inputs = self.processor(text=texts, images=images)

        # build causal mask and position ids for action
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            self.build_causal_mask_and_position_ids(
                model_inputs["attention_mask"], self.dtype
            )
        )

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(self.dtype),
            "depth": depth.to(self.dtype),
            "vlm_position_ids": vlm_position_ids,
            "proprio_position_ids": proprio_position_ids,
            "action_position_ids": action_position_ids,
            "proprios": proprios.to(self.dtype),
        }
        if split_mask:
            image_text_proprio_mask, action_mask = (
                self.split_full_mask_into_submasks(causal_mask)
            )
            inputs["image_text_proprio_mask"] = image_text_proprio_mask
            inputs["action_mask"] = action_mask
        else:
            inputs["causal_mask"] = causal_mask

        # sample flow matching timesteps
        if sample_fm_time:
            inputs["t"] = self.sample_fm_time(len(texts)).to(self.dtype)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def step(self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        depth: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.current_step % self.multistep == 0:
            pred_action_seq = super().infer_action(
                input_ids,
                pixel_values,
                depth,
                image_text_proprio_mask,
                action_mask,
                vlm_position_ids,
                proprio_position_ids,
                action_position_ids,
                proprios,
            )

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.current_step]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.current_step += 1
        if self.current_step == self.multistep:
            self.current_step = 0

        return current_action

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        depth: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        action_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return super().infer_action(
            input_ids,
            pixel_values,
            depth,
            image_text_proprio_mask,
            action_mask,
            vlm_position_ids,
            proprio_position_ids,
            action_position_ids,
            proprios,
        )


if __name__ == "__main__":
    import argparse
    import time

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import AutoTokenizer

    from src.model.vla.processing import VLAProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--load_pretrained_weights", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--loss_only", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert not (args.text_only and args.loss_only)

    torch.manual_seed(args.seed)

    config = OmegaConf.load("config/train/bridge.yaml")
    if args.text_only:
        config.use_lm_head = True
        config.mixture.vlm.use_final_norm = True
    device = "cpu" if args.cpu else "cuda"
    model = PiZero(config)
    model.tie_action_proprio_weights()
    if args.load_pretrained_weights:
        model.load_pretrained_weights()
    dtype = torch.bfloat16 if args.use_bf16 else torch.float32
    model.to(device)
    model.to(dtype)
    model.eval()
    print(f"Using {device} and {dtype}...")

    # dummy image --- replace the first image with a real one
    bsz = 1 if args.text_only else 2
    dummy_images = torch.randint(
        0, 256, (bsz, 3, 224, 224), dtype=torch.uint8
    )  # not used if text_only
    real_image_path = "media/maniskill_pp.png"
    real_image = Image.open(real_image_path).convert("RGB")
    real_image_t = torch.as_tensor(
        np.array(real_image.resize((224, 224))).transpose(2, 0, 1)
    )
    dummy_images[0] = real_image_t

    # text and proprio
    dummy_texts = [
        "this image shows ",
        "this is a nice portrait of London because ",
    ][:bsz]
    dummy_proprio = torch.rand(bsz, config.cond_steps, config.action_dim)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )
    assert tokenizer.padding_side == "right"

    # processor
    num_image_tokens = config.vision.config.num_image_tokens
    processor = VLAProcessor(tokenizer, num_image_tokens, config.max_seq_len)

    # process image and text
    model_inputs = processor(text=dummy_texts, images=dummy_images)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"].to(dtype)

    # inference
    start_time = time.time()
    if args.text_only:  # no sampling
        kv_cache = model.build_text_cache()
        num_tokens_to_generate = 20
        print(f"Generating text of maximum {num_tokens_to_generate} tokens...")

        stop_token = processor.tokenizer.eos_token_id
        generated_tokens = []
        for _ in range(num_tokens_to_generate):
            with torch.inference_mode():
                outputs = model.infer_text(
                    input_ids=input_ids.to(device),
                    pixel_values=pixel_values.to(device),
                    attention_mask=attention_mask.to(device),
                    kv_cache=kv_cache,
                )
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)  # remove batch dimension
            generated_tokens.append(next_token)
            # stop if the stop token has been generated
            if next_token.item() == stop_token:
                break
            # only input the new token the next time since using cache
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=-1
            )
        generated_tokens = torch.cat(generated_tokens, dim=-1)
        decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("\n\n=========================")
        print("Image path:", real_image_path)
        print("Prompt:", dummy_texts[0])
        print("Generated text:", decoded)
    elif args.loss_only:
        dummy_actions = torch.randn(bsz, config.horizon_steps, config.action_dim)
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        t = torch.rand(bsz)
        with torch.inference_mode():
            loss = model(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(dtype).to(device),
                causal_mask=causal_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=dummy_proprio.to(dtype).to(device),
                actions=dummy_actions.to(dtype).to(device),
                t=t.to(dtype).to(device),
            )
        print("\n\n=========================")
        print("Loss:", loss)
    else:  # dummy action generation
        causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
            model.build_causal_mask_and_position_ids(attention_mask, dtype=dtype)
        )
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        with torch.inference_mode():
            actions = model.infer_action(
                input_ids=input_ids.to(device),
                pixel_values=pixel_values.to(dtype).to(device),
                image_text_proprio_mask=image_text_proprio_mask.to(device),
                action_mask=action_mask.to(device),
                vlm_position_ids=vlm_position_ids.to(device),
                proprio_position_ids=proprio_position_ids.to(device),
                action_position_ids=action_position_ids.to(device),
                proprios=dummy_proprio.to(dtype).to(device),
            )
        print("\n\n=========================")
        print("Final action dimensions:", actions.shape)
        print("Final action values:", actions)
    print("Time taken:", time.time() - start_time)
    print("============================\n\n")