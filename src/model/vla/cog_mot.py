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

from src.model.cog_video.schemas import Components, State
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

from src.model.cog_video.models.Components.MOT3DModel import CogVideoXMOT3DModel
from src.model.cog_video.models.Components.Pipeline import CogVideoXMOTPipeline
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from src.model.cog_video.utils import cast_training_params

from diffusers.models.embeddings import get_3d_rotary_pos_embed

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}


class CogMot(nn.Module, NoSyncBase):
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

        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=cfg.video.train_resolution[0],
            train_height=cfg.video.train_resolution[1],
            train_width=cfg.video.train_resolution[2],
        )

        self.components = self.load_components()
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
             dtype=self.state.weight_dtype
        )

        self.state.transformer_config = self.components.transformer.config

        #self.video_transformer = CogVideoXTransformer3DModel.from_pretrained(cfg.video_model_path, subfolder="transformer")

        # Action, proprio, time encoders
        self.action_expert_adaptive_mode = cfg.action_expert_adaptive_mode
        #if cfg.action_expert_adaptive_mode:  # adaLN or adaLN-Zero
        #    self.action_encoder = ActionEncoder(
        #        self.action_dim,
        #        self.action_hidden_size,
        #        time_cond=False,
        #    )
        #    self.time_embedding = SinusoidalPosEmb(
        #        cfg.time_hidden_size, cfg.time_max_period
        #    )
        #else:  # matching pi0
        #    self.action_encoder = ActionEncoder(
        #        self.action_dim,
        #        self.action_hidden_size,
        #        time_cond=True,
        #    )
        #    self.time_embedding = SinusoidalPosEmb(
        #        self.action_hidden_size, cfg.time_max_period
        #    )
        self.latent_action = nn.Parameter(torch.randn(self.num_action_tokens, self.action_hidden_size))
        self.proprio_encoder = nn.Linear(
            self.proprio_dim,
            self.proprio_hidden_size,
        )

        # Action decoder
        #self.action_decoder = nn.Linear(
        #    self.action_hidden_size,
        #    self.action_dim,
        #)

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )
            self.lm_head.weight = self.embed_tokens.weight  # tie weights

    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.cfg.video.model_path)

        components.pipeline_cls = CogVideoXMOTPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        
        transformer_config = CogVideoXMOT3DModel.load_config(cfg.video.video_config_path)
        init_dict, unused_kwargs, hidden_dict = CogVideoXMOT3DModel.extract_init_dict(transformer_config)
        self.components.transformer = CogVideoXMOT3DModel(**init_dict)
        if self.cfg.video.video_model_path is not None:
            new_path = f'{self.cfg.video.video_model_path}/torch_model.pt'
            pretrained_dict = torch.load(new_path, map_location="cpu",weights_only=True)
            pretrained_dict = {k.replace('module.',''):v for k,v in pretrained_dict.items()}
            print(f"load_from_ckpt: {self.cfg.video.video_model_path}/torch_model.pt")
            # update
            state_dict = self.components.transformer.state_dict()
            state_dict.update(pretrained_dict)
            self.components.transformer.load_state_dict(state_dict)

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        self.flow_sampling = "beta"
        if self.flow_sampling == "beta":
            flow_alpha = 1.5
            flow_beta = 1
            self.flow_t_max = 1 - 0.001
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
            self.flow_sig_min = 0.001

        return components

    def prepare_components_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.cfg.video.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.cfg.video.rank,
                lora_alpha=self.cfg.video.lora_alpha,
                init_lora_weights=True,
                target_modules=self.cfg.video.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            #self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        #self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.cfg.video.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.transformer], dtype=torch.float32)

        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the parameters
        trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
        if hasattr(self.components.transformer,'lvm_transformer'):
            lvm_trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.lvm_transformer.parameters()))
            self.state.num_lvm_trainable_parameters = sum(p.numel() for p in lvm_trainable_parameters)

        if hasattr(self.components.transformer,'action_expert'):
            action_trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.action_expert.parameters()))
            self.state.num_action_trainable_parameters = sum(p.numel() for p in action_trainable_parameters)

        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.cfg.video.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        #use_deepspeed_opt = (
        #    self.accelerator.state.deepspeed_plugin is not None
        #    and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        #)
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.cfg.video.optimizer.optimizer_name,
            learning_rate=self.cfg.video.optimizer.learning_rate,
            beta1=self.cfg.video.optimizer.beta1,
            beta2=self.cfg.video.optimizer.beta2,
            beta3=self.cfg.video.optimizer.beta3,
            epsilon=self.cfg.video.optimizer.epsilon,
            weight_decay=self.cfg.video.weight_decay,
           # use_deepspeed=use_deepspeed_opt,
        )

        #num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        #if self.args.train_steps is None:
        #    self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        #    self.state.overwrote_max_train_steps = True

        # use_deepspeed_lr_scheduler = (
        #     self.accelerator.state.deepspeed_plugin is not None
        #     and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        # )
        use_deepspeed_lr_scheduler = False
        #total_training_steps = self.args.train_steps * self.accelerator.num_processes
        #num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.cfg.video.optimizer.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.cfg.video.optimizer.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.cfg.video.optimizer.lr_num_cycles,
                power=self.cfg.video.optimizer.lr_power,
            )

        #self.optimizer = optimizer
        #self.lr_scheduler = lr_scheduler
        return optimizer, lr_scheduler

    @property
    def action_expert_parameters(self):
        return (
            #list(self.action_encoder.parameters())
            #+ list(self.action_decoder.parameters())
            list(self.proprio_encoder.parameters())
            + list(self.latent_action.parameters())
            + list(self.joint_model.mixtures["latent_action"].parameters())
        )  # note: action and proprio share weights

    @property
    def trainable_vlm_parameters(self):
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
            if "embed_tokens" in k:
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
        self, causal_mask: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """split into ones for paligemma and action"""
        image_text_proprio_mask = causal_mask[
            ...,
            : self.max_image_text_tokens + self.num_proprio_tokens,
            : self.max_image_text_tokens + self.num_proprio_tokens,
        ]
        action_mask = causal_mask[..., -self.num_action_tokens :, :]
        return image_text_proprio_mask, action_mask

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

    def __get_training_dtype(self) -> torch.dtype:
        if self.cfg.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.cfg.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.cfg.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.cfg.mixed_precision}")

    # ---------- Inference ----------#

    def _forward_siglip_and_text_embedding(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device

        # text embedding
        # [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.embed_tokens(input_ids)

        # image features from siglip and projector
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        selected_image_feature = self.vision_tower(pixel_values)
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

    def infer_latent_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_text_proprio_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
    ) -> torch.FloatTensor:
        dtype, device = pixel_values.dtype, pixel_values.device
        bsz = pixel_values.size(0)

        latent_caches = self.joint_model.build_latent_caches()

        # merge the text tokens and the image tokens
        inputs_embeds = self._forward_siglip_and_text_embedding(input_ids, pixel_values)

        # proprio
        proprio_embeds = self.proprio_encoder(proprios)

        latent_action_embeds = self.latent_action.repeat(bsz, 1, 1)

        # forward pass thru the vlm and proprio, cache the kv
        hidden_states_all, _, latent_caches = self.joint_model(
            attention_mask=image_text_proprio_mask,
            position_ids_all={
                "vlm": vlm_position_ids,
                "proprio": proprio_position_ids,
                "latent_action": action_position_ids,
            },
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "latent_action": latent_action_embeds,
            },
            kv_caches=kv_caches,
            latent_caches=latent_caches,
            return_caches=True,
            return_latent_caches=True,
        )

        return hidden_states_all, latent_caches

    def forward(
        self,
        image_text_proprio_mask: = None,
        latent: torch.Tensor,
        input_ids: torch.LongTensor,
        #images: torch.Tensor,
        pixel_values: torch.Tensor,
        proprios: torch.Tensor,
        action: torch.Tensor,
        timesteps: Union[int, float, torch.LongTensor],
        attn_mask: Optional[torch.Tensor] = None,
        causal_mask: torch.FloatTensor,
        vlm_position_ids: torch.LongTensor,
        proprio_position_ids: torch.LongTensor,
        action_position_ids: torch.LongTensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        device=pixel_values.device
        dtype=pixel_values.dtype

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape
        mode_list = ['m2v']*batch_size
        attention_kwargs = {}
        attention_kwargs['attention_mask'] = mode_list
         
        # Get prompt embeddings
        prompt_embedding = None
        hidden_states_action, _ = self.infer_latent_action(input_ids, pixel_values, image_text_proprio_mask, vlm_position_ids, proprio_position_ids, action_position_ids, proprios)
        hidden_states_action = hidden_states_action['latent_action']
        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = pixel_values.unsqueeze(2)

        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)
         
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        #total_loss = 0.0

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )

        predicted_noise = self.components.transformer(
                hidden_states=latent_img_noisy,
                encoder_hidden_states=prompt_embedding,
                action_states=hidden_states_action,
                timestep=timesteps,
                action_timestep=timesteps,
                ofs=ofs_emb,
                image_rotary_emb=rotary_emb,
                return_dict=False,
                attention_kwargs = attention_kwargs
            )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_noisy, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin

