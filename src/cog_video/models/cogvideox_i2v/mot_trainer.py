from finetune.datasets import I2VDatasetWithResize, T2VDatasetWithResize
from finetune.datasets import BridgeDatasetWithResize
from typing import Any, Dict, List, Tuple
from accelerate.logging import get_logger
from tqdm import tqdm
from diffusers.utils.export_utils import export_to_video
import json
import hashlib
import wandb
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from ..Components.Pipeline import CogVideoXMOTPipeline
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from ..Components.Pipeline import retrieve_timesteps
from ..Components.joint_model import CogVideoJointModel, CogVideoXMOT3DModel
from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from diffusers.utils.torch_utils import randn_tensor
from ..utils import register
from finetune.constants import LOG_LEVEL, LOG_NAME
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)
from finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)

logger = get_logger(LOG_NAME, LOG_LEVEL)

class CogVideoMOTTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "i2v":
            self.dataset = BridgeDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                sequence_length=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                #action_lens=self.args.action_lens,
                mode = "train",
                dataset_names = self.args.dataset_name,
                trainer=self,
            )
        elif self.args.model_type == "t2v":
            self.dataset = BridgeDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                sequence_length=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                #action_lens=self.args.action_lens,
                mode = "train",
                dataset_names = self.args.dataset_name,
                trainer=self,
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        #for _ in tmp_data_loader:
        #    ...
         

        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )


    @override
    def prepare_for_validation(self):
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "i2v":
            self.val_dataset = BridgeDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                sequence_length=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                #action_lens=self.args.action_lens,
                mode = "test",
                dataset_names = self.args.dataset_name,
                use_feature = False,
                trainer=self,
            )
        elif self.args.model_type == "t2v":
            self.val_dataset = BridgeDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                sequence_length=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                #action_lens=self.args.action_lens,
                mode = "test",
                dataset_names = self.args.dataset_name,
                trainer=self,
                use_feature = False,
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")
        
        tmp_val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_val_data_loader = self.accelerator.prepare_data_loader(tmp_val_data_loader)
        #for _ in tmp_val_data_loader:
        #    ...

        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        self.eval_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5]

        self.val_data_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate_fn,
            batch_size=16,
            num_workers=0,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )


    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXMOTPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
        
        transformer_config = CogVideoXMOT3DModel.load_config('configs/MOT_config.json')
        init_dict, unused_kwargs, hidden_dict = CogVideoXMOT3DModel.extract_init_dict(transformer_config)
        components.transformer = CogVideoXMOT3DModel(**init_dict)
        if self.args.resume_from_checkpoint is not None:
            new_path = f'{self.args.resume_from_checkpoint}/torch_model.pt'
            pretrained_dict = torch.load(new_path, map_location="cpu",weights_only=True)
            pretrained_dict = {k.replace('module.',''):v for k,v in pretrained_dict.items()}
            new_pretrained_dict = {}
            for key,value in pretrained_dict.items():
                if 'action_decoder' in key:
                    print('delete_key:',key)
                else:
                    new_pretrained_dict[key] = value

            print(f"load_from_ckpt: {self.args.resume_from_checkpoint}/torch_model.pt")
            # update
            state_dict = components.transformer.state_dict()
            state_dict.update(pretrained_dict)
            components.transformer.load_state_dict(state_dict)

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

    @override
    def initialize_pipeline(self) -> CogVideoXMOTPipeline:
        pipe = CogVideoXMOTPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "action":[], "proprio":[]}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            action = sample["action"]
            proprio = sample["proprio"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)
            ret["action"].append(action)
            ret["proprio"].append(proprio)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        if not isinstance(ret["prompt_embedding"][0],str):
           ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["action"] = torch.stack(ret["action"])
        ret["proprio"] = torch.stack(ret["proprio"])

        return ret
    @override
    def eval_collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"video": [], "prompt_embedding": [], "images": [], "action":[], "proprio":[]}

        for sample in samples:
            video = sample["video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            action = sample["action"]
            proprio = sample["proprio"]

            ret["video"].append(video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)
            ret["action"].append(action)
            ret["proprio"].append(proprio)

        ret["video"] = torch.stack(ret["video"])
        if not isinstance(ret["prompt_embedding"][0],str):
           ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])
        ret["proprio"] = torch.stack(ret["proprio"])
        ret["action"] = torch.stack(ret["action"])

        return ret
    
    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))
    
    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def prepare_mask(self, mode_list: list, text_seq_length: int, video_seq_length: int, action_seq_length: int):
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
            else:
                attention_mask = attention_mask.bool()
        return attention_mask

    
    @override
    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        #num_validation_samples = len(self.state.validation_prompts)

        #if num_validation_samples == 0:
        #    logger.warning("No validation samples found. Skipping validation.")
        #    return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=["transformer"])
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################
        num_validation_samples = 0
        mode_list = ['a2v','t2v'] if self.args.train_mode == "video" else ['iv2a']

        all_processes_artifacts = []
        if self.args.train_mode == "video":
            for step_num, batch in enumerate(self.val_data_loader):
                for mode in mode_list:
                    for i in range(len(batch["prompt_embedding"])):
                        #if self.state.using_deepspeed and self.accelerator.deepspeed_plugin.zero_stage != 3:
                            # Skip current validation on all processes but one
                            #if i % accelerator.num_processes != accelerator.process_index:
                            #    continue
                        num_validation_samples +=1
                        prompt = batch["prompt_embedding"][i]
                        image = batch["images"][i]
                        video = batch["video"][i]
                        proprio = batch["proprio"][i]
                        action = batch["action"][i]

                        if image is not None:
                            #image = preprocess_image_with_resize(image, self.state.train_height, self.state.train_width)
                            # Convert image tensor (C, H, W) to PIL images
                            image = image.to(torch.uint8)
                            image = image.permute(1, 2, 0).cpu().numpy()
                            image = Image.fromarray(image)

                        if video is not None:
                            #video = preprocess_video_with_resize(
                            #    video, self.state.train_frames, self.state.train_height, self.state.train_width
                            #)
                            # Convert video tensor (F, C, H, W) to list of PIL images
                            video = video.round().clamp(0, 255).to(torch.uint8)
                            video = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]

                        logger.debug(
                            f"Validating sample {i + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {prompt}",
                            main_process_only=False,
                        )
                        validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video, "proprio": proprio, "action": action}, pipe, mode=mode)

                        if (
                            self.state.using_deepspeed
                            and self.accelerator.deepspeed_plugin.zero_stage == 3
                            and not accelerator.is_main_process
                        ):
                            continue

                        prompt_filename = string_to_filename(prompt)[:25]
                        # Calculate hash of reversed prompt as a unique identifier
                        reversed_prompt = prompt[::-1]
                        hash_suffix = hashlib.md5(reversed_prompt.encode()).hexdigest()[:5]

                        artifacts = {
                            "image": {"type": "image", "value": image},
                            "video": {"type": "video", "value": video},
                        }
                        for j, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                            artifacts.update({f"artifact_{step_num}_{j}": {"type": artifact_type, "value": artifact_value}})
                        logger.debug(
                            f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                            main_process_only=False,
                        )

                        for key, value in list(artifacts.items()):
                            artifact_type = value["type"]
                            artifact_value = value["value"]
                            if artifact_type not in ["image", "video"] or artifact_value is None:
                                continue

                            extension = "png" if artifact_type == "image" else "mp4"
                            filename = f"validation-{step}-{mode}-{key}-{accelerator.process_index}-{prompt_filename}.{extension}"
                            validation_path = self.args.output_dir / "validation_res"/ str(step)
                            validation_path.mkdir(parents=True, exist_ok=True)
                            filename = str(validation_path / filename)

                            if artifact_type == "image":
                                logger.debug(f"Saving image to {filename}")
                                artifact_value.save(filename)
                                artifact_value = wandb.Image(filename)
                            elif artifact_type == "video":
                                logger.debug(f"Saving video to {filename}")
                                export_to_video(artifact_value, filename, fps=self.args.gen_fps)
                                artifact_value = wandb.Video(filename, caption=prompt)

                            all_processes_artifacts.append(artifact_value)

            all_artifacts = gather_object(all_processes_artifacts)

            if accelerator.is_main_process:
                tracker_key = "validation"
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        image_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)]
                        video_artifacts = [artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)]
                        tracker.log(
                            {
                                tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                            },
                            step=step,
                        )
        
        else:
            logs = {}
            eval_l1_loss = torch.tensor(0.0, device=self.accelerator.device)
            batch_num = 0
            for step_id, batch in enumerate(self.val_data_loader):
                action = batch["action"]
                #print('action_shape:',action.shape)
                batch_num+= 1
                print('batch_num:',batch_num)

            eval_progress_bar = tqdm(
                range(0, batch_num),
                initial=0,
                desc="Eval steps",
                disable=not self.accelerator.is_local_main_process,
            )
            for step_id, batch in enumerate(self.val_data_loader):
                for mode in mode_list:

                    prompt = batch["prompt_embedding"]
                    image = batch["images"]
                    video = batch["video"]
                    proprio = batch["proprio"]
                    action = batch["action"]
                    if image is not None:
                        #image = preprocess_image_with_resize(image, self.state.train_height, self.state.train_width)
                        # Convert image tensor (C, H, W) to PIL images
                        image = image.to(torch.uint8)
                        image = image.permute(0 , 2, 3, 1).cpu().numpy()
                        image = [Image.fromarray(image[i]) for i in range(image.shape[0])]
                        #image = Image.fromarray(image)
                    
                    validation_artifacts = self.validation_step({"prompt": prompt, "image": image, "video": video, "proprio": proprio, "action": action}, pipe, mode=mode)
                    preds = validation_artifacts["action"]
                    action = action.to(preds.device)
                    current_loss = torch.nn.functional.l1_loss(
                                preds, action)
                    eval_l1_loss += current_loss
                    logger.info(f"eval_l1_loss:{current_loss}")
                eval_progress_bar.update(1)
            
            eval_l1_loss = eval_l1_loss / batch_num
 
            logger.info(
                            f"avg_eval_l1_loss: {eval_l1_loss}",
                            main_process_only=False,
                        )
            logs = {}
            logs["loss"] = eval_l1_loss.detach().item()
            accelerator.log(logs, step=step)

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST)
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        #mode_list = ['a2v','t2v']
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        batch_size = latent.shape[0]
        images = batch["images"]
        proprio = batch["proprio"]
        action = batch["action"]
        proprio = proprio.repeat(1,action.size(1),1)
        action = torch.cat([proprio,action],dim=-1)
        if self.args.train_mode == "video":
            latent = latent.repeat_interleave(2,dim=0)
            images = images.repeat_interleave(2,dim=0)
            prompt_embedding = prompt_embedding.repeat_interleave(2,dim=0)
            action = action.repeat_interleave(2,dim=0)
            mode_list = ['a2v']*batch_size + ['t2v']*batch_size
        else:
            mode_list = ['t2v']*batch_size
        #print('batch_size:',images.shape[0])

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        timesteps = timesteps.long()

        action_timestep = torch.ones(batch_size).to(self.accelerator.device)

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

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        #total_loss = 0.0

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )

        #obtain attn mask

        #attention_mask = self.prepare_mask(mode_list,prompt_embedding.size(1),latent_img_noisy.size(1),action.size(1))
        attention_kwargs = {}
        attention_kwargs['attention_mask'] = mode_list
        
        #for mode in mode_list:
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            action_states=action,
            timestep=timesteps,
            action_timestep=action_timestep,
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
        #total_loss = total_loss + loss

        return loss
    
    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t
    
    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def compute_action_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        batch_size = latent.shape[0]
        images = batch["images"]
        proprio = batch["proprio"]
        action = batch["action"]
        proprio = proprio.repeat(1,action.size(1),1)
        device = self.accelerator.device
        num_inference_steps = 40
        timesteps = None
        timesteps, num_inference_steps = retrieve_timesteps(self.components.scheduler, num_inference_steps, device, timesteps)

        images = images.unsqueeze(2)
        image_latent_dist = self.components.vae.encode(images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor
        image_latents = image_latents.permute(0, 2, 1, 3, 4)

        batch_size, num_channels, num_frames, height, width = latent.shape

        padding_shape = (
            batch_size,
            num_frames - 1,
            num_channels,
            height,
            width,
        )

        dtype = self.components.vae.dtype

        latent_padding = torch.zeros(padding_shape, device=device, dtype=self.components.vae.dtype)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        noise_latents = randn_tensor(latent.shape, generator=self.state.generator, device=device, dtype=dtype)
        noise_latents = noise_latents.permute(0, 2, 1, 3, 4)
        noise_latents = noise_latents * self.components.scheduler.init_noise_sigma

        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        #total_loss = 0.0

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        
        mode_list = ['iv2a']*batch_size
        attention_kwargs = {}
        attention_kwargs['attention_mask'] = mode_list

        #add noise to action
        t = self.sample_fm_time(batch_size).to(dtype)
        t = t.to(device)
        x0 = torch.randn_like(action, device=t.device, dtype=t.dtype)
        x1 = action
        psi_t = self.psi_t(x0, x1, t)
        psi_t = torch.cat([proprio,psi_t],dim=-1)
        
        #for mode in mode_list:
        for i, timestep in enumerate(timesteps):
            noise_latents = self.components.scheduler.scale_model_input(noise_latents, t)
            latent_img_noisy = torch.cat([noise_latents, image_latents], dim=2)

            timestep = timestep.expand(latent_img_noisy.shape[0])
            v_psi = self.components.transformer(
                hidden_states=latent_img_noisy,
                encoder_hidden_states=prompt_embedding,
                action_states=psi_t,
                timestep=timestep,
                action_timestep = t,
                ofs=ofs_emb,
                image_rotary_emb=rotary_emb,
                return_dict=False,
                attention_kwargs = attention_kwargs
            )[1]
            break

        d_psi = x1 - (1 - self.flow_sig_min) * x0
        
        loss = torch.mean((v_psi - d_psi) ** 2)
        loss = loss.mean()

        return loss


    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXMOTPipeline, mode:str
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, video, proprio, action = eval_data["prompt"], eval_data["image"], eval_data["video"], eval_data["proprio"], eval_data["action"]
        
        if self.args.train_mode == "video":
            video_generate = pipe(
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                prompt=prompt,
                image=image,
                proprio=proprio,
                action=action,
                generator=self.state.generator,
                guidance_scale = 1,
                mode = mode
            ).frames[0]
            return [("video", video_generate)]
        else:
            action_generate = pipe.infer_action(
                num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                prompt=prompt,
                image=image,
                proprio=proprio,
                action=action,
                generator=self.state.generator,
                guidance_scale = 1,
                mode = mode
            )[0]
            return {"action": action_generate}


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


register("cogvideomot", "sft", CogVideoMOTTrainer)

if __name__ == "__main__":
    transformer_config = CogVideoXMOT3DModel.load_config('MOT_config/config.json')
    init_dict, unused_kwargs, hidden_dict = CogVideoXMOT3DModel.extract_init_dict(transformer_config)
    transformer = CogVideoXMOT3DModel(**init_dict)