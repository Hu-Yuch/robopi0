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
from src.cog_video.datasets import BridgeDatasetWithResize


SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
ZOE_MEAN, ZOE_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# dummy
print(pretty_errors.__version__)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

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

def _main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # figure out the current gpu
    multi_gpu = torch.cuda.device_count() > 1 or cfg.get("n_nodes", 1) > 1
    if multi_gpu:
        from torch.distributed import destroy_process_group, init_process_group

        def ddp_setup():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            init_process_group(backend="nccl")

        ddp_setup()
        gpu_id = int(os.environ["LOCAL_RANK"])
    else:
        gpu_id = 0
    with open_dict(cfg):
        cfg.gpu_id = gpu_id
        cfg.multi_gpu = multi_gpu

    # seeding
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(f"cuda:{gpu_id}")

    # run agent
    # cls = hydra.utils.get_class(cfg._target_)
    # agent = cls(cfg)
    # agent.run()
    #train_dataset = Dataset_Real(cfg.dataset_args, mode='train')
    train_dataset = BridgeDatasetWithResize(**cfg.data.train, mode='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=cfg.dataset_args.shuffle,
        num_workers=cfg.dataset_args.num_workers,
    )

    torch_dtype = torch.bfloat16 if cfg.get("use_bf16", True) else torch.float32
    torch_dtype = torch.float32
    vision_zoe_config = ZoeDepthConfig.from_pretrained(
        cfg.vision_zoe_path, torch_dtype=torch_dtype, local_files_only=True
    )
    vision_zoe_model = ZoeDepthForDepthEstimation.from_pretrained(  # zoe does not support Flash Attention 2.0 yet.
        cfg.vision_zoe_path,
        config=vision_zoe_config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(device)
    vision_zoe_model = vision_zoe_model.eval()

    for param in vision_zoe_model.parameters():
        param.requires_grad = False

    # register buffer
    patch_size, reso, image_size = cfg.vision.config.patch_size, cfg.ego3d_patch_reso, cfg.vision.config.image_size
    patch_size = 1
    reso = 1
    y, x = torch.meshgrid(torch.arange(0, image_size, patch_size // reso), torch.arange(0, image_size, patch_size // reso), indexing="ij")  # (h//sp w//sp)
    y, x = y + patch_size / reso / 2, x + patch_size / reso / 2
    uv_h = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1).to(device)  # (3 hw)

    tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_path, padding_side="right"
        )
    processor = VLAProcessor(
        tokenizer,
        num_image_tokens=cfg.vision.config.num_image_tokens,
        max_seq_len=cfg.max_seq_len,
        tokenizer_padding=cfg.tokenizer_padding,
    )

    def preprocess_batch(batch, split_mask: bool, sample_fm_time: bool):
        # TODO(allenzren): support multi-image / proprio history
        images = batch["image"]
        proprios = batch["proprio"]
        actions = batch["action"] #.squeeze(1)  # remove the time dimension
        # texts = [
        #     text.decode("utf-8") for text in batch["task"]["language_instruction"]
        # ]
        # print("images", images.shape)
        # print("propios", proprios.shape)
        # print("actions", actions.shape)
        texts = batch["text"]
        intrinsic = batch["intrinsic"]
        #images = einops.rearrange(
        #    images, "B T H W C -> B (T C) H W"
        #)  # remove cond_steps dimension
        model_inputs = processor(text=texts, images=images)

        # build causal mask and position ids for action

        inputs = {
            "input_ids": model_inputs["input_ids"],
            "pixel_values": model_inputs["pixel_values"].to(torch_dtype),
            "proprios": proprios.to(torch_dtype),
            "actions": actions.to(torch_dtype),
            "intrinsic": intrinsic.to(torch_dtype),
        }

        inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def backproject_patch(K: torch.Tensor, depth: torch.Tensor, patch_size=14, reso=2) -> torch.Tensor:
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
        p_cam = (inv(K.float()) @ uv_h.float()) * patch_depth  # (b 3 3) @ (3 hw) -> (b 3 hw) * (b 1 hw) -> (b 3 hw)
        patch_p_cam = p_cam.reshape(b, 3, hp, sub_hp, wp, sub_wp).permute(0, 2, 4, 3, 5, 1).reshape(b, hp * wp, -1)
        return patch_p_cam

    i = 0
    for batch in train_dataloader:
        print('i')
        if i > 100:
            break
        """
        batch: dict with keys 'observation', 'task', 'action', 'dataset_name', 'action_pad_mask'
        observation: 'image_primary' (torch.Size([bsz, 1, H, W, 3], uint8), 'image_wrist', 'timestep' (torch.Size([bsz, 1])), 'pad_mask_dict', 'timestep_pad_mask', 'task_completed' (torch.Size([bsz, window, 4]), 'proprio' (torch.Size([bsz, window, proprio_dim])
        task: 'language_instruction', 'pad_mask_dict', 'image_primary', 'image_wrist', 'timestep' (torch.Size([bsz]))
        action (torch.Size([bsz, window, horizon, action_dim], float32)
        action_pad_mask (torch.Size([bsz, window, horizon, action_dim]))
        """

        inputs = preprocess_batch(batch, split_mask=False, sample_fm_time=True)

        pixel_values = inputs["pixel_values"]
        intrinsic = inputs["intrinsic"]

        pixel_values = (pixel_values + 1) / 2
        zoe_pixel_values, ph, pw = process_zoe(pixel_values, pad_mode="reflect")
        with torch.no_grad():
            pvh, pvw = pixel_values.shape[-2:]
            depth = vision_zoe_model(pixel_values=zoe_pixel_values).predicted_depth
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(pvh+2*ph, pvw+2*pw),
                mode="bicubic",
                align_corners=True,
            )[..., ph:-ph, pw:-pw]
            xyz = backproject_patch(
                intrinsic, depth, patch_size=1, reso=1
            )  # (b, n, 3*4)
            #xyz = backproject_patch(
            #    intrinsic, depth, patch_size=cfg.vision.config.patch_size, reso=cfg.ego3d_patch_reso
            #)  # (b, n, 3*4)
            #print("xyz", xyz.shape)

                # 保存原始图片
        # img = pixel_values[0].permute(1, 2, 0).cpu().numpy()  # 转换为 HxWx3
        # print("img", img.shape)
        # img = (img * 255).astype(np.uint8)  # 转换为0-255范围
        # img = Image.fromarray(img)
        # img.save(f'pointcloud/image_{i}.png')
        # print(f"保存图片到: pointcloud/image_{i}.png")
        
        # pixel_values = pixel_values.reshape(pixel_values.shape[0], 3 ,-1)
        # # 创建点云对象
        # pcd = o3d.geometry.PointCloud()
        # #print("xyz", xyz.shape)
        # # 设置点云坐标
        # pcd.points = o3d.utility.Vector3dVector(xyz[0].cpu().numpy())
        # # 设置点云颜色
        # pcd.colors = o3d.utility.Vector3dVector(pixel_values[0].T.cpu().numpy())
        
        # # 保存点云
        # save_path = f'pointcloud/point_cloud_{i}.ply'
        # o3d.io.write_point_cloud(save_path, pcd)
        # print(f"保存点云到: {save_path}")
        

        
        i += 1
            # 可视化点云
    if multi_gpu:
        destroy_process_group()


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "config/train"),
    config_name="bridge.yaml",
)  # defaults
def main(cfg: OmegaConf):
    _main(cfg)


if __name__ == "__main__":
    main()