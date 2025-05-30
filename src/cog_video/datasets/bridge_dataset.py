import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from tqdm import tqdm
import torch
import random
import imageio
import cv2
from decord import VideoReader, cpu
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
#from finetune.constants import LOG_LEVEL, LOG_NAME
import numpy as np
from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
)



# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

#logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseBridgeDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        dataset_names: str,
        mode: str,
        sequence_length: int,
        #action_lens: int,
        #device: torch.device,
        use_feature = True,
        max_length = None,
        trainer = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.samples = []
        self.ann_files = []
        self.samples_num = []
        #print('action_lens:',action_lens)
        
        dataset_list = dataset_names.split('+')
        self.data_path = [f'{data_root}/{dataset_name}/{"annotation"}/{mode}' for dataset_name in dataset_list]
        self.dataset_root = [f'{data_root}/{dataset_name}' for dataset_name in dataset_list]
        self.video_path = [f'{data_root}/{dataset_name}' for dataset_name in dataset_list]
        self.sequence_length = sequence_length
        self.use_feature = use_feature

        #for dataset_idx, data_path in enumerate(self.data_path):
        #    ann_files = self._init_anns(data_path)
        #    samples = self._init_sequences(ann_files)

        anno_alls = [f'{data_root}/{dataset_name}/{"annotation_all"}/{mode}/all.json' for dataset_name in dataset_list]
    
        for anno_all in anno_alls:
            with open(anno_all, "r") as f:
                labels = json.load(f)
                if mode == "test" and max_length is not None:
                    labels = labels[:max_length]
                self.samples.append(labels)
                self.samples_num.append(len(labels))
            print(f'anno_all: {anno_all}, sample_num:{len(labels)}')

            #self.ann_files.append(ann_files)
            #self.samples.append(samples)
            #self.samples_num.append(len(samples))
            #print(f'dataset_idx: {dataset_idx}, {data_path}')
            #print(f'dataset_idx: {dataset_idx}, mode:{mode}, trajectories_num:{len(ann_files)}, sample_num:{len(samples)},')
        
        self.trainer = trainer

        #self.device = device
        #self.encode_video = trainer.encode_video
        #self.encode_text = trainer.encode_text

        with open(f'config/bridge_statistics.json', "r") as f:
            self.stat = json.load(f)
        self.a_min = np.array(self.stat['action']['p01'])
        self.a_max = np.array(self.stat['action']['p99'])
        self.s_min = np.array(self.stat['proprio']['p01'])
        self.s_max = np.array(self.stat['proprio']['p99'])

        intrinsic_config = json.load(open("scripts/intrinsics.json"))
        self.dataset_intrinsics = {}

        height, width = 224, 224

        for k, v in intrinsic_config.items():
            K = torch.tensor(v["intrinsic"]).float()
            K[:2] *= torch.tensor([width / v["width"], height / v["height"]])[:, None]
            self.dataset_intrinsics[k] = K

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata
    
    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return ann_files
    
    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files}
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):

        samples = []
        try:
            with open(ann_file, "r") as f:
                ann = json.load(f)
        except:
            print(f'skip {ann_file}')
            return samples
        try:
            n_frames = len(ann['action'])
        except:
            n_frames = ann['video_length']

        # create multiple samples for robot data      
        sequence_interval = 1
        start_interval = 4
        # record idx for each clip
        base_idx = np.arange(0,self.sequence_length)*sequence_interval
        max_idx = np.ones_like(base_idx)*(n_frames-1)
        for start_frame in range(0,n_frames,start_interval):
            idx = base_idx + start_frame
            idx = np.minimum(idx,max_idx)
            if len(idx) == self.sequence_length:
                sample = dict()
                sample['ann_file'] = ann_file
                sample['frame_ids'] = idx.tolist()
                samples.append(sample)

        return samples

    def __len__(self):

        return max(self.samples_num)
    
    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id = temp_cam_id, pre_encode = pre_encode, video_dir=video_dir)
        return frames, temp_cam_id
    
    def _get_frames(self, label, frame_ids, cam_id, pre_encode, video_dir):
        # directly load videos latent after svd-vae encoder
        if pre_encode: 
            video_path = label['latent_videos'][cam_id]['latent_video_path']
            try:
                video_path = os.path.join(video_dir,video_path)
                frames = self._load_latent_video(video_path, frame_ids)
            except:
                video_path = video_path.replace("latent_videos", "latent_videos_svd")
                frames = self._load_latent_video(video_path, frame_ids)
        
        #else :
            #img_path = label['videos'][cam_id]['img_path']
            #img_path = os.path.join(video_dir,img_path)
            #frames = []
            #for frame_id in frame_ids:
            #    frame_path = os.path.join(img_path,f'{frame_id}.jpg')
            #    frame = imageio.v2.imread(frame_path)
            #    frames.append(frame)
            #frames = np.stack(frames)
            #frames = torch.from_numpy(frames)

        # load original videos
        elif 'videos' in label: 
            video_path = label['videos'][cam_id]['video_path']
            video_path = os.path.join(video_dir,video_path)
            frames = self._load_video(video_path, frame_ids)
            frames = [cv2.resize(frame, (self.width, self.height)) for frame in frames]
            frames = np.stack(frames)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (l, c, h, w)
            #frames = torch.clamp(frames*255.0,0,255).to(torch.uint8)
            frames = frames.float().contiguous()
        else:
            frames = []
            #for frame_id in frame_ids:
            # for j in range(len(frame_ids)):
            #     frame_path = os.path.join(video_dir,label['img_path'][j])
            #     frame = imageio.v2.imread(frame_path)
            #     frame = cv2.resize(frame, (self.width, self.height))
            #     frames.append(frame)
            for img_path in label['img_path']:
                frame_path = os.path.join(video_dir,img_path)
                frame = imageio.v2.imread(frame_path)
                frame = cv2.resize(frame, (self.width, self.height))
                frames.append(frame)
            #print('frame_num:',len(frames))
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float().contiguous()

        return frames
    
    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).numpy() #(frame, h, w, c)
        # central crop
        #h, w = frame_data.shape[1], frame_data.shape[2]
        #if h > w:
        #    margin = (h - w) // 2
        #    frame_data = frame_data[:, margin:margin + w]
        #elif w > h:
        #    margin = (w - h) // 2
        #    frame_data = frame_data[:, :, margin:margin + h]
        return frame_data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index
        
        sampled_dataset = self.samples[0]
        sampled_video_dir = self.dataset_root[0]

        sampled_idx = index % len(sampled_dataset)
        sample = sampled_dataset[sampled_idx]
        ann_file = os.path.join(sampled_video_dir,sample['ann_file'])
        with open(ann_file, "r") as f:
             action_label = json.load(f)
        
        #    label = json.load(f)

        #ann_file = sample['ann_file']
        # print(index, ann_file)
        frame_ids = sample['frame_ids']
        label = sample
        #with open(ann_file, "r") as f:
        #    label = json.load(f)
        
        episode=label['episode_id']
        curr_idx = frame_ids[0]
        future_idx = frame_ids[-1]
        save_cache_name = str(episode)+'_'+str(curr_idx)+'_'+str(future_idx)
        #video, _ = self._get_obs(label, frame_ids, cam_id=0, pre_encode=False, video_dir=sampled_video_dir)
        #print('video_shape:',video.shape)
        prompt = label['texts']
        #print('prompt:',prompt)
        #image = video[0]
        action = np.stack(action_label['action'])
        action = action[frame_ids]
        action = action[:self.sequence_length]
        proprio = np.stack(action_label['state'])
        proprio = proprio[[curr_idx]]
        action = self.normalize_bound(action, self.a_min, self.a_max)
        proprio = self.normalize_bound(proprio, self.s_min, self.s_max)

        #action = np.stack(action)
        action = torch.from_numpy(action).float().contiguous().to("cpu")
        #proprio = np.stack(proprio)
        proprio = torch.from_numpy(proprio).float().contiguous().to("cpu")

        intrinsics = self.dataset_intrinsics["default"]

        if not self.use_feature:
            video, _ = self._get_obs(label, frame_ids, cam_id=0, pre_encode=False, video_dir=sampled_video_dir)
            image = video[0]
            image = image.to("cpu")
            image = image.to(torch.uint8)
            return {
            "image": image,
            "prompt_embedding": prompt,
            "text": prompt,
            "video": video,
            "action": action,
            "proprio": proprio,
            "intrinsic": intrinsics,
            }

        video_latent_dir = os.path.join(sampled_video_dir , "video_cache" , "cogvideoxbridge") 
        prompt_embeddings_dir = os.path.join(sampled_video_dir , "prompt_cache" , "cogvideoxbridge") 
        
        video_latent_dir = Path(video_latent_dir)
        prompt_embeddings_dir = Path(prompt_embeddings_dir) 
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        encoded_video_path = video_latent_dir / (save_cache_name + ".safetensors")
        prompt_embedding_path = prompt_embeddings_dir / (str(episode) + ".safetensors")
        save = True

        if prompt_embedding_path.exists():
            try:
                prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            except Exception as e:
                print(f"Error loading prompt embedding from {prompt_embedding_path}: {str(e)}")
                # 如果加载失败，重新生成并保存
                prompt_embedding = self.encode_text(prompt)
                prompt_embedding = prompt_embedding.to("cpu")
                prompt_embedding = prompt_embedding[0]
                try:
                    save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
                    print(f"Saved new prompt embedding to {prompt_embedding_path}")
                except Exception as e:
                    print(f"Error saving prompt embedding: {str(e)}")
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            prompt_embedding = prompt_embedding[0]
            try:
                save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
                print(f"Saved prompt embedding to {prompt_embedding_path},shape:{prompt_embedding.shape}")
            except Exception as e:
                print(f"Error saving prompt embedding: {str(e)}")

        if encoded_video_path.exists():
            try:
                encoded_video = load_file(encoded_video_path)["encoded_video"]
                video, _ = self._get_obs(label, [curr_idx], cam_id=0, pre_encode=False, video_dir=sampled_video_dir)
                image = video[0]
                image = image.to("cpu")
                image = image.to(torch.uint8)
            except Exception as e:
                print(f"Error loading encoded video from {encoded_video_path}: {str(e)}")
                # 如果加载失败，重新生成并保存
                video, _ = self._get_obs(label, frame_ids, cam_id=0, pre_encode=False, video_dir=sampled_video_dir)
                image = video[0]
                frames = video
                frames = self.video_transform(frames)
                frames = frames.unsqueeze(0)
                frames = frames.permute(0, 2, 1, 3, 4).contiguous()
                encoded_video = self.encode_video(frames)
                encoded_video = encoded_video[0]
                encoded_video = encoded_video.to("cpu")
                image = image.to("cpu")
                image = image.to(torch.uint8)
                try:
                    save_file({"encoded_video": encoded_video}, encoded_video_path)
                    print(f"Saved new encoded video to {encoded_video_path},shape:{encoded_video.shape}")
                except Exception as e:
                    print(f"Error saving encoded video: {str(e)}")
        else:
            video, _ = self._get_obs(label, frame_ids, cam_id=0, pre_encode=False, video_dir=sampled_video_dir)
            image = video[0]
            frames = video
            frames = self.video_transform(frames)
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            image = image.to("cpu")
            image = image.to(torch.uint8)
            try:
                save_file({"encoded_video": encoded_video}, encoded_video_path)
                print(f"Saved new encoded video to {encoded_video_path},shape:{encoded_video.shape}")
            except Exception as e:
                print(f"Error saving encoded video: {str(e)}")

        action = action.to(encoded_video.dtype)
        proprio = proprio.to(encoded_video.dtype)

        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "text": prompt,
            "encoded_video": encoded_video,
            #"save_cache_name": save_cache_name,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
            "action": action,
            "proprio": proprio,
            "intrinsic": intrinsics,
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class BridgeDatasetWithResize(BaseBridgeDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)


class I2VDatasetWithBuckets(BaseBridgeDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]
        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        video = preprocess_video_with_buckets(video_path, self.video_resolution_buckets)
        image = preprocess_image_with_resize(image_path, video.shape[2], video.shape[3])
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)