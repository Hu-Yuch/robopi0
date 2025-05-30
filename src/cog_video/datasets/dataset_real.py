# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import warnings
import traceback
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms as T
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import imageio
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange
from src.agent.dataset_utils import Resize_Preprocess, ToTensorVideo
from scipy.spatial.transform import Rotation as R  
# from dataset.util import update_paths
# from dataset.dataset_util import euler2rotm, rotm2euler


class Dataset_Real(Dataset):
    def __init__(
            self,
            args,
            mode = 'val'
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode

        # dataset stucture
        # dataset_dir/dataset_name/annotation_name/mode/traj
        # dataset_dir/dataset_name/video/mode/traj
        # dataset_dir/dataset_name/latent_video/mode/traj

        # prepare all datasets path
        dataset_dir = args.dataset_dir
        dataset_names = args.dataset
        dataset_list = dataset_names.split('+')
        self.data_path = [f'{dataset_dir}/{dataset_name}/{args.annotation_name}/{mode}' for dataset_name in dataset_list]
        self.dataset_root = [f'{dataset_dir}/{dataset_name}' for dataset_name in dataset_list]
        self.video_path = self.dataset_root

        # balance the dataset
        self.prob = args.prob
        self.sequence_length = args.sequence_length

        # prepare sample information
        self.samples = []
        self.ann_files = []
        self.samples_num = []
        for dataset_idx, data_path in enumerate(self.data_path):
            if 'calvin' in data_path:
                # train on calvin abc datasets and val on d datasets 
                data_path = data_path.replace('annotation','annotation_abc_13')
                data_path = data_path.replace('val','val_d')

            ann_files = self._init_anns(data_path)

            # for quick debug
            if args.debug:
                ann_files = ann_files[:50]
            
            samples = self._init_sequences(ann_files)

            # for quick validation
            if mode == 'val':
                samples = samples
            
            # gather all sample_info
            self.ann_files.append(ann_files)
            self.samples.append(samples)
            self.samples_num.append(len(samples))
            print(f'dataset_idx: {dataset_idx}, {data_path}')
            print(f'dataset_idx: {dataset_idx}, mode:{mode}, trajectories_num:{len(ann_files)}, sample_num:{len(samples)},')
        
        # print(f"ALL dataset, {sum(self.samples_num)} samples in total")

        if self.args.tie_weight:
            self.samples_all = []
            self.video_path_all = [] 
            for i, samples in enumerate(self.samples):
                self.samples_all += samples
                self.video_path_all += [self.video_path[i]]*len(samples)
            self.samples = self.samples_all
            self.video_path = self.video_path_all
            # random shuffle
            self.idx = np.random.permutation(len(self.samples))
            self.samples = [self.samples[i] for i in self.idx]
            self.video_path = [self.video_path[i] for i in self.idx]
            print(f'{mode} dataset, after tie, len:',len(self.samples),len(self.video_path))

        # load statistics information
        with open(args.statistics_path, 'r') as f:
            self.stat = json.load(f)

        # image normalization tools
        # self.transform = T.Compose([
        #     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        # ])
        # self.preprocess = T.Compose([
        #     ToTensorVideo(),
        #     Resize_Preprocess(tuple(args.video_size)), # 288 512
        #     T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        # ])
        # self.not_norm_preprocess = T.Compose([
        #     ToTensorVideo(),
        #     Resize_Preprocess(tuple(args.video_size))
        # ])
        

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

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
        start_interval = 1
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

        # else: # we directly resize internet video for 1 clip

        #     # filter out very long video clips
        #     if n_frames>100:
        #         return samples

        #     sample = dict()
        #     sample['ann_file'] = ann_file
            
        #     # directly resize internet video to 16 frames
        #     idx = np.linspace(0,n_frames-1,self.sequence_length).astype(int)
        #     sample['frame_ids'] = idx.tolist()
        #     samples.append(sample)


        return samples

    def __len__(self):
        if self.args.tie_weight:
            return len(self.samples)
        else:
            return max(self.samples_num)

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy() #(frame, h, w, c)
        # central crop
        h, w = frame_data.shape[1], frame_data.shape[2]
        if h > w:
            margin = (h - w) // 2
            frame_data = frame_data[:, margin:margin + w]
        elif w > h:
            margin = (w - h) // 2
            frame_data = frame_data[:, :, margin:margin + h]
        return frame_data
    
    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path,'rb') as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
            assert (np.array(frame_ids) < video_tensor.size()[0]).all()
            assert (np.array(frame_ids) >= 0).all()
        frame_data = video_tensor[frame_ids]
        return frame_data
    
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
        
        elif self.args.direct_imgs:
            # /localssd/gyj/data1224/opensource_robotdata/xhand_1206_suction_v4/videos/val/50/0.mp4
            # video_path = label['videos'][cam_id]['video_path']
            # video_path = os.path.join(video_dir,video_path)
            # img_path = video_path.replace('videos','imgs')
            # img_path = img_path.replace('rgb.mp4',f'{cam_id}')
            img_path = label['videos'][cam_id]['img_path']
            img_path = os.path.join(video_dir,img_path)
            frames = []
            # "videos/val/7/rgb.mp4"
            # /localssd/gyj/opensource_robotdata/bridge/imgs/val/7/0/2.jpg
            for frame_id in frame_ids:
                frame_path = os.path.join(img_path,f'{frame_id}.jpg')
                frame = imageio.v2.imread(frame_path)
                frames.append(frame)
            frames = np.stack(frames)
            frames = torch.from_numpy(frames)
            # frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (l, c, h, w)
            # frames = self.preprocess(frames)
        # load original videos
        else: 
            video_path = label['videos'][cam_id]['video_path']
            video_path = os.path.join(video_dir,video_path)
            frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (l, c, h, w)

            if self.args.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames*255.0,0,255).to(torch.uint8)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id = temp_cam_id, pre_encode = pre_encode, video_dir=video_dir)
        return frames, temp_cam_id

    def process_action_xhand(self, label,frame_ids, rel = False):
        num_frames = len(frame_ids)
        states = np.array(label['states'])[frame_ids]
        command = np.array(label['actions'])[frame_ids]

        state_input = states[0:1] #(1,19)
        # always use the set the first item of quat >0
        if state_input[0,3] <0:
            state_input[0,3:7] *= -1 

        states_raw = states if not self.args.learn_command else command

        if not rel:
            state_next = states_raw[:-1] # command
            mu = np.array(self.args.mu)
            std = np.array(self.args.std)

            state_input = (state_input-mu)/std
            action_sclaed = (state_next-mu)/std
        else: # relative ro fiest frame
            xyz, rot, hand = states_raw[:,:3], states_raw[:,3:7], states_raw[:,7:]
            # xyz
            current_xyz = state_input[:,:3]
            delta_xyz = (xyz[:-1]-current_xyz)*self.args.rel_xyz_scale
            # rot
            current_quat = state_input[:,3:7]
            rotm = [R.from_quat(rot[i]).as_matrix() for i in range(len(rot-1))]
            current_rotm = R.from_quat(current_quat[0]).as_matrix()
            rel_rotm = [current_rotm.T @ next_rotm for next_rotm in rotm[:-1]]
            rel_rpy = [R.from_matrix(rot).as_euler('xyz', degrees=False) for rot in rel_rotm]

            rel_rpy = np.array(rel_rpy)*self.args.rel_rot_scale
            # hand
            hand = hand[:-1]*self.args.rel_hand_scale

            action_sclaed = np.concatenate([delta_xyz,rel_rpy,hand],axis=1) # (10,18)

            if self.args.norm_input:
                mu = np.array(self.args.mu)
                std = np.array(self.args.std)
                state_input = (state_input-mu)/std

        return torch.from_numpy(action_sclaed).float(), torch.from_numpy(state_input).float()
    
    def process_action_bridge(self, label,frame_ids, rel = False):
        def normalize_bound(
            data: np.ndarray,
            data_min: np.ndarray,
            data_max: np.ndarray,
            clip_min: float = -1,
            clip_max: float = 1,
            eps: float = 1e-8,
        ) -> np.ndarray:
            ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
            return np.clip(ndata, clip_min, clip_max)
        
        num_frames = len(frame_ids)
        proprio = np.array(label['state'])[frame_ids]
        action = np.array(label['action'])[frame_ids]
        proprio = proprio[0:1] #(1,19)

        # normalize proprio and action
        proprio = normalize_bound(proprio, np.array(self.stat['proprio']['p01']), np.array(self.stat['proprio']['p99']))
        action = normalize_bound(action, np.array(self.stat['action']['p01']), np.array(self.stat['action']['p99']))

        return torch.from_numpy(action).float(), torch.from_numpy(proprio).float()

    def __getitem__(self, index):

        if self.args.tie_weight:
            sample = self.samples[index]
            sampled_video_dir = self.video_path[index]
        else:
            # sample the ann_file
            dataset_idx = np.random.choice(len(self.samples), p=self.args.prob)
            sampled_dataset = self.samples[dataset_idx]
            sampled_video_dir = self.dataset_root[dataset_idx]

            sampled_idx = index % len(sampled_dataset)
            sample = sampled_dataset[sampled_idx]

        ann_file = sample['ann_file']
        # print(index, ann_file)
        frame_ids = sample['frame_ids']
        with open(ann_file, "r") as f:
            label = json.load(f)
        
        # prepare sample 

        curr_idx = frame_ids[0]
        future_idx = frame_ids[-1]
        video1, _ = self._get_obs(label, [curr_idx], cam_id=0, pre_encode=False, video_dir=sampled_video_dir)
        # video2, _ = self._get_obs(label, [curr_idx], cam_id=1, pre_encode=False, video_dir=sampled_video_dir)
        # print(video1.shape,video2.shape)
        # rgb_obs = {'cond_static':video1[0], 'cond_gripper':video1[0], 'gen_static':video2[0], 'gen_gripper':video2[1]} 
        action, proprio = self.process_action_bridge(label,frame_ids,rel=self.args.relative)

        data = dict()
        data['lang_text'] = label['texts'][0] # text for condition
        obs = {'image_primary':video1,'proprio':proprio} 
        data['observation'] = obs
        data['action'] = action
        

        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/dataset_args.yaml")
    args = parser.parse_args()
    data_config = OmegaConf.load('training/dataset_args.yaml')
    # diffusion_config = OmegaConf.load("configs/base/diffusion.yaml")
    args = OmegaConf.load(args.config)
    args = OmegaConf.merge(data_config.train_args, args)
    # args = OmegaConf.merge(diffusion_config, args)
    # update_paths(args)
    dataset = Dataset_Real(args,mode='train')

    data_loader = DataLoader(dataset=dataset, 
                                    batch_size=16, 
                                    shuffle=False, 
                                    num_workers=16)
    for data in tqdm(data_loader,total=len(data_loader)):
        print(data['rgb_obs']['cond_static'].shape)
        print(data['actions'].shape)
        pass