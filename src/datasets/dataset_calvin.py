from collections import defaultdict, Counter
import itertools
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CalvinDataset(Dataset):

    def __init__(
        self,
        # required
        root,
        instructions_path=None,
        split = 'training'
        # dataset specification
    ):

        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root = [Path(r).expanduser() for r in root]
        self._root = [r / Path(split) for r in root]
        self._relative_action = relative_action
        
        self.annotations = np.load(
            instructions_path.as_posix()+'/lang_annotations/auto_lang_ann.npy',
            allow_pickle=True
        ).item()
        
        self.file_id = []
        self.file_id.extend([str(p) for p in self._root[0].rglob("*.dat")])
        #for anno_ind, (start_id, end_id) in enumerate(self.annotations['info']['indx']):
            
        print(f"Created dataset from {self._root[0]}")

    def __len__(self):
        return len(self.file_id)

    def __getitem__(self, episode_id):
        """
        the episode item: [
            [frame_ids],  # we use chunk and max_episode_length to index it
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
        ]
        """
        file = self.file_id[episode_id]

        # Load episode
        try:
            with open(file, "rb") as f:
                episode = pickle.loads(blosc.decompress(f.read()))
        except UnpicklingError as e:
            print(f"Can't load {file}: {e}")

        episode_len = len(episode[0])
        # Dynamic chunking so as not to overload GPU memory
        idx = random.randint(
            0, episode_len - 1
        )

        annotation_id = episode[-1][0]
        language = lang_data["language"]["ann"][annotation_id]
        def get_bbox_from_mask(mask):
            # 找到mask中所有True值的    坐标
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                return None
            # 计算边界框坐标
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            return [x_min, y_min, x_max, y_max]  # [x1, y1, x2, y2]格式
        w = episode[idx][1].shape[3]
        h = episode[idx][1].shape[2]
        data = dict()
        data['lang_text'] = language
        data['image'] = episode[idx][1]
        data['depth'] = episode[idx][3]
        red_mask = episode[idx][5]
        blue_mask = episode[idx][6]
        pink_mask = episode[idx][7]
        colors = ['red' ,'blue' ,'pink']
        bbox = {}
        bbox['red'] = get_bbox_from_mask(red_mask)
        bbox['blue'] = get_bbox_from_mask(blue_mask)
        bbox['pink'] = get_bbox_from_mask(pink_mask)

        colors = ['red' ,'blue' ,'pink']
        question = language + ',locate all the colorful blocks in the instruction'+'\n'
        answer = 'answer:'
        for color in colors:
            if color in language:
                xmin_token = int(bbox[color][0]*1000/w)
                xmin_token = '<loc'+str(xmin_token).zfill(4)+'>'
                xmax_token = int(bbox[color][2]*1000/w)
                xmax_token = '<loc'+str(xmax_token).zfill(4)+'>'
                ymin_token = int(bbox[color][1]*1000/h)
                ymin_token = '<loc'+str(ymin_token).zfill(4)+'>'
                ymax_token = int(bbox[color][3]*1000/h)
                ymax_token = '<loc'+str(ymax_token).zfill(4)+'>'
                
                # 构建更规范的VLM文本格式
                vlm_text = f"The {color} block is located at position {xmin_token} {ymin_token} to {xmax_token} {ymax_token} in the image."
                answer = answer + vlm_text

        # 添加总结信息
        
        if len([c for c in colors if c in language]) == 0:
            answer += "No blocks were found in the instruction."

        data['vlm_question'] = question
        data['vlm_answer'] = answer

        return data
