from typing import List, Optional
from pathlib import Path
import os
import pickle
import open3d as o3d
import tap
import cv2
import numpy as np
import torch
import blosc
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
#from calvin_env.envs.play_table_env import get_env

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.utils_with_calvin import (
#     keypoint_discovery,
#     #deproject,
#     get_gripper_camera_view_matrix,
#     convert_rotation
# )


class Arguments(tap.Tap):
    traj_len: int = 10
    execute_every: int = 5
    save_path: str = '/cephfs/shared/hyc/data/calvin/package'
    root_dir: str = '/cephfs/shared/hyc/data/calvin/task_ABC_D'
    mode: str = 'close_loop'  # [keypose, close_loop]
    tasks: Optional[List[str]] = None
    split: str = 'training'  # [training, validation]
    cuda: int = 0

def cam2img(xyz, camera):
    fx = fy = 1143
    cx = cy = 10
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]])  # 3x3 内参
    Tcw = camera['static_cam_viewMatrix'].reshape((4, 4)).T  # 4x4 外参（世界到相机），实际请替换为你的矩阵
    P_world = np.array([xyz[0], xyz[1], xyz[2], 1])  # 齐次坐标

    # 1. 世界坐标转相机坐标
    P_cam = Tcw @ P_world  # shape: (4,)
    Xc, Yc, Zc = P_cam[:3]
    Zc = -Zc

    # 2. 相机坐标归一化
    x = Xc / Zc
    y = -Yc / Zc

    u = x*fx + camera['static_cam_width']//2
    v = y*fy + camera['static_cam_width']//2
    uv = np.array([u,v])
    # 3. 投影到像素
    #uv1 = K @ np.array([x, y, 1])
    #uv1[0:2] = uv1[0:2] + camera['static_cam_width']//2

    return uv

def process_datas(datas, mode, traj_len, execute_every, predictor):
    """Fetch and drop datas to make a trajectory

    Args:
        datas: a dict of the datas to be saved/loaded
            - static_pcd: a list of nd.arrays with shape (height, width, 3)
            - static_rgb: a list of nd.arrays with shape (height, width, 3)
            - gripper_pcd: a list of nd.arrays with shape (height, width, 3)
            - gripper_rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (7,)
        mode: a string of [keypose, close_loop]
        traj_len: an int of the length of the trajectory
        execute_every: an int of execution frequency
        keyframe_inds: an Integer array with shape (num_keyframes,)

    Returns:
        the episode item: [
            [frame_ids],
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
            [annotation_ind] # wrt frame_ids, (1,)
        ]
    """
    # upscale gripper camera
    h, w = datas['static_rgb'][0].shape[:2]
    datas['static_rgb'] = datas['static_rgb'][::args.execute_every]
    datas['gripper_rgb'] = datas['gripper_rgb'][::args.execute_every]
    datas['red_mask'] = datas['red_mask'][::args.execute_every]
    datas['blue_mask'] = datas['blue_mask'][::args.execute_every]
    datas['pink_mask'] = datas['pink_mask'][::args.execute_every]
    datas['gripper_depth'] = datas['gripper_depth'][::args.execute_every]
    datas['static_depth'] = datas['static_depth'][::args.execute_every]
    datas['proprios'] = datas['proprios'][::args.execute_every]
    datas['scene_obs'] = datas['scene_obs'][::args.execute_every]
    datas['annotation_id'] = datas['annotation_id'][::args.execute_every]

    static_rgb = np.stack(datas['static_rgb'], axis=0) # (traj_len, H, W, 3)
    gripper_rgb = np.stack(datas['gripper_rgb'], axis=0) # (traj_len, H, W, 3)

    red_mask = compute_mask(datas['static_rgb'], datas['red_mask'], predictor)
    blue_mask = compute_mask(datas['static_rgb'], datas['blue_mask'], predictor)
    pink_mask = compute_mask(datas['static_rgb'], datas['pink_mask'], predictor)

    assert len(red_mask) == len(datas['static_rgb'])

    red_mask = np.stack(red_mask, axis=0)
    blue_mask = np.stack(blue_mask, axis=0)
    pink_mask = np.stack(pink_mask, axis=0)

    depth_gripper = np.stack(datas['gripper_depth'], axis=0)
    depth_static = np.stack(datas['static_depth'], axis=0)

    scene_obs = np.stack(datas['scene_obs'], axis=0)

    gripper_tensors = [
        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
        for a in datas['proprios']
    ]
    # prepare frame_ids
    frame_ids = [i for i in range(len(static_rgb))]

    # Save everything to disk
    state_dict = [
        frame_ids,
        static_rgb,
        gripper_rgb,
        depth_static,
        depth_gripper,
        red_mask,
        blue_mask,
        pink_mask,
        scene_obs,
        gripper_tensors,
        datas['annotation_id'],

    ]

    return state_dict

def compute_mask(img, uv, predictor):
    mask_list = []
    bs = 30
    lenth = len(uv)
    j = 0
    while j*bs < lenth:
        top = min((j+1)*bs, lenth)
        point_batch = []
        label_batch = []
        img_batch = []
        for i in range(j*bs, top):
            input_point = np.array([[uv[i][0], uv[i][1]]])
            input_label = np.array([1])

            point_batch.append(input_point)
            label_batch.append(input_label)
            img_batch.append(img[i].astype(np.uint8))

        predictor.set_image_batch(img_batch)
        masks_batch, scores_batch, _ = predictor.predict_batch(
            point_batch,
            label_batch,
            multimask_output=True,
        )
        #print('masks_batch_shape:',masks_batch[0].shape)
        for masks, scores in zip(masks_batch,scores_batch):
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            mask_list.append(masks[0])
        j = j + 1

    return mask_list

def load_episode(root_dir, split, episode, datas, ann_id, camera, predictor):
    """Load episode and process datas

    Args:
        root_dir: a string of the root directory of the dataset
        split: a string of the split of the dataset
        episode: a string of the episode name
        datas: a dict of the datas to be saved/loaded
            - static_pcd: a list of nd.arrays with shape (height, width, 3)
            - static_rgb: a list of nd.arrays with shape (height, width, 3)
            - gripper_pcd: a list of nd.arrays with shape (height, width, 3)
            - gripper_rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (8,)
            - annotation_id: a list of ints
    """
    data = np.load(f'{root_dir}/{split}/{episode}')

    rgb_static = data['rgb_static']  # (200, 200, 3)
    rgb_gripper = data['rgb_gripper']  # (84, 84, 3)
    depth_static = data['depth_static']  # (200, 200)
    depth_gripper = data['depth_gripper']  # (84, 84)
    proprio = data['robot_obs']
    scene_obs = data['scene_obs']

    objects = ['red','blue','pink']
    uv_list = []
    for object in objects:
        if object  == 'blue':
            euler_x, euler_y, euler_z = data['scene_obs'][9+6], data['scene_obs'][10+6], data['scene_obs'][11+6]
            R = o3d.geometry.get_rotation_matrix_from_xyz([euler_x, euler_y, euler_z])
            T = np.eye(4)  # 先创建4x4单位阵
            T[:3, :3] = R  # 
            T[0,3] = data['scene_obs'][6+6]
            T[1,3] = data['scene_obs'][7+6]
            T[2,3] = data['scene_obs'][8+6]
        elif object  == 'red':
            euler_x, euler_y, euler_z = data['scene_obs'][9], data['scene_obs'][10], data['scene_obs'][11]
            R = o3d.geometry.get_rotation_matrix_from_xyz([euler_x, euler_y, euler_z])
            T = np.eye(4)  # 先创建4x4单位阵
            T[:3, :3] = R  # 
            T[0,3] = data['scene_obs'][6]
            T[1,3] = data['scene_obs'][7]
            T[2,3] = data['scene_obs'][8]
        else:
            euler_x, euler_y, euler_z = data['scene_obs'][9+12], data['scene_obs'][10+12], data['scene_obs'][11+12]
            R = o3d.geometry.get_rotation_matrix_from_xyz([euler_x, euler_y, euler_z])
            T = np.eye(4)  # 先创建4x4单位阵
            T[:3, :3] = R  # 
            T[0,3] = data['scene_obs'][6+12]
            T[1,3] = data['scene_obs'][7+12]
            T[2,3] = data['scene_obs'][8+12]

        view_cam = camera['static_cam_viewMatrix'].reshape((4, 4)).T
        #viewpoints.append(Viewpoint('{}'.format(1), image_width, image_height, intrinsics, view_cam))
        origin = np.array([T[0,3], T[1,3], T[2,3]])      # 箭头起点

        uv = cam2img(origin, camera)
        #uv = uv / camera['static_cam_width']
        uv_int = tuple(np.round(uv).astype(int))
        #cv2.circle(rgb_show, uv_int, radius=5, color=(0,0,255), thickness=-1)
        uv_list.append(uv)
    # Put them into a dict

    datas['static_depth'].append(depth_static)  # (200, 200, 3)
    datas['static_rgb'].append(rgb_static)  # (200, 200, 3)
    datas['gripper_depth'].append(depth_gripper)  # (84, 84, 3)
    datas['gripper_rgb'].append(rgb_gripper)  # (84, 84, 3)
    datas['proprios'].append(proprio)  # (8,)
    datas['annotation_id'].append(ann_id)  # int
    datas['red_mask'].append(uv_list[0])
    datas['blue_mask'].append(uv_list[1])
    datas['pink_mask'].append(uv_list[2])
    datas['scene_obs'].append(scene_obs)

def init_datas():
    datas = {
        'red_mask': [],
        'blue_mask': [],
        'pink_mask': [],
        'static_rgb': [],
        'static_depth': [],
        'gripper_rgb': [],
        'gripper_depth': [],
        'annotation_id':[],
        'proprios':[],
        'scene_obs':[],
    }
    return datas


def main(split, args):
    """
    CALVIN contains long videos of "tasks" executed in order
    with noisy transitions between them. The 'annotations' json contains
    info on how to segment those videos.

    Original CALVIN annotations:
    {
        'info': {
            'episodes': [],
            'indx': [(788072, 788136), (899273, 899337), (1427083, 1427147)]
                list of tuples indicating start-end of a task
        },
        'language': {
            'ann': list of str with len=17870, instructions,
            'task': list of str with len=17870, task names,
            'emb': array (17870, 1, 384)
        }
    }

    Save:
    state_dict = [
        frame_ids,  # [0, 1, 2...]
        rgb_pcd,  # tensor [len(frame_ids), ncam, 2, 3, 200, 200]
        action_tensors,  # [tensor(1, 8)]
        camera_dicts,  # [{'front': (0, 0), 'wrist': (0, 0)}]
        gripper_tensors,  # [tensor(1, 8)]
        trajectories,  # [tensor(N, 8) or tensor(2, 8) if keyposes]
        datas['annotation_id']  # [int]
    ]
    """
    annotations = np.load(
        f'{args.root_dir}/{split}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()

    camera = np.load('camera.npy', allow_pickle=True).item()

    device = 'cuda:'+ str(args.cuda)
    sam2_checkpoint = "/cephfs/cjyao/code/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)

    for anno_ind, (start_id, end_id) in enumerate(annotations['info']['indx']):
        # Step 1. load episodes of the same task
        len_anno = len(annotations['info']['indx'])
        #if args.tasks is not None and annotations['language']['task'][anno_ind] not in args.tasks:
        #    continue
        datas = init_datas()

        if split == 'training':
            scene_info = np.load(
                f'{args.root_dir}/training/scene_info.npy',
                allow_pickle=True
            ).item()
            if ("calvin_scene_B" in scene_info and
                start_id <= scene_info["calvin_scene_B"][1]):
                scene = "B"
            elif ("calvin_scene_C" in scene_info and
                  start_id <= scene_info["calvin_scene_C"][1]):
                scene = "C"
            elif ("calvin_scene_A" in scene_info and
                  start_id <= scene_info["calvin_scene_A"][1]):
                scene = "A"
            else:
                scene = "D"
        else:
            scene = 'D'

        if args.tasks is not None and not scene in args.tasks:
            continue

        print(f'Processing {anno_ind}/{len_anno}, start_id:{start_id}, end_id:{end_id}')

        for ep_id in range(start_id, end_id + 1):
            episode = 'episode_{:07d}.npz'.format(ep_id)
            load_episode(
                args.root_dir,
                split,
                episode,
                datas,
                anno_ind,
                camera,
                predictor
            )

        # Step 2. detect keyframes within the episode

        state_dict = process_datas(
            datas, args.mode, args.traj_len, args.execute_every, predictor
        )

        # Step 4. save to .dat file
        ep_save_path = f'{args.save_path}/{split}/{scene}+0/ann_{anno_ind}.dat'
        os.makedirs(os.path.dirname(ep_save_path), exist_ok=True)
        with open(ep_save_path, "wb") as f:
            f.write(blosc.compress(pickle.dumps(state_dict)))


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args.split, args)
