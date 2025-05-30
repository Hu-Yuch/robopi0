import os
import numpy as np
from decord import VideoReader, cpu
import mediapy
import json
import torch
def load_video(video_path, frame_ids):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    assert (np.array(frame_ids) < len(vr)).all()
    assert (np.array(frame_ids) >= 0).all()
    vr.seek(0)
    frame_data = vr.get_batch(frame_ids).asnumpy() #(frame, h, w, c)
    # central crop
    # h, w = frame_data.shape[1], frame_data.shape[2]
    # if h > w:
    #     margin = (h - w) // 2
    #     frame_data = frame_data[:, margin:margin + w]
    # elif w > h:
    #     margin = (w - h) // 2
    #     frame_data = frame_data[:, :, margin:margin + h]
    return frame_data


path = '/localssd/gyj/opensource_robotdata'

dataset_paths = os.listdir(path)
dataset_paths = [os.path.join(path, i) for i in dataset_paths]
# print(dataset_paths)
dataset_paths = ['/localssd/gyj/opensource_robotdata/rt1']
# python test.py
print(dataset_paths)
for dataset_path in dataset_paths:
    os.makedirs(os.path.join(dataset_path, 'imgs/train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'imgs/val'), exist_ok=True)
    types = ['val', 'train']

    for type in types:
        trajs = os.listdir(os.path.join(dataset_path, 'annotation', type)) #'/localssd/gyj/data1224/opensource_robotdata/xhand_1025_v2/videos/train/1'
        
        for num, traj_json in enumerate(trajs):
            traj = traj_json.split('.')[0]
            anno = os.path.join(dataset_path, 'annotation', f'{type}/{traj}.json')
            print(f'{num}/{len(trajs)}',anno)
            videos = os.listdir(os.path.join(dataset_path, 'videos', type, traj))
            # load anno 
            with open(anno, 'r') as f:
                anno = json.load(f)
            length = len(anno['action'])
            for cam_i in range(1):
                # load video
                video_path = os.path.join(dataset_path, 'videos', type, traj, f'rgb.mp4')
                frames = load_video(video_path, range(length))
                frames = np.array(frames)
                frames = torch.tensor(frames)
                # print(frames.shape)
                # frames = frames.to('cuda:0')
                # frames = frames.to('cuda') if torch.cuda.is_available() else frames  # 确保 CUDA 可用
                # print(torch.cuda.is_available())

                # check if cuda is available
                # if torch.cuda.is_available():
                #     frames = frames.cuda()
                
                # resize to 224
                frames = torch.nn.functional.interpolate(frames.permute(0,3,1,2).float(), size=(224,224), mode='bilinear').to(torch.uint8).permute(0,2,3,1)
                print(frames.shape)
                frames = frames.cpu().numpy()
                # save frames
                os.makedirs(os.path.join(dataset_path, 'imgs', type, traj, str(cam_i)), exist_ok=True)
                for i, frame in enumerate(frames):
                    mediapy.write_image(os.path.join(dataset_path, 'imgs', type, traj, str(cam_i), f'{i}.jpg'), frame)



