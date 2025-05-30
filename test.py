import os
import numpy as np
# from decord import VideoReader, cpu
import json
import torch
import torch.nn.functional as F
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
#from sam2.build_sam import build_sam2
#from sam2.sam2_image_predictor import SAM2ImagePredictor
import blosc
import pickle
# start load /localssd/gyj/opensource_robotdata/bridge/annotation/val/2.json

data = np.load('/cephfs/cjyao/data/calvin/episode_0021219.npz')

rgb_static = data['rgb_static']  # (200, 200, 3)
rgb_gripper = data['rgb_gripper']  # (84, 84, 3)
depth_static = data['depth_static']  # (200, 200)
depth_gripper = data['depth_gripper']  # (84, 84)

camera = np.load('/cephfs/shared/hyc/data/calvin/task_ABC_D/camera.npy', allow_pickle=True).item()

def deproject(camera, depth_img, homogeneous=False, sanity_check=False):
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
        device = depth_img.device
        b, c, h, w = depth_img.shape
        new_h, new_w = 16*4, 16*4

        new_depth = F.interpolate(depth_img, size=(new_h, new_w), mode="area")

        u, v = np.meshgrid(np.arange(new_h), np.arange(new_w))
        u, v = u.ravel(), v.ravel()


        # Unproject to world coordinates
        T_world_cam = np.linalg.inv(np.array(camera['static_cam_viewMatrix']).reshape((4, 4)).T)
        T_world_cam = torch.from_numpy(T_world_cam).to(device).unsqueeze(0)
        z = new_depth[:, 0, v, u]
        u = torch.from_numpy(u).to(device)
        v = torch.from_numpy(v).to(device)
        u = u.unsqueeze(0).repeat(b,1)
        v = v.unsqueeze(0).repeat(b,1)
        foc = torch.tensor(camera['static_cam_height']).to(device) / (2 * torch.tan(torch.tensor(camera['static_cam_fov']).to(device) / 2))
        x = (u*h/new_h - torch.tensor(camera['static_cam_width']).to(device) // 2) * z / foc
        y = -(v*w/new_w - torch.tensor(camera['static_cam_height']).to(device) // 2) * z / foc
        z = -z
        ones = torch.ones_like(z).to(device)
        print('x_shape:',x.shape)
        print('y_shape:',y.shape)
        print('z_shape:',z.shape)
        cam_pos = torch.stack([x, y, z, ones], axis=1)
        T_world_cam = T_world_cam.to(cam_pos.dtype)
        world_pos = T_world_cam @ cam_pos
        print('world_pos:', world_pos.shape)

        if not homogeneous:
            world_pos = world_pos[0,:3]

        return world_pos

depth = torch.from_numpy(depth_static)
depth = depth.unsqueeze(0)
depth = depth.unsqueeze(0)

xyz = deproject(camera, depth)

img = torch.from_numpy(rgb_static).unsqueeze(0)
img = img.permute(0,3,1,2)
img = F.interpolate(img, size=(64, 64), mode="bilinear")

save_img = rgb_static.astype(np.uint8)
save_img = Image.fromarray(save_img)
save_img.save(f'pointcloud/image_0.png')
        # print("img", img.shape)
        # img = (img * 255).astype(np.uint8)  # 转换为0-255范围
        # img = Image.fromarray(img)
        # img.save(f'pointcloud/image_{i}.png')
        # print(f"保存图片到: pointcloud/image_{i}.png")
img = img.permute(0,2,3,1)
pixel_values = img.reshape(img.shape[0], -1 ,3)/255

# 创建点云对象
pcd = o3d.geometry.PointCloud()
#print("xyz", xyz.shape)
# 设置点云坐标
pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy().T)
# 设置点云颜色
pcd.colors = o3d.utility.Vector3dVector(pixel_values[0].cpu().numpy())

# 保存点云
save_path = f'pointcloud/point_cloud_0.ply'
o3d.io.write_point_cloud(save_path, pcd)

file = '/cephfs/shared/hyc/data/calvin/package/training/A+0/ann_16.dat'
with open(file, "rb") as f:
    content = pickle.loads(blosc.decompress(f.read()))
static_rgb = content[1]
red_mask = content[5]
blue_mask = content[6]

red_mask = red_mask.astype(bool)
blue_mask = blue_mask.astype(bool)

# 创建一个红色遮罩层
mask_overlay = np.zeros_like(static_rgb)
mask_overlay[blue_mask] = [255, 0, 0]  # 红色

# 将原图与遮罩层按0.5的透明度混合
result = cv2.addWeighted(static_rgb, 1, mask_overlay, 0.5, 0)
print('static_rgb_shape:',static_rgb.shape)
print('result_shape:',result.shape)
cv2.imwrite('save/result.png', cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR))
cv2.imwrite('save/input.png', cv2.cvtColor(static_rgb[0], cv2.COLOR_RGB2BGR))

ys, xs = np.where(blue_mask[0])  # 假设blue_mask shape为 (N, H, W)
if len(xs) > 0 and len(ys) > 0:
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    # 在static_rgb[0]上画框
    bbox_img = static_rgb[0].copy()
    # 画矩形，颜色为绿色，线宽2
    cv2.rectangle(bbox_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    # 保存带框图片
    cv2.imwrite('save/input_with_bbox.png', cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))

# 显示结果
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
path = '/cephfs/shared/hyc/data/calvin/task_ABC_D/training/lang_annotations/auto_lang_ann.npy'

annotations = np.load(
            path,
            allow_pickle=True
            ).item()
lang_text = annotations["language"]["ann"]
for i in range(100):
    print('instru_:',lang_text[i])

