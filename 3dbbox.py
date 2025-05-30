import open3d as o3d
import numpy as np 
import os
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Viewpoint(object):
    def __init__(self, name, w, h, intrinsics, pose):
        super(Viewpoint, self).__init__()
        self.name = name
        self.w = w
        self.h = h
        self.intrinsics = intrinsics
        self.pose = pose


def add_view_o3d(w, viewpoint, color=[1.0, 0.0, 0.0]):
    v = o3d.geometry.LineSet.create_camera_visualization(
        viewpoint.w, viewpoint.h, 
        viewpoint.intrinsics, viewpoint.pose, 
        scale=0.5) 
    v.paint_uniform_color(color)
    w.add_geometry(viewpoint.name, v)
    return w


def visualize_views(viewpoints=[], points=None, colors=None, arrow = None, bboxs = []):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    w = o3d.visualization.O3DVisualizer(width=1024)
    w.show_ground = True
    w.show_axes = True

    for viewpoint in viewpoints:
        add_view_o3d(w, viewpoint, color=[0., 1., 0.])
    
    if not points is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        w.add_geometry("rgb_pointcloud", pcd)
    if not arrow is None:
        w.add_geometry("arrow", arrow)
    # if not bbox is None:
    #     w.add_geometry("bbox", bbox)
    for i, bbox in enumerate(bboxs):
        name = "bbox_" + str(i)
        w.add_geometry(name, bbox)

    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    # w.enable_raw_mode(True)
    w.show_skybox(False)
    app.add_window(w)
    app.run()

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
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.linalg.inv(np.array(camera['static_cam_viewMatrix']).reshape((4, 4)).T)
    z = depth_img[v, u]
    foc = camera['static_cam_height'] / (2 * np.tan(camera['static_cam_fov'] / 2))
    x = (u - camera['static_cam_width'] // 2) * z / foc
    y = -(v - camera['static_cam_height'] // 2) * z / foc
    z = -z
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    # Sanity check by using camera.deproject function.  Check 2000 points.
    if sanity_check:
        sample_inds = np.random.permutation(u.shape[0])[:2000]
        for ind in sample_inds:
            cam_world_pos = cam.deproject((u[ind], v[ind]), depth_img, homogeneous=True)
            assert np.abs(cam_world_pos-world_pos[:, ind]).max() <= 1e-3

    if not homogeneous:
        world_pos = world_pos[:3]

    return world_pos

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

def get_bbox(center, R ,size):
    dx, dy, dz = size / 2
    local_corners = np.array([
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
    ])
    bbox_points = (R @ local_corners.T).T + center
    lines = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
    [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
    [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # 红色
    return line_set

# 旋转并平移到全局坐标

if __name__ == '__main__':

    camera = np.load('camera.npy', allow_pickle=True).item()
    data = np.load(f'data/episode_0039801.npz')
    # annotations = np.load(
    #     f'auto_lang_ann.npy',
    #     allow_pickle=True
    # ).item()
    #text = annotations['language']['task'][anno_ind]
    #print(annotations['language']['task'][0:10])

    rgb_static = data['rgb_static']
    print('rgb_static_shape:',rgb_static.shape)
    depth_static = data['depth_static']
    #rgb_static = cv2.resize(rgb_static, (200, 200), interpolation=cv2.INTER_LINEAR)
    rgb_points = rgb_static.reshape(-1, 3)/255

    xyz = deproject(camera, depth_static)
    viewpoints = []
    image_width, image_height = 64, 64
    #intrinsics = np.array([[119.4256, 0, 32],[0, 119.4256, 32],[0, 0, 1]], dtype=float)

    objects = ['red','blue','pink']
    bboxs = []
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
        print('euler_x:',euler_x)
        print('euler_y:',euler_y)
        print('euler_z:',euler_z)

        #viewpoints.append(Viewpoint('{}'.format(1), image_width, image_height, intrinsics, view_cam))
        origin = np.array([T[0,3], T[1,3], T[2,3]])      # 箭头起点
        print('origin:',origin)
        length = 0.5
        # arrow = o3d.geometry.TriangleMesh.create_arrow(
        #     cylinder_radius=0.01,
        #     cone_radius=0.02,
        #     cylinder_height=length * 0.8,
        #     cone_height=length * 0.2
        # )
        # arrow.rotate(R, center=(0, 0, 0))
        # arrow.translate(origin)

        size = np.array([0.12, 0.12, 0.12])
        bbox = get_bbox(origin, R, size)

        uv = cam2img(origin, camera)
        #uv = uv / camera['static_cam_width']

        rgb_show = rgb_static.astype(np.uint8)
        #rgb_show  = cv2.cvtColor(rgb_show, cv2.COLOR_RGB2BGR)

        uv_int = tuple(np.round(uv).astype(int))
        #cv2.circle(rgb_show, uv_int, radius=5, color=(0,0,255), thickness=-1)
        uv_list.append(uv)
        bboxs.append(bbox)
        #plt.imshow(cv2.cvtColor(rgb_show, cv2.COLOR_BGR2RGB))
    # plt.imshow(rgb_show)
    # plt.scatter([uv_list[0][0]], [uv_list[0][1]], c='red', s=50)
    # plt.scatter([uv_list[1][0]], [uv_list[1][1]], c='blue', s=50)
    # plt.scatter([uv_list[2][0]], [uv_list[2][1]], c='pink', s=50)
    # plt.show()

    #visualize_views(viewpoints, xyz.T, rgb_points, None, bboxs)
    device = 'cuda:0'
    sam2_checkpoint = "/cephfs/cjyao/code/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)

    predictor.set_image(rgb_show)

    for i,uv in enumerate(uv_list):
        input_point = np.array([[uv[0], uv[1]]])
        input_label = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # 创建彩色mask图像
        mask = masks[0].astype(np.bool_)  # 使用astype而不是to
        mask_vis = np.zeros_like(rgb_show)
        mask_vis[mask] = [0, 255, 0]  # 使用绿色显示mask区域
        
        # 将原始图像和mask叠加
        result = cv2.addWeighted(rgb_show, 0.5, mask_vis, 0.5, 0)
        
        # 保存结果
        cv2.imwrite(f'save/mask_{i}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"保存mask.png")

    cv2.imwrite(f'save/input.png', cv2.cvtColor(rgb_show, cv2.COLOR_RGB2BGR))
    print(f"保存input.png")


