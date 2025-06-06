#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import open3d as o3d
import numpy as np
import torch
import random
import cv2
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye

# Set environment and model
os.environ['KITTI360_DATASET'] = '/home/karan-sankla/LIDAR_RADAR/KITTI360_sample'
model = YOLO('yolo11x-seg.pt')
output_dir = '/home/karan-sankla/LIDAR_RADAR/Predictions'
os.makedirs(output_dir, exist_ok=True)


# ======================
# Data Loader
# ======================
class Kitti360Viewer3DRaw:
    def __init__(self, seq=0, mode='velodyne'):
        base_path = os.environ['KITTI360_DATASET']
        sequence = f'2013_05_28_drive_{seq:04d}_sync'
        sensor_dir = 'velodyne_points' if mode == 'velodyne' else 'sick_points'
        self.raw3DPcdPath = os.path.join(base_path, 'data_3d_raw', sequence, sensor_dir, 'data')

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd, [-1, 4])
        return pcd


# ======================
# Segmentation with YOLO
# ======================
def imagesegmentation(image):
    results = model.predict(image, device='0', classes=2, retina_masks=True)
    for result in results:
        img = result.orig_img.copy()
        boxes = result.boxes.xyxy.cpu().numpy()
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                color = tuple(random.randint(0, 255) for _ in range(3))
                overlay = np.zeros_like(img, dtype=np.uint8)
                overlay[mask > 0.5] = color
                img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)
                x1, y1, x2, y2 = map(int, box)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                y_inds, x_inds = np.where(mask > 0.5)
                if x_inds.size > 0 and y_inds.size > 0:
                    cv2.putText(img, f'Car {i + 1}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return img
    return image  # fallback


# ======================
# Projection & Main Logic
# ======================
def projectVeloToImage(cam_id=0, seq=0):
    kitti360Path = os.environ['KITTI360_DATASET']
    sequence = f'2013_05_28_drive_{seq:04d}_sync'
    velo = Kitti360Viewer3DRaw(seq=seq, mode='velodyne')

    # Load calibration
    TrCam0ToVelo = loadCalibrationRigid(os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt'))
    TrCamToPose = loadCalibrationCameraToPose(os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt'))

    # Choose camera type
    camera = CameraPerspective(kitti360Path, sequence, cam_id) if cam_id in [0, 1] else CameraFisheye(kitti360Path, sequence, cam_id)

    # Compute transform from velodyne to current camera
    TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[f'image_{cam_id:02d}']
    TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
    TrVeloToCam = np.linalg.inv(TrCamToVelo)

    TrVeloToRect = camera.R_rect @ TrVeloToCam if cam_id in [0, 1] else TrVeloToCam

    files = sorted(glob.glob(os.path.join(velo.raw3DPcdPath, '*.bin')))
    for file in files:
        frame = int(os.path.basename(file).split('.')[0])
        points = velo.loadVelodyneData(frame)
        points[:, 3] = 1  # Homogeneous

        # Project points to image
        pointsCam = (TrVeloToRect @ points.T).T[:, :3]
        u, v, depth = camera.cam2image(pointsCam.T)
        u, v = u.astype(int), v.astype(int)

        # Create depth map
        depthMap = np.zeros((camera.height, camera.width))
        mask = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (depth > 0) & (depth < 30)
        depthMap[v[mask], u[mask]] = depth[mask]

        # Load image for segmentation
        sub_dir = 'data_rect' if cam_id in [0, 1] else 'data_rgb'
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, f'image_{cam_id:02d}', sub_dir, f'{frame:010d}.png')
        if not os.path.isfile(imagePath):
            print(f"[!] Skipping missing image {imagePath}")
            continue

        bgr_image = cv2.imread(imagePath)
        img_with_seg = imagesegmentation(bgr_image.copy())

        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        axs[0].imshow(depthMap, cmap='jet')
        axs[0].set_title('Projected Depth')
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(img_with_seg, cv2.COLOR_BGR2RGB))
        axs[1].set_title('YOLO Object Detection')
        axs[1].axis('off')

        plt.suptitle(f'Sequence {seq:04d}, Camera {cam_id:02d}, Frame {frame:010d}')
        plt.tight_layout()
        plt.show()

        # Save result
        save_path = os.path.join(output_dir, f'frame_{frame:010d}.png')
        cv2.imwrite(save_path, img_with_seg)
        print(f"[✓] Saved to {save_path}")

def visualize_pointcloud_with_mask3D(seq=0, frame=0, cam_id=0):
    from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
    from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
    from PIL import Image

    # Setup paths and camera
    kitti360Path = os.environ['KITTI360_DATASET']
    sequence = '2013_05_28_drive_%04d_sync' % seq
    camera = CameraPerspective(kitti360Path, sequence, cam_id) if cam_id in [0, 1] else CameraFisheye(kitti360Path, sequence, cam_id)

    # Load calibrations
    TrCam0ToVelo = loadCalibrationRigid(os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt'))
    TrCamToPose = loadCalibrationCameraToPose(os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt'))
    TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose['image_%02d' % cam_id]
    TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
    TrVeloToCam = np.linalg.inv(TrCamToVelo)
    TrVeloToRect = camera.R_rect @ TrVeloToCam if cam_id in [0, 1] else TrVeloToCam

    # Load point cloud
    viewer = Kitti360Viewer3DRaw(mode='velodyne', seq=seq)
    points = viewer.loadVelodyneData(frame)
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))  # Nx4
    pointsCam = (TrVeloToRect @ points_hom.T).T
    u, v, depth = camera.cam2image(pointsCam[:, :3].T)
    u, v = u.astype(int), v.astype(int)

    # Load image and run segmentation
    image_path = os.path.join(kitti360Path, 'data_2d_raw', sequence, f'image_{cam_id:02d}', 'data_rect' if cam_id in [0, 1] else 'data_rgb', f'{frame:010d}.png')
    color_image = np.array(Image.open(image_path))
    masks = imagesegmentation(color_image)  # Get YOLO masks

    # Match points with mask
    point_colors = np.ones((points.shape[0], 3)) * 0.5  # default gray
    if masks is not None:
        for i, mask in enumerate(masks):
            if mask.shape[0] != camera.height or mask.shape[1] != camera.width:
                mask = cv2.resize(mask.astype(np.float32), (camera.width, camera.height))

            mask = mask > 0.5
            in_bounds = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
            for idx in np.where(in_bounds)[0]:
                if mask[v[idx], u[idx]]:
                    point_colors[idx] = [1.0, 0.0, 0.0]  # red for detected cars

    # Visualize in 3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.visualization.draw_geometries([pcd], window_name=f"3D Point Cloud with YOLO Mask")

def visualize_pointcloud(seq=0, frame=0):
    viewer = Kitti360Viewer3DRaw(seq=seq)
    points = viewer.loadVelodyneData(frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # gray points

    o3d.visualization.draw_geometries([pcd], window_name=f"PointCloud Frame {frame}")


if __name__ == '__main__':
    visualize_pointcloud_with_mask3D(seq=0, frame=0, cam_id=0)
    visualize_pointcloud(seq=0, frame=0)



