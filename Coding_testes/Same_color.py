import os
import glob
import copy
import random
import numpy as np
import cv2
import torch
import open3d as o3d
from ultralytics import YOLO
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective

# Model and data paths
output_dir = '/home/karan-sankla/LIDAR_RADAR/Predictions'
os.environ['KITTI360_DATASET'] = '/home/karan-sankla/LIDAR_RADAR/KITTI360_sample'
model = YOLO('yolo11x-seg.pt')


class Kitti360Viewer3DRaw:
    def __init__(self, seq=0, mode='velodyne'):
        kitti360Path = os.environ['KITTI360_DATASET']
        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, 'velodyne_points', 'data')

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32).reshape(-1, 4)
        return pcd


def image_segmentation(image):
    results = model.predict(image, device='0', classes=2, retina_masks=True)
    all_masks = []
    mask_colors = []

    for result in results:
        img = result.orig_img.copy()
        boxes = result.boxes.xyxy.cpu().numpy()

        if result.masks is None or len(boxes) == 0:
            return img, [], []

        masks = result.masks.data.cpu().numpy()
        for mask in masks:
            color = tuple(random.randint(0, 255) for _ in range(3))
            color_mask = np.zeros_like(img, dtype=np.uint8)
            all_masks.append(mask)
            mask_colors.append(color)
            color_mask[mask > 0.5] = color
            img = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)

        return img, all_masks, mask_colors

    return None, None, None


def projectVeloToImage(seq=0, cam_id=0):
    kitti360Path = os.environ['KITTI360_DATASET']
    sequence = '2013_05_28_drive_%04d_sync' % seq
    velo = Kitti360Viewer3DRaw(seq=seq)
    camera = CameraPerspective(kitti360Path, sequence, cam_id)

    # Load calibration
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[f'image_{cam_id:02d}']
    TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
    TrVeloToCam = np.linalg.inv(TrCamToVelo)
    TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam)

    available_files = sorted(glob.glob(os.path.join(velo.raw3DPcdPath, '*.bin')))
    print(f"[INFO] Found {len(available_files)} Velodyne frames.")

    # Simple frame-by-frame iteration with debugging checks
    for frame_idx, file in enumerate(available_files):
        frame = int(os.path.basename(file).split('.')[0])
        print(f"\n[INFO] Processing frame {frame} ({frame_idx + 1}/{len(available_files)})...")

        # Load point cloud data
        try:
            points = velo.loadVelodyneData(frame)
            print(f"[DEBUG] Loaded {len(points)} points from frame {frame}")
        except Exception as e:
            print(f"[ERROR] Failed to load velodyne data for frame {frame}: {e}")
            continue

        # Apply transformations
        points[:, 3] = 1  # homogeneous coords
        pointsCam = np.matmul(TrVeloToRect, points.T).T[:, :3]
        u, v, depth = camera.cam2image(pointsCam.T)
        u, v = u.astype(int), v.astype(int)

        # Load corresponding image
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, f'image_{cam_id:02d}',
                                 'data_rect' if cam_id in [0, 1] else 'data_rgb',
                                 '%010d.png' % frame)
        if not os.path.isfile(imagePath):
            print(f"[WARN] Image not found: {imagePath}")
            continue

        bgr_image = cv2.imread(imagePath)
        seg_image, masks, mask_colors = image_segmentation(bgr_image.copy())

        if masks is None:
            print(f"[INFO] No cars detected in frame {frame}, showing plain cloud.")
            masks, mask_colors = [], []

        # Filter valid points and separate by segmentation
        valid = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (depth > 0) & (depth < 30)
        colored_points, colored_colors, full_points = [], [], []

        valid_indices = np.where(valid)[0]
        print(f"[DEBUG] Frame {frame}: {len(valid_indices)} points passed validation filter")

        for idx in valid_indices:
            x, y = u[idx], v[idx]
            pt = points[idx, :3]
            matched = False
            for i, mask in enumerate(masks):
                if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 0.5:
                    color = np.array(mask_colors[i]) / 255.0
                    colored_points.append(pt)
                    colored_colors.append(color)
                    matched = True
                    break
            if not matched:
                full_points.append(pt)

        print(f"[DEBUG] Frame {frame}: {len(colored_points)} car points, {len(full_points)} background points")

        if not colored_points and not full_points:
            print(f"[WARN] No valid LiDAR points in frame {frame}")
            continue

        # Create visualizer for this frame
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Frame {frame}", width=800, height=600)

        # Add point clouds to visualizer
        if full_points:
            bg_pcd = o3d.geometry.PointCloud()
            bg_pcd.points = o3d.utility.Vector3dVector(np.array(full_points))
            bg_pcd.paint_uniform_color([0.5, 0.5, 0.5])
            vis.add_geometry(bg_pcd)

        if colored_points:
            car_pcd = o3d.geometry.PointCloud()
            car_pcd.points = o3d.utility.Vector3dVector(np.array(colored_points))
            car_pcd.colors = o3d.utility.Vector3dVector(np.array(colored_colors))
            vis.add_geometry(car_pcd)

        # Set view parameters for better visualization
        vis.get_render_option().point_size = 2.0
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])

        if len(points) > 0:
            center = np.mean(np.array(points[:, :3]), axis=0)
            ctr.set_lookat(center)
        ctr.set_zoom(0.5)

        # Update render once before running
        vis.poll_events()
        vis.update_renderer()

        # Wait for user to close this window
        print(f"[INFO] Showing frame {frame}. Close window to proceed to next frame...")
        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    seq = 0
    cam_id = 0
    projectVeloToImage(seq=seq, cam_id=cam_id)