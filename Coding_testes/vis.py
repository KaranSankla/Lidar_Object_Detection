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
import time

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
    results = model.predict(image, device='0', classes=2, retina_masks=True)  # class 2 = car
    all_masks = []
    mask_colors = []
    boxes = []

    for result in results:
        img = result.orig_img.copy()

        if result.masks is None or len(result.boxes) == 0:
            return img, [], [], []

        # Get boxes and masks
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.data.cpu().numpy()

        # Generate consistent colors for each car instance
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            # Create a fixed color based on instance index (to ensure consistency)
            # Using HSV color space for better color distribution
            hue = (i * 30) % 180  # Different hue for each car
            color_hsv = np.array([hue, 255, 255], dtype=np.uint8)
            color_bgr = cv2.cvtColor(color_hsv.reshape(1, 1, 3), cv2.COLOR_HSV2BGR).flatten()
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

            # Store the mask and color
            all_masks.append(mask)
            mask_colors.append(color)

            # Draw the mask with this color
            color_mask = np.zeros_like(img, dtype=np.uint8)
            color_mask[mask > 0.5] = color
            img = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)

            # Draw bounding box with same color
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        return img, all_masks, mask_colors, boxes

    return None, None, None, None


def create_pcd_file(points, colors=None):
    """Create a point cloud object for visualization"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd


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

    # Process and visualize each frame
    for idx, file in enumerate(available_files):
        frame = int(os.path.basename(file).split('.')[0])
        print(f"\n[INFO] Processing frame {frame} ({idx + 1}/{len(available_files)})...")

        # Load and process point cloud data
        try:
            points = velo.loadVelodyneData(frame)
            print(f"[DEBUG] Loaded {len(points)} points from frame {frame}")
        except Exception as e:
            print(f"[ERROR] Failed to load velodyne data for frame {frame}: {e}")
            continue

        # Transform points to camera frame
        points_orig = points.copy()  # Keep original for debugging
        points[:, 3] = 1  # homogeneous coords
        pointsCam = np.matmul(TrVeloToRect, points.T).T[:, :3]
        u, v, depth = camera.cam2image(pointsCam.T)
        u, v = u.astype(int), v.astype(int)

        # Load image and perform segmentation
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, f'image_{cam_id:02d}',
                                 'data_rect' if cam_id in [0, 1] else 'data_rgb',
                                 '%010d.png' % frame)
        if not os.path.isfile(imagePath):
            print(f"[WARN] Image not found: {imagePath}")
            continue

        bgr_image = cv2.imread(imagePath)
        seg_image, masks, mask_colors, boxes = image_segmentation(bgr_image.copy())

        if masks is None or len(masks) == 0:
            print(f"[INFO] No cars detected in frame {frame}, showing plain cloud.")

            # Visualize raw point cloud when no cars are detected
            pcd_raw = create_pcd_file(points_orig[:, :3])
            print(f"[INFO] Visualizing raw point cloud for frame {frame}")
            o3d.visualization.draw_geometries([pcd_raw], window_name=f"Raw Point Cloud - Frame {frame}", width=800,
                                              height=600)

            # Continue to next frame
            input("\nPress Enter to continue to next frame...\n")
            continue

        # Filter valid points that are within camera view
        valid = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (depth > 0) & (depth < 30)
        valid_indices = np.where(valid)[0]
        print(f"[DEBUG] Frame {frame}: {len(valid_indices)} points passed validation filter")

        if len(valid_indices) == 0:
            print(f"[WARN] No valid LiDAR points in frame {frame}")
            continue

        # Get only valid LiDAR points
        u_valid = u[valid]
        v_valid = v[valid]
        points_valid = points_orig[valid_indices, :3]

        # Initialize colors for all points (grey by default)
        point_colors = np.ones((len(points_valid), 3)) * 0.5  # Default gray

        # Create a list of point clouds - one for background and one for each car
        pcds = []

        # Background points (will be updated as we assign points to cars)
        bg_points = points_valid.copy()
        bg_assigned = np.zeros(len(points_valid), dtype=bool)

        # Process each car mask with its corresponding color
        for mask_idx, (mask, color) in enumerate(zip(masks, mask_colors)):
            # Convert BGR color (0-255) to RGB (0-1) for point cloud
            normalized_color = np.array([color[2], color[1], color[0]], dtype=float) / 255.0

            # Find points that belong to this specific car mask
            car_points_indices = []
            for i, (uu, vv) in enumerate(zip(u_valid, v_valid)):
                if 0 <= uu < camera.width and 0 <= vv < camera.height:
                    if mask[vv, uu] > 0.5:
                        car_points_indices.append(i)

            car_points_indices = np.array(car_points_indices)

            if len(car_points_indices) > 0:
                print(f"[INFO] Car {mask_idx + 1}: Found {len(car_points_indices)} points")

                # Extract these car points
                car_points = points_valid[car_points_indices]

                # Create colors for this car's points (consistent with the mask)
                car_colors = np.tile(normalized_color, (len(car_points), 1))

                # Create a point cloud for this car
                car_pcd = create_pcd_file(car_points, car_colors)
                pcds.append(car_pcd)

                # Mark these points as assigned to avoid duplication in background
                bg_assigned[car_points_indices] = True

        # Create background point cloud (points not assigned to any car)
        bg_indices = np.where(~bg_assigned)[0]
        if len(bg_indices) > 0:
            bg_points = points_valid[bg_indices]
            bg_colors = np.ones((len(bg_points), 3)) * 0.5  # Gray
            bg_pcd = create_pcd_file(bg_points, bg_colors)
            pcds.insert(0, bg_pcd)  # Add background as first element

        # Visualize point clouds
        if pcds:
            print(f"[INFO] Visualizing segmented point cloud for frame {frame} with {len(pcds) - 1} cars")
            o3d.visualization.draw_geometries(pcds,
                                              window_name=f"Segmented Point Cloud - Frame {frame}",
                                              width=800, height=600)

            # Also visualize the segmented image
            #cv2.imshow(f"Segmented Image - Frame {frame}", seg_image)
            #cv2.waitKey(100)  # Brief display
        else:
            # No point clouds to visualize
            print(f"[WARN] No point clouds to visualize for frame {frame}")

        # Give user time to see the visualizatio
        cv2.destroyAllWindows()


if __name__ == '__main__':
    seq = 0
    cam_id = 0
    projectVeloToImage(seq=seq, cam_id=cam_id)