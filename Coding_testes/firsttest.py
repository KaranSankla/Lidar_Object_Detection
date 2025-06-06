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
import json

# Model and data paths
output_dir = '/home/karan-sankla/LIDAR_RADAR/Predictions'
os.environ['KITTI360_DATASET'] = '/home/karan-sankla/LIDAR_RADAR/KITTI360_sample'
model = YOLO('yolo11x-seg.pt')
bbox_dir = os.path.join(os.environ['KITTI360_DATASET'], 'bboxes_3D_cam0')


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


def transform_bboxes_to_velodyne(bboxes_3d, TrVeloToCam):
    """
    Transform 3D bounding boxes from camera coordinates to Velodyne coordinates
    """
    TrCamToVelo = np.linalg.inv(TrVeloToCam)
    transformed_bboxes = []

    for bbox in bboxes_3d:
        bbox_transformed = bbox.copy()
        if 'corners_cam0' in bbox_transformed:
            # Get corners in camera coordinates
            corners_cam = np.array(bbox_transformed['corners_cam0'])

            # Add homogeneous coordinate
            corners_cam_homo = np.hstack([corners_cam, np.ones((corners_cam.shape[0], 1))])

            # Transform to Velodyne coordinates
            corners_velo = np.matmul(TrCamToVelo, corners_cam_homo.T).T[:, :3]

            bbox_transformed['corners_velo'] = corners_velo.tolist()

        transformed_bboxes.append(bbox_transformed)

    return transformed_bboxes


def debug_coordinate_systems(points, bboxes_3d, coordinate_frame="velodyne"):
    """
    Debug function to understand coordinate systems
    """
    print(f"\n=== COORDINATE SYSTEM DEBUG ({coordinate_frame}) ===")
    print(f"Point cloud shape: {points.shape}")
    print(f"Point cloud X range: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}")
    print(f"Point cloud Y range: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}")
    print(f"Point cloud Z range: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}")

    if bboxes_3d:
        # Check which coordinate system to use for bboxes
        corners_key = 'corners_velo' if coordinate_frame == "velodyne" else 'corners_cam0'
        if corners_key in bboxes_3d[0]:
            corners = np.array(bboxes_3d[0][corners_key])
            print(f"\nFirst bbox corners shape: {corners.shape}")
            print(f"Bbox X range: {corners[:, 0].min():.2f} to {corners[:, 0].max():.2f}")
            print(f"Bbox Y range: {corners[:, 1].min():.2f} to {corners[:, 1].max():.2f}")
            print(f"Bbox Z range: {corners[:, 2].min():.2f} to {corners[:, 2].max():.2f}")
        else:
            print(f"[WARN] No {corners_key} found in bbox")
    print("=== END DEBUG ===\n")


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


def load_bounding_boxes(json_path):
    """Load 3D bounding boxes from JSON file"""
    try:
        with open(json_path, 'r') as f:
            boxes = json.load(f)
        print(f"[INFO] Successfully loaded {len(boxes)} bounding boxes from {json_path}")
        return boxes
    except FileNotFoundError:
        print(f"[WARN] Bounding box file not found: {json_path}")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to load bounding boxes: {e}")
        return []


def create_bbox_lineset(corners, color=[1, 0, 0]):
    """Create a 3D bounding box line set from 8 corners"""
    # Define lines between the 8 corners of a 3D box
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # front face
        [4, 5], [5, 7], [7, 6], [6, 4],  # back face
        [0, 4], [1, 5], [2, 6], [3, 7]  # side edges
    ]
    corners_np = np.array(corners)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners_np),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def project_3d_bbox_to_2d(bbox_3d, camera):
    """Project 3D bounding box corners to 2D image coordinates"""
    try:
        # Extract corners from the JSON format (use camera coordinates for projection)
        corners_3d = np.array(bbox_3d['corners_cam0'])  # Camera coordinates for projection

        # Project to 2D
        u, v, depth = camera.cam2image(corners_3d.T)

        # Get 2D bounding box (min/max of projected corners)
        valid_depth = depth > 0
        if np.any(valid_depth):
            u_valid = u[valid_depth]
            v_valid = v[valid_depth]
            x_min, x_max = np.min(u_valid), np.max(u_valid)
            y_min, y_max = np.min(v_valid), np.max(v_valid)
            return [x_min, y_min, x_max, y_max], corners_3d

    except Exception as e:
        print(f"[ERROR] Failed to project 3D bbox: {e}")

    return None, None


def calculate_iou_2d(box1, box2):
    """Calculate IoU between two 2D bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    if xi_max <= xi_min or yi_max <= yi_min:
        return 0.0

    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_detections_to_bboxes(boxes_2d, bboxes_3d, mask_colors, camera, iou_threshold=0.1):
    """Match 2D detections to 3D bounding boxes based on IoU overlap"""
    matched_pairs = []

    if not bboxes_3d:
        print("[INFO] No 3D bounding boxes loaded from JSON")
        return matched_pairs

    print(f"[INFO] Attempting to match {len(boxes_2d)} 2D detections with {len(bboxes_3d)} 3D bboxes")

    for i, box_2d in enumerate(boxes_2d):
        best_match_idx = -1
        best_iou = 0
        best_corners_velo = None
        best_corners_cam = None

        # Convert 2D detection box format if needed
        if len(box_2d) == 4:
            detection_2d = box_2d  # [x1, y1, x2, y2]
        else:
            continue

        # Try to match with each 3D bbox
        for j, bbox_3d in enumerate(bboxes_3d):
            projected_2d, corners_cam = project_3d_bbox_to_2d(bbox_3d, camera)

            if projected_2d is not None:
                iou = calculate_iou_2d(detection_2d, projected_2d)
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_match_idx = j
                    best_corners_cam = corners_cam
                    best_corners_velo = np.array(bbox_3d['corners_velo']) if 'corners_velo' in bbox_3d else None

        # If we found a good match, add it to results
        if best_match_idx >= 0 and best_corners_velo is not None:
            color = np.array([mask_colors[i][2], mask_colors[i][1], mask_colors[i][0]], dtype=float) / 255.0
            matched_pairs.append((best_corners_velo, color))  # Use Velodyne coordinates for visualization
            print(f"[INFO] Matched detection {i} with 3D bbox {best_match_idx} (IoU: {best_iou:.3f})")
        else:
            print(f"[WARN] No matching 3D bbox found for detection {i}")

    return matched_pairs


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

        # Load 3D bounding boxes for this frame
        bbox_path = os.path.join(bbox_dir, f"BBoxes_{int(frame)}.json")
        bboxes_3d = load_bounding_boxes(bbox_path)

        if not bboxes_3d:
            print(f"[WARN] No bounding boxes loaded for frame {frame}")
            continue

        # Transform bounding boxes to Velodyne coordinates
        bboxes_3d = transform_bboxes_to_velodyne(bboxes_3d, TrVeloToCam)

        # Debug coordinate systems (now both should be in Velodyne frame)
        debug_coordinate_systems(points, bboxes_3d, "velodyne")

        # Transform points to camera frame for projection
        points_orig = points.copy()  # Keep original for visualization
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
            print(f"[INFO] No cars detected in frame {frame}, showing plain cloud with bboxes.")

            # Visualize raw point cloud with bounding boxes (both in Velodyne coordinates)
            geometries = []
            pcd_raw = create_pcd_file(points_orig[:, :3])
            geometries.append(pcd_raw)

            # Add bounding boxes in Velodyne coordinates
            for bbox in bboxes_3d:
                if 'corners_velo' in bbox:
                    corners = np.array(bbox['corners_velo'])
                    bbox_lineset = create_bbox_lineset(corners, color=[1, 0, 0])
                    geometries.append(bbox_lineset)

            print(f"[INFO] Visualizing raw point cloud with bboxes for frame {frame}")
            o3d.visualization.draw_geometries(geometries, window_name=f"Raw Point Cloud + BBoxes - Frame {frame}",
                                              width=800, height=600)

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

        # Create a list of geometries for visualization
        geometries = []

        # Background points (will be updated as we assign points to cars)
        bg_points = points_valid.copy()
        bg_assigned = np.zeros(len(points_valid), dtype=bool)

        # Process each car mask with its corresponding color
        for mask_idx, (mask, color) in enumerate(zip(masks, mask_colors)):
            # Project mask to 2D
            mask_resized = cv2.resize(mask.astype(np.uint8), (camera.width, camera.height))

            # Get 2D mask indices
            mask_indices = mask_resized[v_valid, u_valid] > 0.5

            if np.count_nonzero(mask_indices) == 0:
                continue

            # Assign color to these points
            car_points = points_valid[mask_indices]
            bg_assigned[mask_indices] = True
            car_color = np.array(color[::-1]) / 255.0  # Convert BGR to RGB
            pcd_car = create_pcd_file(car_points, colors=np.tile(car_color, (len(car_points), 1)))
            geometries.append(pcd_car)

        # Add unmatched background points as gray
        remaining_points = points_valid[~bg_assigned]
        if len(remaining_points) > 0:
            pcd_bg = create_pcd_file(remaining_points, colors=np.tile([0.5, 0.5, 0.5], (len(remaining_points), 1)))
            geometries.append(pcd_bg)

        # Add matched 3D bounding boxes (now in Velodyne coordinates)
        matched_pairs = match_detections_to_bboxes(boxes, bboxes_3d, mask_colors, camera)
        for corners, color in matched_pairs:

            bbox_lineset = create_bbox_lineset(corners, color=color)
            geometries.append(bbox_lineset)

        # Visualize
        print(f"[INFO] Visualizing segmented point cloud with matching bounding boxes for frame {frame}")
        o3d.visualization.draw_geometries(geometries, window_name=f"Segmented + BBoxes - Frame {frame}",
                                          width=800, height=600)

        # Optional wait for user input before next frame
        input("\nPress Enter to continue to next frame...\n")

        # Clean up
        cv2.destroyAllWindows()


if __name__ == '__main__':
    seq = 0
    cam_id = 0
    projectVeloToImage(seq=seq, cam_id=cam_id)