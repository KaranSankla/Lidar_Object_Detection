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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

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


def generate_consistent_colors(n_objects):
    """Generate visually distinct colors for each object"""
    colors = []
    for i in range(n_objects):
        # Use golden ratio for better color distribution
        hue = (i * 137.508) % 360  # Golden angle in degrees
        saturation = 0.8 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.8 + (i % 2) * 0.2  # Vary brightness slightly

        # Convert HSV to RGB
        h_i = int(hue / 60) % 6
        f = (hue / 60) - h_i
        p = value * (1 - saturation)
        q = value * (1 - f * saturation)
        t = value * (1 - (1 - f) * saturation)

        if h_i == 0:
            r, g, b = value, t, p
        elif h_i == 1:
            r, g, b = q, value, p
        elif h_i == 2:
            r, g, b = p, value, t
        elif h_i == 3:
            r, g, b = p, q, value
        elif h_i == 4:
            r, g, b = t, p, value
        else:
            r, g, b = value, p, q

        # Convert to BGR for OpenCV and scale to 0-255
        bgr_color = (int(b * 255), int(g * 255), int(r * 255))
        colors.append(bgr_color)

    return colors


def image_segmentation(image):
    results = model.predict(image, device='0', classes=2, retina_masks=True)  # class 2 = car
    all_masks = []
    mask_colors = []
    boxes = []
    confidences = []

    for result in results:
        img = result.orig_img.copy()

        if result.masks is None or len(result.boxes) == 0:
            return img, [], [], [], []

        # Get boxes, masks, and confidences
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.data.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        # Sort by confidence (highest first) for better matching
        sorted_indices = np.argsort(confidences)[::-1]
        boxes = boxes[sorted_indices]
        masks = masks[sorted_indices]
        confidences = confidences[sorted_indices]

        # Generate consistent colors for all detections
        mask_colors = generate_consistent_colors(len(boxes))

        # Draw masks and boxes
        for i, (box, mask, conf, color) in enumerate(zip(boxes, masks, confidences, mask_colors)):
            # Store the mask and color
            all_masks.append(mask)

            # Draw the mask with this color
            color_mask = np.zeros_like(img, dtype=np.uint8)
            color_mask[mask > 0.5] = color
            img = cv2.addWeighted(img, 1.0, color_mask, 0.4, 0)

            # Draw bounding box with same color and confidence
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'Car {i}: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img, all_masks, mask_colors, boxes, confidences

    return None, None, None, None, None


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
        [0, 5], [1, 4], [2, 7], [3, 6]  # side edges
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

            # Calculate additional metrics for better matching
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            area = width * height

            bbox_2d_info = {
                'bbox': [x_min, y_min, x_max, y_max],
                'center': [center_x, center_y],
                'size': [width, height],
                'area': area,
                'avg_depth': np.mean(depth[valid_depth])
            }

            return bbox_2d_info, corners_3d

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


def calculate_matching_score(detection_info, bbox_3d_info, weight_iou=0.5, weight_center=0.3, weight_size=0.2):
    """
    Calculate comprehensive matching score between 2D detection and projected 3D bbox
    """
    # IoU score
    iou = calculate_iou_2d(detection_info['bbox'], bbox_3d_info['bbox'])

    # Center distance (normalized by image diagonal)
    center_dist = np.linalg.norm(np.array(detection_info['center']) - np.array(bbox_3d_info['center']))
    center_score = max(0, 1 - center_dist / 1000)  # Normalize by typical image size

    # Size similarity
    det_area = detection_info['size'][0] * detection_info['size'][1]
    bbox_area = bbox_3d_info['area']
    if det_area > 0 and bbox_area > 0:
        size_ratio = min(det_area, bbox_area) / max(det_area, bbox_area)
    else:
        size_ratio = 0

    # Combined score
    total_score = weight_iou * iou + weight_center * center_score + weight_size * size_ratio

    return total_score, {
        'iou': iou,
        'center_score': center_score,
        'size_score': size_ratio,
        'total_score': total_score
    }


def improved_match_detections_to_bboxes(boxes_2d, bboxes_3d, mask_colors, camera,
                                        min_score_threshold=0.3, min_iou_threshold=0.15):
    """
    Improved matching using Hungarian algorithm and multiple criteria
    """
    matched_pairs = []

    if not bboxes_3d or len(boxes_2d) == 0:
        print("[INFO] No detections or 3D bounding boxes to match")
        return matched_pairs

    print(f"[INFO] Matching {len(boxes_2d)} 2D detections with {len(bboxes_3d)} 3D bboxes")

    # Prepare detection info
    detection_infos = []
    for i, box_2d in enumerate(boxes_2d):
        if len(box_2d) == 4:
            x1, y1, x2, y2 = box_2d
            detection_info = {
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'size': [x2 - x1, y2 - y1],
                'area': (x2 - x1) * (y2 - y1)
            }
            detection_infos.append(detection_info)

    # Project all 3D bboxes and calculate matching scores
    bbox_3d_infos = []
    valid_bbox_indices = []

    for j, bbox_3d in enumerate(bboxes_3d):
        projected_info, corners_cam = project_3d_bbox_to_2d(bbox_3d, camera)
        if projected_info is not None:
            bbox_3d_infos.append(projected_info)
            valid_bbox_indices.append(j)

    if not bbox_3d_infos:
        print("[WARN] No valid 3D bbox projections found")
        return matched_pairs

    # Create cost matrix for Hungarian algorithm
    n_detections = len(detection_infos)
    n_bboxes = len(bbox_3d_infos)
    cost_matrix = np.zeros((n_detections, n_bboxes))
    score_details = {}

    for i, det_info in enumerate(detection_infos):
        for j, bbox_info in enumerate(bbox_3d_infos):
            score, details = calculate_matching_score(det_info, bbox_info)
            cost_matrix[i, j] = 1 - score  # Convert to cost (lower is better)
            score_details[(i, j)] = details

    # Apply Hungarian algorithm
    det_indices, bbox_indices = linear_sum_assignment(cost_matrix)

    # Filter matches based on thresholds
    for det_idx, bbox_idx in zip(det_indices, bbox_indices):
        score_info = score_details[(det_idx, bbox_idx)]
        total_score = score_info['total_score']
        iou = score_info['iou']

        if total_score >= min_score_threshold and iou >= min_iou_threshold:
            # Get the original 3D bbox index
            original_bbox_idx = valid_bbox_indices[bbox_idx]
            bbox_3d = bboxes_3d[original_bbox_idx]

            if 'corners_velo' in bbox_3d:
                corners_velo = np.array(bbox_3d['corners_velo'])

                # Use consistent color mapping - ensure index bounds
                if det_idx < len(mask_colors):
                    color = np.array([mask_colors[det_idx][2], mask_colors[det_idx][1],
                                      mask_colors[det_idx][0]], dtype=float) / 255.0
                else:
                    # Fallback color if index is out of bounds
                    color = np.array([1.0, 0.0, 0.0])  # Red as fallback

                matched_pairs.append((corners_velo, color))

                print(f"[INFO] Matched detection {det_idx} with 3D bbox {original_bbox_idx}")
                print(f"        Scores - IoU: {iou:.3f}, Center: {score_info['center_score']:.3f}, "
                      f"Size: {score_info['size_score']:.3f}, Total: {total_score:.3f}")
            else:
                print(f"[WARN] No Velodyne corners found for bbox {original_bbox_idx}")
        else:
            print(f"[INFO] Rejected match det{det_idx}-bbox{bbox_idx}: "
                  f"score={total_score:.3f}, IoU={iou:.3f}")

    # Handle unmatched 3D bboxes (show in default color)
    matched_detection_indices = set()
    for det_idx, bbox_idx in zip(det_indices, bbox_indices):
        score_info = score_details[(det_idx, bbox_idx)]
        if score_info['total_score'] >= min_score_threshold and score_info['iou'] >= min_iou_threshold:
            matched_detection_indices.add(det_idx)

    matched_bbox_indices = set()
    for det_idx, bbox_idx in zip(det_indices, bbox_indices):
        score_info = score_details[(det_idx, bbox_idx)]
        if score_info['total_score'] >= min_score_threshold and score_info['iou'] >= min_iou_threshold:
            matched_bbox_indices.add(valid_bbox_indices[bbox_idx])

    for i, bbox_3d in enumerate(bboxes_3d):
        if i not in matched_bbox_indices and 'corners_velo' in bbox_3d:
            corners_velo = np.array(bbox_3d['corners_velo'])
            # Use a distinct color for unmatched bboxes
            default_color = [0.7, 0.7, 0.7]  # Light gray
            matched_pairs.append((corners_velo, default_color))
            print(f"[INFO] Added unmatched 3D bbox {i} in default color")

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
        seg_image, masks, mask_colors, boxes, confidences = image_segmentation(bgr_image.copy())

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

        # Add improved matched 3D bounding boxes
        matched_pairs = improved_match_detections_to_bboxes(boxes, bboxes_3d, mask_colors, camera)
        for corners, color in matched_pairs:
            bbox_lineset = create_bbox_lineset(corners, color=color)
            geometries.append(bbox_lineset)

        # Show segmented image
        cv2.imshow(f'Segmented Image - Frame {frame}', seg_image)
        cv2.waitKey(1)

        # Visualize
        print(f"[INFO] Visualizing segmented point cloud with improved matching for frame {frame}")
        o3d.visualization.draw_geometries(geometries, window_name=f"Improved Segmented + BBoxes - Frame {frame}",
                                          width=800, height=600)

        # Optional wait for user input before next frame
        input("\nPress Enter to continue to next frame...\n")

        # Clean up
        cv2.destroyAllWindows()


if __name__ == '__main__':
    seq = 0
    cam_id = 0
    projectVeloToImage(seq=seq, cam_id=cam_id)