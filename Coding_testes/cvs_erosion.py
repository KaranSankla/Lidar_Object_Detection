import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob
import cv2
import json
from ultralytics import YOLO
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective

# Setup
os.environ['KITTI360_DATASET'] = '/home/karan-sankla/LIDAR_RADAR/KITTI360_sample'
model = YOLO('yolo11x-seg.pt')
bbox_dir = os.path.join(os.environ['KITTI360_DATASET'], 'bboxes_3D_cam0')


class Kitti360Viewer3DRaw:
    def __init__(self, seq=0):
        kitti360Path = os.environ['KITTI360_DATASET']
        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, 'velodyne_points', 'data')

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError(f'{pcdFile} does not exist!')
        return np.fromfile(pcdFile, dtype=np.float32).reshape(-1, 4)


def load_bounding_boxes(json_path):
    """Load 3D bounding boxes from JSON file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No bounding boxes found: {json_path}")
        return []


def transform_bboxes_to_velodyne(bboxes_3d, TrVeloToCam):
    """Transform 3D bounding boxes from camera to Velodyne coordinates"""
    TrCamToVelo = np.linalg.inv(TrVeloToCam)

    for bbox in bboxes_3d:
        if 'corners_cam0' in bbox:
            corners_cam = np.array(bbox['corners_cam0'])
            corners_cam_homo = np.hstack([corners_cam, np.ones((corners_cam.shape[0], 1))])
            corners_velo = np.matmul(TrCamToVelo, corners_cam_homo.T).T[:, :3]
            bbox['corners_velo'] = corners_velo.tolist()

    return bboxes_3d


def filter_visible_bboxes(bboxes_3d, camera):
    """Keep only bounding boxes visible in camera view"""
    filtered = []

    for bbox in bboxes_3d:
        if 'corners_cam0' not in bbox:
            continue

        corners_3d = np.array(bbox['corners_cam0'])
        u, v, depth = camera.cam2image(corners_3d.T)

        # Check if any corners are in front of camera and within image bounds
        valid_depth = depth > 0.1
        valid_image = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        valid_points = np.sum(valid_depth & valid_image)

        if valid_points >= 2:  # At least 2 corners visible
            filtered.append(bbox)

    return filtered


def image_segmentation_with_erosion(image, erosion_kernel_size=3, erosion_iterations=1):
    """Detect cars in image using YOLO with erosion to reduce bleed-out effect"""
    results = model.predict(image, device='0', classes=2, retina_masks=True)  # class 2 = car

    for result in results:
        if result.masks is None or len(result.boxes) == 0:
            return image, [], [], [], []

        # Get detections
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.data.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        # Sort by confidence
        sorted_indices = np.argsort(confidences)[::-1]
        boxes = boxes[sorted_indices]
        masks = masks[sorted_indices]
        confidences = confidences[sorted_indices]

        # Apply erosion to each mask
        eroded_masks = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))

        for mask in masks:
            mask_uint8 = (mask * 255).astype(np.uint8)
            eroded_mask = cv2.erode(mask_uint8, kernel, iterations=erosion_iterations)
            eroded_mask_float = eroded_mask.astype(np.float32) / 255.0
            eroded_masks.append(eroded_mask_float)

        masks = np.array(eroded_masks)
        colors = [(int(i * 60) % 255, int(i * 120) % 255, int(i * 180) % 255) for i in range(len(boxes))]

        return result.orig_img.copy(), masks, colors, boxes, confidences

    return None, None, None, None, None


def oriented_point_in_bbox(points, bbox_corners):
    """Check if points are inside oriented 3D bounding box"""
    if len(points) == 0:
        return np.array([])

    try:
        # Get three orthogonal vectors of the box
        v1 = bbox_corners[1] - bbox_corners[0]  # width
        v2 = bbox_corners[3] - bbox_corners[0]  # length
        v3 = bbox_corners[4] - bbox_corners[0]  # height

        # Transform points to box coordinate system
        points_rel = points - bbox_corners[0]

        # Project points onto box axes
        proj1 = np.dot(points_rel, v1) / np.dot(v1, v1)
        proj2 = np.dot(points_rel, v2) / np.dot(v2, v2)
        proj3 = np.dot(points_rel, v3) / np.dot(v3, v3)

        # Check if projections are within [0, 1] for all axes
        inside = (proj1 >= 0) & (proj1 <= 1) & \
                 (proj2 >= 0) & (proj2 <= 1) & \
                 (proj3 >= 0) & (proj3 <= 1)

        return inside

    except:
        # Fallback to axis-aligned method
        min_coords = np.min(bbox_corners, axis=0)
        max_coords = np.max(bbox_corners, axis=0)
        inside = np.all((points >= min_coords) & (points <= max_coords), axis=1)
        return inside


def extract_car_points_by_mask(points_valid, u_valid, v_valid, masks, camera):
    """Extract car points for each detection mask"""
    car_point_sets = []

    for mask in masks:
        mask_resized = cv2.resize(mask.astype(np.uint8), (camera.width, camera.height))
        mask_indices = mask_resized[v_valid, u_valid] > 0.5

        if np.count_nonzero(mask_indices) > 0:
            car_points = points_valid[mask_indices]
            car_point_sets.append(car_points)
        else:
            car_point_sets.append(np.array([]).reshape(0, 3))

    return car_point_sets


def calculate_car_point_statistics(car_point_sets, bboxes_3d, colors, min_points=10):
    """Calculate detailed statistics for car points vs bounding boxes"""
    car_statistics = []

    if not bboxes_3d or len(car_point_sets) == 0:
        return car_statistics

    print(f"Total car detections: {len(car_point_sets)}")
    print(f"Total 3D bounding boxes: {len(bboxes_3d)}")

    for car_idx, car_points in enumerate(car_point_sets):
        total_car_points = len(car_points)

        if total_car_points == 0:
            continue

        best_match_count = 0
        best_bbox_idx = -1

        # Test each 3D bounding box
        for bbox_idx, bbox_3d in enumerate(bboxes_3d):
            if 'corners_velo' not in bbox_3d:
                continue

            corners_velo = np.array(bbox_3d['corners_velo'])
            inside_mask = oriented_point_in_bbox(car_points, corners_velo)
            points_inside = np.sum(inside_mask)

            if points_inside > best_match_count:
                best_match_count = points_inside
                best_bbox_idx = bbox_idx

        # Calculate statistics
        if best_bbox_idx >= 0 and best_match_count >= min_points:
            points_inside = best_match_count
            points_outside = total_car_points - points_inside
            inside_percentage = (points_inside / total_car_points) * 100
            outside_percentage = (points_outside / total_car_points) * 100

            car_stats = {
                'car_id': car_idx,
                'matched_bbox_id': best_bbox_idx,
                'total_points': total_car_points,
                'points_inside_bbox': points_inside,
                'points_outside_bbox': points_outside,
                'inside_percentage': inside_percentage,
                'outside_percentage': outside_percentage,
                'color': colors[car_idx]
            }
        else:
            # No good match found
            car_stats = {
                'car_id': car_idx,
                'matched_bbox_id': -1,
                'total_points': total_car_points,
                'points_inside_bbox': 0,
                'points_outside_bbox': total_car_points,
                'inside_percentage': 0.0,
                'outside_percentage': 100.0,
                'color': colors[car_idx]
            }

        car_statistics.append(car_stats)

    return car_statistics


def append_to_master_csv(car_statistics, frame_number, master_csv_path="results/master_car_statistics.csv"):
    """Append car statistics to master CSV file"""
    if not car_statistics:
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(master_csv_path), exist_ok=True)

    # Prepare data
    csv_data = []
    for stats in car_statistics:
        row = {
            'frame': frame_number,
            'car_id': stats['car_id'],
            'matched_bbox_id': stats['matched_bbox_id'],
            'total_points': stats['total_points'],
            'points_inside_bbox': stats['points_inside_bbox'],
            'points_outside_bbox': stats['points_outside_bbox'],
            'inside_percentage': round(stats['inside_percentage'], 2),
            'outside_percentage': round(stats['outside_percentage'], 2),
            'is_matched': stats['matched_bbox_id'] >= 0,
            'timestamp': datetime.now().isoformat()
        }
        csv_data.append(row)

    df_new = pd.DataFrame(csv_data)

    # Append to or create master file
    if os.path.exists(master_csv_path):
        df_new.to_csv(master_csv_path, mode='a', header=False, index=False)
        print(f"Appended {len(csv_data)} rows to master CSV: {master_csv_path}")
    else:
        df_new.to_csv(master_csv_path, index=False)
        print(f"Created new master CSV: {master_csv_path}")


def analyze_master_csv(master_csv_path="results/master_car_statistics.csv"):
    """Analyze the master CSV file to get overall statistics"""
    if not os.path.exists(master_csv_path):
        print(f"Master CSV file not found: {master_csv_path}")
        return

    df = pd.read_csv(master_csv_path)

    print(f"\n{'=' * 60}")
    print(f"{'OVERALL ANALYSIS':^60}")
    print(f"{'=' * 60}")

    print(f"Total frames processed: {df['frame'].nunique()}")
    print(f"Total car detections: {len(df)}")
    print(f"Successfully matched cars: {df['is_matched'].sum()}")
    print(f"Unmatched cars: {(~df['is_matched']).sum()}")
    print(f"Average matching rate: {df['is_matched'].mean() * 100:.1f}%")

    # Statistics for matched cars only
    matched_df = df[df['is_matched'] == True]
    if len(matched_df) > 0:
        print(f"\nMatched Cars Statistics:")
        print(f"Average points per car: {matched_df['total_points'].mean():.1f}")
        print(f"Average inside percentage: {matched_df['inside_percentage'].mean():.1f}%")
        print(f"Min inside percentage: {matched_df['inside_percentage'].min():.1f}%")
        print(f"Max inside percentage: {matched_df['inside_percentage'].max():.1f}%")

    return df


def process_frames(seq=0, cam_id=0):
    """Main processing function that generates master CSV"""
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

    # Get available frames
    available_files = sorted(glob.glob(os.path.join(velo.raw3DPcdPath, '*.bin')))
    print(f"Found {len(available_files)} frames to process")

    for file in available_files:
        frame = int(os.path.basename(file).split('.')[0])
        print(f"\nProcessing frame {frame}...")

        # Load point cloud
        try:
            points = velo.loadVelodyneData(frame)
        except Exception as e:
            print(f"Failed to load frame {frame}: {e}")
            continue

        # Load 3D bounding boxes
        bbox_path = os.path.join(bbox_dir, f"BBoxes_{frame}.json")
        bboxes_3d_raw = load_bounding_boxes(bbox_path)
        if not bboxes_3d_raw:
            continue

        # Filter visible bboxes and transform to Velodyne coordinates
        bboxes_3d_filtered = filter_visible_bboxes(bboxes_3d_raw, camera)
        bboxes_3d = transform_bboxes_to_velodyne(bboxes_3d_filtered, TrVeloToCam)

        # Transform points for projection
        points_homo = points.copy()
        points_homo[:, 3] = 1
        pointsCam = np.matmul(TrVeloToRect, points_homo.T).T[:, :3]
        u, v, depth = camera.cam2image(pointsCam.T)
        u, v = u.astype(int), v.astype(int)

        # Load and segment image
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, f'image_{cam_id:02d}',
                                 'data_rect' if cam_id in [0, 1] else 'data_rgb', f'{frame:010d}.png')

        if not os.path.isfile(imagePath):
            continue

        bgr_image = cv2.imread(imagePath)
        seg_image, masks, colors, boxes, confidences = image_segmentation_with_erosion(bgr_image.copy())

        # Filter valid points
        valid = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (depth > 0) & (depth < 50)
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            continue

        u_valid = u[valid]
        v_valid = v[valid]
        points_valid = points[valid_indices, :3]

        if masks is not None and len(masks) > 0:
            # Extract car points and calculate statistics
            car_point_sets = extract_car_points_by_mask(points_valid, u_valid, v_valid, masks, camera)
            car_statistics = calculate_car_point_statistics(car_point_sets, bboxes_3d, colors, min_points=10)

            # Save to master CSV
            if car_statistics:
                append_to_master_csv(car_statistics, frame)

    # Analyze results after processing all frames
    analyze_master_csv()


# Usage
if __name__ == '__main__':
    process_frames(seq=0, cam_id=0)