import os
import glob
import numpy as np
import cv2
import torch
import open3d as o3d
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


def image_segmentation_with_erosion(image, erosion_kernel_size=3, erosion_iterations=1):
    """
    Detect cars in image using YOLO with erosion to reduce bleed-out effect

    Args:
        image: Input BGR image
        erosion_kernel_size: Size of erosion kernel (3, 5, 7, etc.)
        erosion_iterations: Number of erosion iterations
    """
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

        # Apply erosion to each mask to reduce bleed-out
        eroded_masks = []
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))

        for mask in masks:
            # Convert mask to uint8 format
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Apply erosion
            eroded_mask = cv2.erode(mask_uint8, kernel, iterations=erosion_iterations)

            # Convert back to float format (0-1 range)
            eroded_mask_float = eroded_mask.astype(np.float32) / 255.0
            eroded_masks.append(eroded_mask_float)

        # Replace original masks with eroded masks
        masks = np.array(eroded_masks)

        # Generate colors (simple approach)
        colors = [(int(i * 60) % 255, int(i * 120) % 255, int(i * 180) % 255) for i in range(len(boxes))]

        # Draw results
        img = result.orig_img.copy()
        for i, (box, mask, conf, color) in enumerate(zip(boxes, masks, confidences, colors)):
            # Draw mask
            color_mask = np.zeros_like(img, dtype=np.uint8)
            color_mask[mask > 0.5] = color
            img = cv2.addWeighted(img, 1.0, color_mask, 0.4, 0)

            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'Car {i}: {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img, masks, colors, boxes, confidences

    return None, None, None, None, None


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


def point_in_bbox(points, bbox_corners):
    """
    Check if points are inside a 3D bounding box using vectorized operations.

    Args:
        points: Nx3 array of 3D points
        bbox_corners: 8x3 array of bounding box corners

    Returns:
        Boolean array indicating which points are inside the bbox
    """
    if len(points) == 0:
        return np.array([])

    # Calculate bounding box parameters
    min_coords = np.min(bbox_corners, axis=0)
    max_coords = np.max(bbox_corners, axis=0)

    # Simple axis-aligned bounding box check (fast approximation)
    inside = np.all((points >= min_coords) & (points <= max_coords), axis=1)

    return inside


def oriented_point_in_bbox(points, bbox_corners):
    """
    More accurate check for points inside oriented 3D bounding box.
    This uses the cross product method to handle rotated boxes.

    Args:
        points: Nx3 array of 3D points
        bbox_corners: 8x3 array of bounding box corners (in specific order)

    Returns:
        Boolean array indicating which points are inside the bbox
    """
    if len(points) == 0:
        return np.array([])

    # Assume corners are ordered as:
    # 0-3: bottom face (counterclockwise), 4-7: top face (counterclockwise)
    # Extract box vectors
    try:
        # Get three orthogonal vectors of the box
        v1 = bbox_corners[1] - bbox_corners[0]  # width
        v2 = bbox_corners[3] - bbox_corners[0]  # length
        v3 = bbox_corners[4] - bbox_corners[0]  # height

        # Transform points to box coordinate system
        points_rel = points - bbox_corners[0]  # Translate to box origin

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
        # Fallback to axis-aligned method if oriented method fails
        return point_in_bbox(points, bbox_corners)


def extract_car_points_by_mask(points_valid, u_valid, v_valid, masks, camera):
    """
    Extract car points for each detection mask.

    Returns:
        List of point arrays, one for each car detection
    """
    car_point_sets = []

    for mask_idx, mask in enumerate(masks):
        # Resize mask to camera resolution
        mask_resized = cv2.resize(mask.astype(np.uint8), (camera.width, camera.height))

        # Find points that fall within this mask
        mask_indices = mask_resized[v_valid, u_valid] > 0.5

        if np.count_nonzero(mask_indices) > 0:
            car_points = points_valid[mask_indices]
            car_point_sets.append(car_points)
        else:
            car_point_sets.append(np.array([]).reshape(0, 3))

    return car_point_sets


def match_car_points_to_bboxes(car_point_sets, bboxes_3d, colors, min_points=10, use_oriented=True):
    """
    Match car point sets to 3D bounding boxes based on point intersection count.

    Args:
        car_point_sets: List of Nx3 arrays, each containing points for one car detection
        bboxes_3d: List of 3D bounding box dictionaries with 'corners_velo'
        colors: List of colors for each car detection
        min_points: Minimum number of points required for a match
        use_oriented: Whether to use oriented bounding box check (slower but more accurate)

    Returns:
        List of matched (corners, color, point_count) tuples
    """
    matched_pairs = []

    if not bboxes_3d or len(car_point_sets) == 0:
        return matched_pairs

    # For each car detection, find the best matching 3D bbox
    for car_idx, car_points in enumerate(car_point_sets):
        if len(car_points) == 0:
            continue

        best_match_count = 0
        best_bbox_idx = -1

        # Test each 3D bounding box
        for bbox_idx, bbox_3d in enumerate(bboxes_3d):
            if 'corners_velo' not in bbox_3d:
                continue

            corners_velo = np.array(bbox_3d['corners_velo'])

            # Count points inside this bounding box
            if use_oriented:
                inside_mask = oriented_point_in_bbox(car_points, corners_velo)
            else:
                inside_mask = point_in_bbox(car_points, corners_velo)

            point_count = np.sum(inside_mask)

            # Update best match if this bbox contains more points
            if point_count > best_match_count and point_count >= min_points:
                best_match_count = point_count
                best_bbox_idx = bbox_idx

        # Add matched pair if we found a good match
        if best_bbox_idx >= 0:
            corners_velo = np.array(bboxes_3d[best_bbox_idx]['corners_velo'])
            color = np.array([colors[car_idx][2], colors[car_idx][1], colors[car_idx][0]]) / 255.0  # BGR to RGB
            matched_pairs.append((corners_velo, color, best_match_count))
            print(f"  Matched car {car_idx} to bbox {best_bbox_idx} with {best_match_count} points")

    return matched_pairs


def create_point_cloud(points, colors=None):
    """Create Open3D point cloud"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd


def create_bbox_lines(corners, color=[1, 0, 0]):
    """Create 3D bounding box wireframe"""
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # front face
        [4, 5], [5, 7], [7, 6], [6, 4],  # back face
        [0, 5], [1, 4], [2, 7], [3, 6]  # connections
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def calculate_car_point_statistics(car_point_sets, bboxes_3d, colors, min_points=10, use_oriented=True):
    """
    Calculate detailed statistics for car points vs bounding boxes.

    Args:
        car_point_sets: List of Nx3 arrays, each containing points for one car detection
        bboxes_3d: List of 3D bounding box dictionaries with 'corners_velo'
        colors: List of colors for each car detection
        min_points: Minimum number of points required for a match
        use_oriented: Whether to use oriented bounding box check

    Returns:
        List of dictionaries containing detailed statistics for each matched car
    """
    car_statistics = []

    if not bboxes_3d or len(car_point_sets) == 0:
        return car_statistics

    print(f"\n=== Car Point Statistics ===")
    print(f"Total car detections: {len(car_point_sets)}")
    print(f"Total 3D bounding boxes: {len(bboxes_3d)}")

    # For each car detection, find the best matching 3D bbox
    for car_idx, car_points in enumerate(car_point_sets):
        total_car_points = len(car_points)

        if total_car_points == 0:
            print(f"\nCar {car_idx}: No points detected")
            continue

        print(f"\nCar {car_idx}: {total_car_points} total points")

        best_match_count = 0
        best_bbox_idx = -1
        best_inside_mask = None

        # Test each 3D bounding box
        for bbox_idx, bbox_3d in enumerate(bboxes_3d):
            if 'corners_velo' not in bbox_3d:
                continue

            corners_velo = np.array(bbox_3d['corners_velo'])

            # Count points inside this bounding box
            if use_oriented:
                inside_mask = oriented_point_in_bbox(car_points, corners_velo)
            else:
                inside_mask = point_in_bbox(car_points, corners_velo)

            points_inside = np.sum(inside_mask)

            # Update best match if this bbox contains more points
            if points_inside > best_match_count:
                best_match_count = points_inside
                best_bbox_idx = bbox_idx
                best_inside_mask = inside_mask

        # Calculate statistics for the best match
        if best_bbox_idx >= 0 and best_match_count >= min_points:
            points_inside = best_match_count
            points_outside = total_car_points - points_inside
            inside_percentage = (points_inside / total_car_points) * 100
            outside_percentage = (points_outside / total_car_points) * 100

            # Create statistics dictionary
            car_stats = {
                'car_id': car_idx,
                'matched_bbox_id': best_bbox_idx,
                'total_points': total_car_points,
                'points_inside_bbox': points_inside,
                'points_outside_bbox': points_outside,
                'inside_percentage': inside_percentage,
                'outside_percentage': outside_percentage,
                'color': colors[car_idx],
                'corners_velo': np.array(bboxes_3d[best_bbox_idx]['corners_velo']),
                'inside_mask': best_inside_mask,
                'car_points': car_points
            }

            car_statistics.append(car_stats)

            # Print detailed statistics
            print(f"  ✓ Matched to 3D bbox {best_bbox_idx}")
            print(f"  │ Points inside bbox:  {points_inside:4d} ({inside_percentage:5.1f}%)")
            print(f"  │ Points outside bbox: {points_outside:4d} ({outside_percentage:5.1f}%)")
            print(f"  └ Total points:        {total_car_points:4d} (100.0%)")

        else:
            # No good match found
            print(f"  ✗ No matching 3D bbox found (best match: {best_match_count} points < {min_points} threshold)")

            # Still store statistics for unmatched cars
            car_stats = {
                'car_id': car_idx,
                'matched_bbox_id': -1,
                'total_points': total_car_points,
                'points_inside_bbox': 0,
                'points_outside_bbox': total_car_points,
                'inside_percentage': 0.0,
                'outside_percentage': 100.0,
                'color': colors[car_idx],
                'corners_velo': None,
                'inside_mask': None,
                'car_points': car_points
            }
            car_statistics.append(car_stats)

    return car_statistics


def print_summary_statistics(car_statistics):
    """Print a summary of all car statistics"""
    if not car_statistics:
        print("\nNo car statistics to display.")
        return

    print(f"\n{'=' * 60}")
    print(f"{'SUMMARY STATISTICS':^60}")
    print(f"{'=' * 60}")

    matched_cars = [stats for stats in car_statistics if stats['matched_bbox_id'] >= 0]
    unmatched_cars = [stats for stats in car_statistics if stats['matched_bbox_id'] < 0]

    print(f"Total cars detected: {len(car_statistics)}")
    print(f"Successfully matched: {len(matched_cars)}")
    print(f"Unmatched: {len(unmatched_cars)}")

    if matched_cars:
        print(f"\n{'Car ID':<8} {'BBox ID':<8} {'Total':<8} {'Inside':<8} {'Outside':<8} {'Inside %':<10}")
        print("-" * 60)

        for stats in matched_cars:
            print(f"{stats['car_id']:<8} "
                  f"{stats['matched_bbox_id']:<8} "
                  f"{stats['total_points']:<8} "
                  f"{stats['points_inside_bbox']:<8} "
                  f"{stats['points_outside_bbox']:<8} "
                  f"{stats['inside_percentage']:<10.1f}")

        # Calculate overall statistics
        total_points = sum(stats['total_points'] for stats in matched_cars)
        total_inside = sum(stats['points_inside_bbox'] for stats in matched_cars)
        total_outside = sum(stats['points_outside_bbox'] for stats in matched_cars)
        avg_inside_percentage = (total_inside / total_points * 100) if total_points > 0 else 0

        print("-" * 60)
        print(
            f"{'TOTAL':<8} {'':<8} {total_points:<8} {total_inside:<8} {total_outside:<8} {avg_inside_percentage:<10.1f}")


def create_colored_point_cloud_with_bbox_analysis(car_statistics):
    """
    Create colored point clouds showing inside vs outside bbox points

    Returns:
        List of Open3D geometries for visualization
    """
    geometries = []

    for stats in car_statistics:
        if stats['matched_bbox_id'] < 0:
            # Unmatched car - show all points in original color
            car_color = np.array([stats['color'][2], stats['color'][1], stats['color'][0]]) / 255.0
            pcd_car = create_point_cloud(stats['car_points'],
                                         np.tile(car_color, (len(stats['car_points']), 1)))
            geometries.append(pcd_car)
            continue

        car_points = stats['car_points']
        inside_mask = stats['inside_mask']

        # Points inside bounding box - original color
        if np.any(inside_mask):
            points_inside = car_points[inside_mask]
            inside_color = np.array([stats['color'][2], stats['color'][1], stats['color'][0]]) / 255.0
            pcd_inside = create_point_cloud(points_inside,
                                            np.tile(inside_color, (len(points_inside), 1)))
            geometries.append(pcd_inside)

        # Points outside bounding box - darker/different shade
        if np.any(~inside_mask):
            points_outside = car_points[~inside_mask]
            # Make outside points darker (multiply by 0.5) or use red tint
            outside_color = np.array([stats['color'][2], stats['color'][1], stats['color'][0]]) / 255.0
            pcd_outside = create_point_cloud(points_outside,
                                             np.tile(outside_color, (len(points_outside), 1)))
            geometries.append(pcd_outside)

        # Add the bounding box
        if stats['corners_velo'] is not None:
            bbox_color = np.array([stats['color'][2], stats['color'][1], stats['color'][0]]) / 255.0
            bbox_lines = create_bbox_lines(stats['corners_velo'], color=bbox_color)
            geometries.append(bbox_lines)

    return geometries
def process_frame_with_statistics(seq=0, cam_id=0):
    """Main processing function with detailed point statistics"""
    # ... (keeping all the existing setup code the same until the matching part)

    # Setup
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

        # ... (keep all existing loading and processing code until the matching part)

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
        seg_image, masks, colors, boxes, confidences = image_segmentation_with_erosion(bgr_image.copy(),
                                                                                       erosion_kernel_size=3,
                                                                                       erosion_iterations=1)

        # Filter valid points
        valid = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (depth > 0) & (depth < 30)
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            continue

        u_valid = u[valid]
        v_valid = v[valid]
        points_valid = points[valid_indices, :3]

        if masks is not None and len(masks) > 0:
            # Extract car points for each detection
            car_point_sets = extract_car_points_by_mask(points_valid, u_valid, v_valid, masks, camera)

            # Calculate detailed statistics
            car_statistics = calculate_car_point_statistics(car_point_sets, bboxes_3d, colors,
                                                            min_points=10, use_oriented=True)

            # Print summary
            print_summary_statistics(car_statistics)

            # Create visualization with enhanced coloring
            geometries = create_colored_point_cloud_with_bbox_analysis(car_statistics)

            # Add background points
            bg_assigned = np.zeros(len(points_valid), dtype=bool)
            for car_points in car_point_sets:
                if len(car_points) > 0:
                    for i, point in enumerate(points_valid):
                        if not bg_assigned[i]:
                            distances = np.linalg.norm(car_points - point, axis=1)
                            if np.any(distances < 1e-6):
                                bg_assigned[i] = True

            remaining_points = points_valid[~bg_assigned]
            if len(remaining_points) > 0:
                pcd_bg = create_point_cloud(remaining_points, np.tile([0.5, 0.5, 0.5], (len(remaining_points), 1)))
                geometries.append(pcd_bg)

        else:
            # No detections - show raw point cloud with all bboxes
            geometries = []
            pcd_raw = create_point_cloud(points_valid)
            geometries.append(pcd_raw)

            for bbox in bboxes_3d:
                if 'corners_velo' in bbox:
                    corners = np.array(bbox['corners_velo'])
                    bbox_lines = create_bbox_lines(corners, color=[1, 0, 0])
                    geometries.append(bbox_lines)

        # Visualize
        print(f"Visualizing frame {frame} with {len(geometries)} objects")
        o3d.visualization.draw_geometries(geometries, window_name=f"Frame {frame} - Enhanced Statistics",
                                          width=1000, height=700)

        input("Press Enter to continue...")
        cv2.destroyAllWindows()


# Usage example:
if __name__ == '__main__':
    process_frame_with_statistics(seq=0, cam_id=0)