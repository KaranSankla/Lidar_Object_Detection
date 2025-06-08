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


def image_segmentation(image):
    """Detect cars in image using YOLO"""
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


def calculate_iou_2d(box1, box2):
    """Calculate IoU between two 2D bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
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


def match_detections_to_bboxes(boxes_2d, bboxes_3d, colors, camera, min_iou=0.25):
    """Match 2D detections with 3D bounding boxes using IoU"""
    matched_pairs = []

    if not bboxes_3d or len(boxes_2d) == 0:
        return matched_pairs

    # For each 2D detection, find best matching 3D bbox
    for det_idx, box_2d in enumerate(boxes_2d):
        x1, y1, x2, y2 = box_2d
        best_iou = 0
        best_bbox_idx = -1

        for bbox_idx, bbox_3d in enumerate(bboxes_3d):
            if 'corners_cam0' not in bbox_3d:
                continue

            # Project 3D bbox to 2D
            corners_3d = np.array(bbox_3d['corners_cam0'])
            u, v, depth = camera.cam2image(corners_3d.T)

            # Get 2D bounding box of projection
            valid_depth = depth > 0
            if np.sum(valid_depth) == 0:
                continue

            u_valid = u[valid_depth]
            v_valid = v[valid_depth]
            bbox_2d = [np.min(u_valid), np.min(v_valid), np.max(u_valid), np.max(v_valid)]

            # Calculate IoU
            iou = calculate_iou_2d([x1, y1, x2, y2], bbox_2d)

            if iou > best_iou and iou > min_iou:
                best_iou = iou
                best_bbox_idx = bbox_idx

        # Add matched pair
        if best_bbox_idx >= 0 and 'corners_velo' in bboxes_3d[best_bbox_idx]:
            corners_velo = np.array(bboxes_3d[best_bbox_idx]['corners_velo'])
            color = np.array([colors[det_idx][2], colors[det_idx][1], colors[det_idx][0]]) / 255.0  # BGR to RGB
            matched_pairs.append((corners_velo, color))

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


def process_frame(seq=0, cam_id=0):
    """Main processing function"""
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
        seg_image, masks, colors, boxes, confidences = image_segmentation(bgr_image.copy())

        # Filter valid points
        valid = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (depth > 0) & (depth < 30)
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            continue

        u_valid = u[valid]
        v_valid = v[valid]
        points_valid = points[valid_indices, :3]

        # Create visualization
        geometries = []

        if masks is not None and len(masks) > 0:
            # Color points by car segmentation
            bg_assigned = np.zeros(len(points_valid), dtype=bool)

            for mask_idx, (mask, color) in enumerate(zip(masks, colors)):
                mask_resized = cv2.resize(mask.astype(np.uint8), (camera.width, camera.height))
                mask_indices = mask_resized[v_valid, u_valid] > 0.5

                if np.count_nonzero(mask_indices) > 0:
                    car_points = points_valid[mask_indices]
                    bg_assigned[mask_indices] = True
                    car_color = np.array([color[2], color[1], color[0]]) / 255.0  # BGR to RGB
                    pcd_car = create_point_cloud(car_points, np.tile(car_color, (len(car_points), 1)))
                    geometries.append(pcd_car)

            # Add background points
            remaining_points = points_valid[~bg_assigned]
            if len(remaining_points) > 0:
                pcd_bg = create_point_cloud(remaining_points, np.tile([0.5, 0.5, 0.5], (len(remaining_points), 1)))
                geometries.append(pcd_bg)

            # Match and add 3D bounding boxes
            matched_pairs = match_detections_to_bboxes(boxes, bboxes_3d, colors, camera)
            for corners, color in matched_pairs:
                bbox_lines = create_bbox_lines(corners, color=color)
                geometries.append(bbox_lines)

        else:
            # No detections - show raw point cloud with all bboxes
            pcd_raw = create_point_cloud(points_valid)
            geometries.append(pcd_raw)

            for bbox in bboxes_3d:
                if 'corners_velo' in bbox:
                    corners = np.array(bbox['corners_velo'])
                    bbox_lines = create_bbox_lines(corners, color=[1, 0, 0])
                    geometries.append(bbox_lines)

        # Visualize
        print(f"Visualizing frame {frame} with {len(geometries)} objects")
        o3d.visualization.draw_geometries(geometries, window_name=f"Frame {frame}", width=800, height=600)

        # Show segmented image
        #if seg_image is not None:
            #cv2.imshow(f'Segmented Image - Frame {frame}', seg_image)
            #cv2.waitKey(1)

        input("Press Enter to continue...")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    process_frame(seq=0, cam_id=0)