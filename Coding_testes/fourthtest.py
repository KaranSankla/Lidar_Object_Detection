import os
import glob
import numpy as np
import cv2
import torch
from scipy.spatial.distance import cdist
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


def project_3d_bboxes_to_points(bboxes_3d, u_valid, v_valid, points_valid, camera, yolo_boxes, colors):
    """Project 3D bounding boxes and filter points that fall within their 2D projections, only for YOLO detected objects"""
    projected_pairs = []
    print('interating through yolo boxes:')

    if not bboxes_3d or yolo_boxes is None or len(yolo_boxes) == 0:
        return projected_pairs

    # For each YOLO detection, find the best matching 3D bbox
    for yolo_idx, yolo_box in enumerate(yolo_boxes):
        x1, y1, x2, y2 = yolo_box
        yolo_center_x = (x1 + x2) / 2
        yolo_center_y = (y1 + y2) / 2

        best_bbox_idx = -1
        min_distance = float('inf')

        # Find the 3D bbox whose projection center is closest to YOLO detection center
        for bbox_idx, bbox_3d in enumerate(bboxes_3d):
            if 'corners_cam0' not in bbox_3d or 'corners_velo' not in bbox_3d:
                continue

            # Project 3D bbox corners to 2D
            corners_3d = np.array(bbox_3d['corners_cam0'])
            u_proj, v_proj, depth_proj = camera.cam2image(corners_3d.T)

            # Check if bbox is visible
            valid_depth = depth_proj > 0
            if np.sum(valid_depth) == 0:
                continue

            # Get 2D bounding box center of the projection
            u_valid_corners = u_proj[valid_depth]
            v_valid_corners = v_proj[valid_depth]

            proj_center_x = (np.min(u_valid_corners) + np.max(u_valid_corners)) / 2
            proj_center_y = (np.min(v_valid_corners) + np.max(v_valid_corners)) / 2

            # Calculate distance between centers
            distance = np.sqrt((yolo_center_x - proj_center_x) ** 2 + (yolo_center_y - proj_center_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                best_bbox_idx = bbox_idx

        # If we found a matching 3D bbox, process it
        if best_bbox_idx >= 0 and min_distance < 100:  # Distance threshold
            bbox_3d = bboxes_3d[best_bbox_idx]
            corners_3d = np.array(bbox_3d['corners_cam0'])
            u_proj, v_proj, depth_proj = camera.cam2image(corners_3d.T)

            valid_depth = depth_proj > 0
            u_valid_corners = u_proj[valid_depth]
            v_valid_corners = v_proj[valid_depth]

            x_min, x_max = np.min(u_valid_corners), np.max(u_valid_corners)
            y_min, y_max = np.min(v_valid_corners), np.max(v_valid_corners)

            # Find points that fall within this 2D bounding box
            inside_bbox = ((u_valid >= x_min) & (u_valid <= x_max) &
                           (v_valid >= y_min) & (v_valid <= y_max))

            if np.count_nonzero(inside_bbox) > 0:
                # Get the 3D corners in Velodyne coordinates
                corners_velo = np.array(bbox_3d['corners_velo'])

                # Use unique color for each YOLO detection
                if colors and len(colors) > yolo_idx:
                    color = np.array(
                        [colors[yolo_idx][2], colors[yolo_idx][1], colors[yolo_idx][0]]) / 255.0  # BGR to RGB
                else:
                    # Generate unique colors if not enough YOLO colors
                    np.random.seed(yolo_idx)  # Ensure consistent colors
                    color = np.random.rand(3)


                # Get the points inside this bbox
                bbox_points = points_valid[inside_bbox]

                projected_pairs.append((corners_velo, color, bbox_points))


    return projected_pairs


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
        [0, 4], [1, 5], [2, 6], [3, 7]  # connections
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
            print(f"Loaded {len(points)} points")
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
        print(f"Found {len(bboxes_3d_filtered)} visible bboxes")

        # Transform points for projection
        points_homo = points.copy()
        points_homo[:, 3] = 1
        pointsCam = np.matmul(TrVeloToRect, points_homo.T).T[:, :3]
        u, v, depth = camera.cam2image(pointsCam.T)
        u, v = u.astype(int), v.astype(int)

        # Load and segment image
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, f'image_{cam_id:02d}',
                                 'data_rect' if cam_id in [0, 1] else 'data_rgb', f'{frame:010d}.png')
        print(f"Loading image {imagePath}")

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
        print(f"Found {len(points_valid)} valid points")

        # Create visualization
        geometries = []

        # Project 3D bboxes and get corresponding points (only for YOLO detected objects)
        projected_pairs = project_3d_bboxes_to_points(bboxes_3d, u_valid, v_valid, points_valid, camera, boxes, colors)
        print(f"Found {len(projected_pairs)} projected bounding boxes")

        # Track which points have been assigned to bboxes
        assigned_points = np.zeros(len(points_valid), dtype=bool)
        print(f"Assigning points to bboxes:")

        # Add bbox points and wireframes (only for YOLO detected objects)
        for corners_velo, color, bbox_points in projected_pairs:
            if len(bbox_points) > 0:
                # Create point cloud for this bbox with unique color
                pcd_bbox = create_point_cloud(bbox_points, np.tile(color, (len(bbox_points), 1)))
                geometries.append(pcd_bbox)

                # Mark these points as assigned
                # for i, point in enumerate(points_valid):
                #     if np.any([np.allclose(point, bp, atol=1e-6) for bp in bbox_points]):
                #         assigned_points[i] = True
                dists = cdist(points_valid, bbox_points)
                assigned_points[np.any(dists < 0.01, axis=1)] = True  # 1cm tolerance

                # Add bbox wireframe with matching color
                bbox_lines = create_bbox_lines(corners_velo, color=color)
                geometries.append(bbox_lines)

        # Add remaining unassigned points as background
        print('assigning remaining points to background:')
        remaining_points = points_valid[~assigned_points]
        if len(remaining_points) > 0:
            pcd_bg = create_point_cloud(remaining_points, np.tile([0.5, 0.5, 0.5], (len(remaining_points), 1)))
            geometries.append(pcd_bg)

        # If no YOLO detections or projected pairs, show raw point cloud only (no bboxes)
        if len(projected_pairs) == 0:
            pcd_raw = create_point_cloud(points_valid)
            geometries.append(pcd_raw)
        print(f"Created {len(geometries)} visualization objects")

        # Visualize
        print(f"Visualizing frame {frame} with {len(geometries)} objects")
        print(f"Found {len(projected_pairs)} projected bounding boxes")
        o3d.visualization.draw_geometries(geometries, window_name=f"Frame {frame}", width=800, height=600)

        # Show segmented image (optional)
        if seg_image is not None:
            cv2.imshow(f'Segmented Image - Frame {frame}', seg_image)
            cv2.waitKey(1)


        cv2.destroyAllWindows()


if __name__ == '__main__':
    process_frame(seq=0, cam_id=0)