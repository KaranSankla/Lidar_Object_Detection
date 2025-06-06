import cv2
import numpy as np
import os
import math
from ultralytics import YOLO

# Paths
image_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\images'
image_labels_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\labels'
calibration_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\calib'
output_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\results'
os.makedirs(output_folder_path, exist_ok=True)


def calculate_iou(box1, box2):
    """f
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1: List or tuple with coordinates [x_min, y_min, x_max, y_max].
        box2: List or tuple with coordinates [x_min, y_min, x_max, y_max].
    Returns:
        IoU value as a float.
    """
    # Coordinates of the intersection rectangle
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Intersection area
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection_area = inter_width * inter_height

    # Areas of the individual bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union area
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    # IoU
    iou = intersection_area / union_area
    return iou


def load_intrinsic_matrix(calibration_file_path):
    try:
        intrinsic_matrix = np.loadtxt(calibration_file_path)
        return intrinsic_matrix
    except Exception as e:
        print(f"Error loading intrinsic matrix from {calibration_file_path}: {e}")
        return None


# Calculate distance based on bounding box and intrinsic parameters
def calculate_distance_aligned(intrinsic, bbox, camera_height=1.65):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_min, y_min, x_max, y_max = bbox
    c1 = (x_min, y_min)
    c2 = (x_max, y_min)
    c3 = (x_max, y_max)
    c4 = (x_min, y_max)

    # Midpoints
    m1 = ((x_min + x_max) / 2, y_min)
    m2 = (x_max, (y_min + y_max) / 2)
    m3 = ((x_min + x_max) / 2, y_max)
    m4 = (x_min, (y_min + y_max) / 2)

    # Combine all points
    points = [c1, c2, c3, c4, m1, m2, m3, m4]
    distances = []  # Store distances for all points
    for (u, v) in points:
        try:
            Z = (camera_height * fy) / (v - cy)  # Depth along Z-axis
            X = (u - cx) * Z / fx  # X-coordinate in world space
            distance = np.sqrt(X ** 2 + camera_height ** 2 + Z ** 2)
            distances.append(distance)
        except ZeroDivisionError:
            print(f"Division by zero for point ({u}, {v}). Check camera parameters or bounding box.")
            distances.append(float('inf'))  # Add infinity if division by zero occurs

    # Return the minimum distance among all points
    return min(distances)


model = YOLO("yolo11x.pt")


# Loop through images
for filename in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, filename)
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {filename}")
        continue

    results = model.predict(source=image_path, conf=0.5)
    ground_truth_boxes = []
    detected_boxes = []
    # Maintain a counter for unique IDs
    car_id_counter = 0

    # Dictionary to map detected car IDs to ground truth IDs and other information
    car_id_mapping = {}

    # Process YOLO detections
    for result in results:
        for box in result.boxes:
            if box.cls[0] == 2:  # Class ID for "car"
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # YOLO format conversion
                detected_boxes.append([x1, y1, x2, y2])
                confidence = box.conf[0]
                # if confidence > 0.5:
                calibration_file_name = filename.replace(".jpg", ".txt").replace(".png", ".txt")
                calibration_file_path = os.path.join(calibration_folder_path, calibration_file_name)

                # Load intrinsic matrix and calculate distance
                if os.path.exists(calibration_file_path):
                    intrinsic_matrix = load_intrinsic_matrix(calibration_file_path)
                    if intrinsic_matrix is not None:
                        distance_car = calculate_distance_aligned(intrinsic_matrix, [x1, y1, x2, y2])
                    else:
                        distance_car = float('inf')
                else:
                    print(f"Calibration file not found: {calibration_file_path}")
                    distance_car = float('inf')

                # Assign a unique ID to the detected car
                car_id_counter += 1
                car_id_mapping[car_id_counter] = {
                    "bbox": [x1, y1, x2, y2],
                    "distance": distance_car,
                    "matched_gt": None,
                }

                # Annotate detected car with ID and distance
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"ID: {car_id_counter}, {distance_car:.2f}m",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 255),
                    1,
                )

    # Process ground truth labels
    label_file_name = filename.replace(".jpg", ".txt").replace(".png", ".txt")
    label_file_path = os.path.join(image_labels_path, label_file_name)
    if os.path.exists(label_file_path):
        with open(label_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) < 6:
                    continue
                class_id, x_min, y_min, x_max, y_max, GT_distance = data[:6]
                x_min, y_min, x_max, y_max = map(int, map(float, [x_min, y_min, x_max, y_max]))
                GT_distance = float(GT_distance)
                ground_truth_boxes.append([x_min, y_min, x_max, y_max])

                # Annotate ground truth box with distance
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"GT: {GT_distance:.2f}m",
                    (x_min, y_min - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1,
                )

    # Match detected cars with ground truth boxes
    for car_id, detected_data in car_id_mapping.items():
        detected_box = detected_data["bbox"]
        for gt_index, ground_truth_box in enumerate(ground_truth_boxes):
            iou = calculate_iou(detected_box, ground_truth_box)
            if iou > 0.5:  # Threshold for matching
                detected_data["matched_gt"] = gt_index
                print(f"Detected car ID {car_id} matches GT box {gt_index} with IoU: {iou:.2f}")

                # Annotate matched ID
                x1, y1, x2, y2 = detected_box
                cv2.putText(
                    img,
                    f"Matched GT: {gt_index}, IoU: {iou:.2f}",
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )
                break

    # Save annotated image
    output_path = os.path.join(output_folder_path, filename)
    cv2.imwrite(output_path, img)
    print(f"Results saved to {output_path}")
