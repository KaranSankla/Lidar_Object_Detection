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
    for result in results:
        for box in result.boxes:
            if box.cls[0] == 2:  # Class ID for "car"
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # YOLO format conversion
                detected_boxes.append([x1, y1, x2, y2])
                calibration_file_name = filename.replace(".jpg", ".txt").replace(".png", ".txt")
                calibration_file_path = os.path.join(calibration_folder_path, calibration_file_name)
                if os.path.exists(calibration_file_path):
                    intrinsic_matrix = load_intrinsic_matrix(calibration_file_path)
                    if intrinsic_matrix is not None:
                        distance_car = calculate_distance_aligned(intrinsic_matrix, [x1, y1, x2, y2])

                        print(f"Detected car at distance: {distance_car:.2f}m")
                else:
                    print(f"Calibration file not found: {calibration_file_path}")

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"YOLO: {distance_car:.2f}m",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 255),
                    1,
                )
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
                x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])  # Convert to floats
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                ground_truth_boxes.append([x_min, y_min, x_max, y_max])  # Convert to integers
                GT_distance = float(GT_distance)

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
                print(f"Detected GT: {GT_distance:.2f}m")



    for detected_box in detected_boxes:
        for ground_truth_box in ground_truth_boxes:
            iou = calculate_iou(detected_box, ground_truth_box)
            if iou > 0.4:
                print(f"IoU between {detected_box} and {ground_truth_box}: {iou:.2f}")


                # Save annotated image
                output_path = os.path.join(output_folder_path, filename)
                cv2.imwrite(output_path, img)
                print(f"Results saved to {output_path}")

