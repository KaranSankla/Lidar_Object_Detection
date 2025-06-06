import cv2
from ultralytics import YOLO
import os

# Paths
image_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\images'
image_labels_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\labels'
output_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\results'
os.makedirs(output_folder_path, exist_ok=True)

# Load YOLO model
model = YOLO("yolo11x.pt")

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

# Paths and dependencies remain the same...

for filename in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, filename)
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {filename}")
        continue

    # Predict using YOLO
    results = model.predict(source=image_path, conf=0.5)

    detected_boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls == 2:  # Class ID for "car"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_boxes.append([x1, y1, x2, y2])

    # Get corresponding label file for ground truth
    label_file_name = filename.replace(".jpg", ".txt").replace(".png", ".txt")
    label_file_path = os.path.join(image_labels_path, label_file_name)

    ground_truth_boxes = []
    if os.path.exists(label_file_path):
        with open(label_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                if len(data) < 5:
                    print(f"Invalid data in file {label_file_name}: {line}")
                    continue
                class_id, x_min, y_min, x_max, y_max = data[:5]
                x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                ground_truth_boxes.append([x_min, y_min, x_max, y_max])

    # Draw boxes with IoU > 0.5
    for detected_box in detected_boxes:
        for ground_truth_box in ground_truth_boxes:
            iou = calculate_iou(detected_box, ground_truth_box)
            if iou > 0.5:
                # Draw the detected box (red)
                cv2.rectangle(img,
                              (detected_box[0], detected_box[1]),
                              (detected_box[2], detected_box[3]),
                              (0, 0, 255),
                              2)
                cv2.putText(img,
                            f"Det IoU: {iou:.2f}",
                            (detected_box[0], detected_box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1)

                # Draw the ground truth box (green)
                cv2.rectangle(img,
                              (ground_truth_box[0], ground_truth_box[1]),
                              (ground_truth_box[2], ground_truth_box[3]),
                              (0, 255, 0),
                              2)
                # cv2.putText(img,
                #             "GT",
                #             (ground_truth_box[0], ground_truth_box[1] - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5,
                #             (0, 255, 0),
                #             1)

    # Save the output
    output_path = os.path.join(output_folder_path, filename)
    cv2.imwrite(output_path, img)
    print(f"Results saved to {output_path}")
