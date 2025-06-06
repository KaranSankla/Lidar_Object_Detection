from ultralytics import YOLO
import os
import cv2

# Path to input images
image_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\images'

# Path to save results
output_folder_path = r'D:\Masters\ComputerVision\CodingEX\Task2\KITTI_Selection\KITTI_Selection\results'
os.makedirs(output_folder_path, exist_ok=True)

# Load a YOLO model
model = YOLO("yolo11x.pt")  # Replace with your YOLOv8 model

# Class index for "car" (check model.names for correct index if unsure)
car_class_index = 2  # Update this if necessary based on your model's classes

# Loop through each file in the input folder
for filename in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, filename)

    # Check if the file is an image
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    # Predict using YOLO
    results = model.predict(source=image_path, conf=0.5)

    # Process predictions to filter for "car" class
    for result in results:
        car_detections = [
            box for box in result.boxes if int(box.cls) == car_class_index
        ]

        # Load the original image for visualization
        img = cv2.imread(image_path)

        for box in car_detections:
            # Get bounding box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integers
            confidence = box.conf[0]  # Confidence score

            # Draw bounding box and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"Car {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            print(f"Car detected at: {x1, y1, x2, y2} with confidence {confidence}")

        # Save the image with detections
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, img)

print("Detection and saving completed.")
