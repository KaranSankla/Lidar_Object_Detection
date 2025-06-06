import cv2
import numpy as np
import os
import random

import torch
from ultralytics import YOLO

input_dir = '/home/karan-sankla/LIDAR_RADAR/Predictions/data'
output_dir = '/home/karan-sankla/LIDAR_RADAR/Predictions'
os.makedirs(output_dir, exist_ok=True)

model = YOLO('yolo11x-seg.pt')

for image_file in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_file)

    # Run prediction
    results = model.predict(image_path, device='0', classes=2, retina_masks = True, )

    for result in results:
        img = result.orig_img.copy()  # Get the plotted image once
        boxes = result.boxes.xyxy.cpu().numpy()

        if result.masks is not None:
            masks = result.masks.data
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            car_id_count = 0

            for mask, box in zip(masks, boxes):
                car_id_count += 1
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                color_mask = np.zeros_like(img,dtype=np.uint8)

                color_mask[mask > 0.5] = color

                img = cv2.addWeighted(img, 1.0, color_mask, 0.5, 0)
                x1, y1, x2, y2 = map(int, box)

                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                y_indices, x_indices = np.where(mask > 0.5)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x1, y1 = np.min(x_indices), np.min(y_indices)
                    x2, y2 = np.max(x_indices), np.max(y_indices)

                    cv2.putText(img, f'Car {car_id_count}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save the final image
        save_path = os.path.join(output_dir, image_file)
        cv2.imwrite(save_path, img)

cv2.destroyAllWindows()
