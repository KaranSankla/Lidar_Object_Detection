# Lidar_Object_Detection
# 🚗 3D Car Detection & Segmentation with LiDAR and YOLOv8 on KITTI-360

This project performs instance-level car segmentation and 3D visualization by combining LiDAR point clouds with camera images from the KITTI-360 dataset. Using YOLOv8 for 2D detection and Open3D for 3D rendering, it maps segmented cars onto the LiDAR point cloud and evaluates detection accuracy using point-level metrics.

---

## 🔍 Features

- 2D car detection using YOLOv8 on KITTI-360 RGB images
- LiDAR-to-camera projection for point cloud fusion
- Instance segmentation masks mapped to 3D LiDAR points
- 3D bounding box visualization using Open3D
- Point-level evaluation metrics: inside/outside %, IoU, precision, recall
- Supports mask erosion for robust segmentation matching

---

## 🧠 Technologies Used

- Python 3.8+
- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics)
- Open3D for 3D point cloud visualization
- NumPy, OpenCV, Matplotlib
- KITTI-360 Python API and annotation tools

---

   
## 📊 Evaluation Metrics

- For each detected car:

- Inside Percentage: % of object points within predicted 3D bbox

- Outside Percentage: % of object points outside the box

- Match Ratio: inside / outside point ratio
  
---

## ✨ Acknowledgements

- KITTI-360 dataset by Karlsruhe Institute of Technology (KIT)

- Ultralytics YOLOv8

- Open3D Visualization Library
  
- Python community for powerful open-source tools
  
---
