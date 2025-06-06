# KITTI-360 dataset sample

This is sample of [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php) dataset. It contains the 20 frames of Cam 1 and 2, as well as the point clouds of the Velodyne LiDAR.

For how the data is structured, please consult the official [documentation](https://www.cvlibs.net/datasets/kitti-360/documentation.php)

You can also take a look at the [GitHub-Repo](https://github.com/autonomousvision/kitti360Scripts) to find some helper functions (like projecting the point cloud to the image space and how to load data) already implemented.

The folder bboxes_3D_cam0 was created by us. It contains a JSON-File for every frame. Each object has an index, which is a unique ID for a car, as well as 8 corners for a 3D bounding box. The bounding box is given in the coordinate space of cam0.

The corner points are ordered like this:

     Top view:
       4 -------- 5
      /          /
     7 -------- 6

     Bottom view:
       0 -------- 1
      /          /
     3 -------- 2
