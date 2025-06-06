
#################
## Import modules
#################
import sys
# walk directories
import glob
# access to OS functionality
import os
# copy things
import copy

import cv2
# numpy
import numpy as np
import torch
import random
# open3d
import open3d
from ultralytics import YOLO
# matplotlib for colormaps
import matplotlib.cm
import matplotlib.pyplot as plt
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# struct for reading binary ply files
import struct
output_dir = '/home/karan-sankla/LIDAR_RADAR/Predictions'
model = YOLO('yolo11x-seg.pt')

os.environ['KITTI360_DATASET'] = '/home/karan-sankla/LIDAR_RADAR/KITTI360_sample'

seq = 0
# the main class that loads raw 3D scans
class Kitti360Viewer3DRaw(object):

    # Constructor
    def __init__(self, seq=0, mode='velodyne'):

        if 'KITTI360_DATASET' in os.environ:
            kitti360Path = os.environ['KITTI360_DATASET']
        else:
            kitti360Path = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), '..', '..')

        if mode == 'velodyne':
            self.sensor_dir = 'velodyne_points'
        elif mode == 'sick':
            self.sensor_dir = 'sick_points'
        else:
            raise RuntimeError('Unknown sensor type!')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath = os.path.join(kitti360Path, 'data_3d_raw', sequence, self.sensor_dir, 'data')

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd, [-1, 4])
        return pcd

def image_segmentation(image):
    results = model.predict(image, device='0', classes=2, retina_masks=True)
    all_masks = []

    for result in results:
        img = result.orig_img.copy()
        boxes = result.boxes.xyxy.cpu().numpy()

        if result.masks is None or len(boxes) == 0:
            return None, None

        masks = result.masks.data.cpu().numpy()
        car_id_count = 0

        for mask, box in zip(masks, boxes):
            car_id_count += 1
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color_mask = np.zeros_like(img, dtype=np.uint8)
            all_masks = masks
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

        return img, all_masks

    return None, None  # <- ensure a tuple is always returned



def projectVeloToImage(seq=0):
    sequence = '2013_05_28_drive_%04d_sync' % seq
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq)
    kitti360Path = os.environ['KITTI360_DATASET']

    camera = CameraPerspective(kitti360Path, sequence, 0)

    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    fileCameraToPose = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_pose.txt')

    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    # velodyne to all cameras
    TrVeloToCam = {}
    for k, v in TrCamToPose.items():
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[f'image_{cam_id:02d}']
        TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

    TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
    cm = plt.get_cmap('jet')

    available_files = sorted(glob.glob(os.path.join(velo.raw3DPcdPath, '*.bin')))
    for file in available_files:
        frame = int(os.path.basename(file).split('.')[0])
        points = velo.loadVelodyneData(frame)
        points[:, 3] = 1

        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:, :3]

        u, v, depth = camera.cam2image(pointsCam.T)
        u = u.astype(int)
        v = v.astype(int)

        # Load RGB image
        imagePath = os.path.join(kitti360Path, 'data_2d_raw', sequence, 'image_%02d' % cam_id,
                                 'data_rect' if cam_id in [0, 1] else 'data_rgb',
                                 '%010d.png' % frame)
        if not os.path.isfile(imagePath):
            raise RuntimeError(f'Image file {imagePath} does not exist!')
        bgr_image = cv2.imread(imagePath)
        masking_image, masks = image_segmentation(bgr_image.copy())
        if masks is None:
            print(f"[INFO] No cars detected in frame {frame}, skipping.")
            continue

        # Associate projected points to car masks
        valid = np.logical_and.reduce((
            u >= 0, u < camera.width,
            v >= 0, v < camera.height,
            depth > 0, depth < 30
        ))

        per_car_depth_maps = []

        for i, mask in enumerate(masks):
            depthMap = np.zeros((camera.height, camera.width))

            for idx in np.where(valid)[0]:
                x, y = u[idx], v[idx]
                if mask[y, x] > 0.5:
                    depthMap[y, x] = depth[idx]

            per_car_depth_maps.append((i + 1, depthMap))


        for car_id, depthMap in per_car_depth_maps:
            if np.max(depthMap) == 0:
                continue  # Skip if no depth points
            depthImage = cm(depthMap / np.max(depthMap))[..., :3]
            image_withseg = np.array(masking_image) / 255.
            image_withseg[depthMap > 0] = depthImage[depthMap > 0]
            image_withseg = np.uint8(image_withseg * 255)
            image_withseg = cv2.cvtColor(image_withseg, cv2.COLOR_RGB2BGR)


            layout = (2, 1) if cam_id in [0, 1] else (1, 2)
            fig, axs = plt.subplots(*layout, figsize=(18, 12))
            axs[0].imshow(depthMap, cmap='jet')
            axs[0].set_title(f"Depth Map - Car {car_id} (Frame {frame})")
            axs[0].axis('off')
            axs[1].imshow(image_withseg)
            axs[1].set_title('Depth Overlaid on Segmented Image')
            axs[1].axis('off')
            #plt.show()
            save_path = os.path.join(output_dir, f'{frame:010d},depth_map_car_{car_id:02d}_.png')
            fig.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
            plt.close(fig)


        # Depth overlay visualization
        # depthMap = np.zeros((camera.height, camera.width))
        # mask = valid
        # depthMap[v[mask], u[mask]] = depth[mask]
        #
        # image_withseg = np.array(masking_image) / 255.
        # depthImage = cm(depthMap / depthMap.max())[..., :3]
        # image_withseg[depthMap > 0] = depthImage[depthMap > 0]
        # image_withseg = np.uint8(image_withseg * 255)
        # image_withseg = cv2.cvtColor(image_withseg, cv2.COLOR_RGB2BGR)
        #
        # layout = (2, 1) if cam_id in [0, 1] else (1, 2)
        # fig, axs = plt.subplots(*layout, figsize=(18, 12))
        # axs[0].imshow(depthMap, cmap='jet')
        # axs[0].set_title('Projected Depth')
        # axs[0].axis('off')
        # axs[1].imshow(image_withseg)
        # axs[1].set_title('Depth Overlaid on Segmented Image')
        # axs[1].axis('off')
        # plt.suptitle(f'Sequence {seq:04d}, Camera {cam_id}, Frame {frame:010d}')
        # plt.show()
        #
        # save_path = os.path.join(output_dir, 'frame_%010d.png' % frame)
        # cv2.imwrite(save_path, image_withseg)



if __name__ == '__main__':

    visualizeIn2D = True
    # sequence index
    seq = 0
    cam_id = 0

    # visualize raw 3D velodyne scans in 2D
    if visualizeIn2D:
        projectVeloToImage(seq=seq)

    # visualize raw 3D scans in 3D
    else:
        mode = 'sick'
        frame = 1000

        v = Kitti360Viewer3DRaw(mode=mode)
        if mode == 'velodyne':
            points = v.loadVelodyneData(frame)
        elif mode == 'sick':
            points = v.loadSickData(frame)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[:, :3])

        open3d.visualization.draw_geometries([pcd])