from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries

from shape_reconstruction.pytorch.utils.input import RealSense

if __name__ == '__main__':
    root = Path('../Segmentation/data/real/data/')

    for i in range(22):
        depth = np.array(Image.open(f'./shape_reconstruction/tensorrt/assets/real_data/partial{i}.png'))
        pc = RealSense.pointcloud(depth.astype(np.uint16))

        aux = PointCloud()
        aux.points = Vector3dVector(pc)

        _, ind = aux.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)

        inlier_cloud = aux.select_by_index(ind)

        pc = np.array(inlier_cloud.points)
        idx = np.random.choice(pc.shape[0], (2024), replace=False)
        pc = pc[idx]

        mean = np.mean(pc, axis=0)
        var = np.sqrt(np.max(np.sum((pc - mean) ** 2, axis=1)))
        pc = (pc - mean) / (var * 2)
        pc[..., -1] = -pc[..., -1]

        np.save(f'./shape_reconstruction/tensorrt/assets/real_data/partial{i}', pc)
