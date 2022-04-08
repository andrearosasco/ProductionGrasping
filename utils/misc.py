import copy

import cv2
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries


def draw_pcs(*pcs):
    res = []
    for pc in pcs:
        o3d_pc = PointCloud()
        o3d_pc.points = Vector3dVector(pc)
        o3d_pc.paint_uniform_color(np.random.rand(3))
        res.append(o3d_pc)
    draw_geometries(res)


def draw_mask(rgb, mask):
    overlay = copy.deepcopy(rgb)
    if np.any(mask == 1):
        overlay[mask == 1] = np.array([0, 0, 128])
    res1 = cv2.addWeighted(rgb, 1, overlay, 0.5, 0)

    res2 = copy.deepcopy(rgb)
    res2[mask == 0] = np.array([0, 0, 0])

    return cv2.cvtColor(res1, cv2.COLOR_RGB2BGR), cv2.cvtColor(res2, cv2.COLOR_RGB2BGR)
