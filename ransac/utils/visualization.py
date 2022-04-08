import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector


def plot_plane(a, b, c, d):
    xy = (np.random.rand(1000000, 2) - 0.5) * 2
    z = - ((a * xy[..., 0] + b * xy[..., 1] + d) / c)

    xy = xy[(-1 < z) & (z < 1)]
    z = z[(-1 < z) & (z < 1)]

    plane = np.concatenate([xy, z[..., None]], axis=1)

    aux = PointCloud()
    aux.points = Vector3dVector(plane)
    return aux