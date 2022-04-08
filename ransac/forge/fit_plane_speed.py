import copy
import multiprocessing

import numpy as np
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries

from utils.timer import Timer


def torch_qr(points):
    device = points.device

    A = torch.concat([points[..., :2], torch.ones([points.shape[0], 1], device=device)], dim=-1)
    b = points[..., 2:]

    q, r = torch.qr(A)

    p = q.T @ b
    x = torch.inverse(r) @ p

    return x

def torch_svd(points):
    # now find the best-fitting plane for the test points

    # subtract out the centroid and take the SVD
    center = torch.mean(points, dim=0, keepdim=True)

    svd = torch.svd(points - center)

    # Extract the left singular vectors
    left = svd.V
    left = left[..., -1]

    v = left / torch.norm(left)
    p = - (left @ center.T)
    # distance = (points[..., 3] @ v) - p

    return torch.concat([v, p])


def torch_lstsq(points):
    device = points.device

    A = torch.concat([points[..., :2], torch.ones([points.shape[0], 1], device=device)], dim=-1)
    b = points[..., 2:]

    x = torch.linalg.lstsq(A, b)

    return x.solution[:, 0]


def numpy_qr(points):
    A = np.concatenate([points[..., :2], np.ones([points.shape[0], 1])], axis=-1)
    b = points[..., 2:]

    q, r = np.linalg.qr(A, mode='reduced')

    p = q.T @ b
    x = np.linalg.inv(r) @ p

    return x

def numpy_svd(points):
    # subtract out the centroid and take the SVD
    center = np.mean(points, axis=0, keepdims=True)

    # If the points are co-planar we should set full_matrices=True
    # to get the right dimensions. However it's too slow and I'm willing
    # to take the risk
    svd = np.linalg.svd(points - center, full_matrices=False)

    # Extract the right singular vectors
    left = svd[2]

    # Last right eigenvector is the normal of the plane
    # The first two eigenvectors define a base for the plane
    # The first eigenvector is the direction with greatest std deviation
    left = left[-1]

    v = left / np.linalg.norm(left)
    p = - (left @ center.T)
    # distance = (points[..., 3] @ v) - p

    return np.concatenate([v, p])


def numpy_lstsq(points):
    A = np.concatenate([points[..., :2], np.ones([points.shape[0], 1])], axis=-1)
    b = points[..., 2:]

    x = np.linalg.lstsq(b, A)

    return x[0]


def speed(device):
    m = 1000  # number of points
    delta = 0.1  # size of random displacement
    origin = torch.rand(3, 1)  # random origin for the plane
    basis = torch.rand(3, 2)  # random basis vectors for the plane
    coefficients = torch.rand(2, m)  # random coefficients for points on the plane

    # generate random points on the plane and add random displacement
    points = basis @ coefficients \
             + torch.tile(origin, (1, m)) \
             + delta * torch.rand(3, m)

    points = points.T

    np_points = copy.deepcopy(points.numpy())
    tc_points = points.to(device)

    for _ in range(1000):
        # with Timer('torch_qr'):
        #     _ = torch_qr(tc_points)
        # with Timer('torch_svd'):
        #     _ = torch_svd(tc_points)
        # with Timer('torch_lstsq'):
        #     _ = torch_lstsq(tc_points)
        # with Timer('np_qr'):
        #     _ = numpy_qr(np_points)
        with Timer('np_svd'):
            _ = numpy_svd(np_points)
        # with Timer('np_lstsq'):
        #     _ = numpy_lstsq(np_points)

    for k in Timer.timers:
        print(k, Timer.timers[k] / Timer.counters[k], Timer.timers[k], 1 / Timer.timers[k])


def output():
    while True:
        m = 1000  # number of points
        delta = 0.1  # size of random displacement
        origin = torch.rand(3, 1)  # random origin for the plane
        basis = torch.rand(3, 2)  # random basis vectors for the plane
        coefficients = torch.rand(2, m)  # random coefficients for points on the plane

        # generate random points on the plane and add random displacement
        points = basis @ coefficients \
                 + torch.tile(origin, (1, m)) \
                 + delta * torch.rand(3, m)

        points = points.T

        plane_model1 = numpy_svd(points.numpy())
        plane_model2 = torch_svd(points).numpy()

        aux1 = PointCloud()
        aux1.points = Vector3dVector(points)
        aux1.paint_uniform_color([1, 0, 0])

        aux2 = plot_plane(*plane_model1)
        aux2.paint_uniform_color([0, 1, 0])

        aux3 = plot_plane(*plane_model2)
        aux3.paint_uniform_color([0, 0, 1])

        draw_geometries([aux1, aux2, aux3])


def plot_plane(a, b, c, d):
    xy = (np.random.rand(1000000, 2) - 0.5) * 2
    z = - ((a * xy[..., 0] + b * xy[..., 1] + d) / c)

    xy = xy[(-1 < z) & (z < 1)]
    z = z[(-1 < z) & (z < 1)]

    plane = np.concatenate([xy, z[..., None]], axis=1)

    aux = PointCloud()
    aux.points = Vector3dVector(plane)
    return aux


def plot_plane2(a, b, c):
    xy = (np.random.rand(1000, 2) - 0.5) * 10
    z = (a * xy[..., 0] + b * xy[..., 1] + c)

    plane = np.concatenate([xy, z[..., None]], axis=1)

    aux = PointCloud()
    aux.points = Vector3dVector(plane)
    return aux

def gen_points(num_points):
    num_points = int(num_points)

    delta = 0.1  # size of random displacement
    origin = torch.rand(3, 1)  # random origin for the plane
    basis = torch.rand(3, 2)  # random basis vectors for the plane
    coefficients = torch.rand(2, num_points)  # random coefficients for points on the plane

    # generate random points on the plane and add random displacement
    points = basis @ coefficients \
             + torch.tile(origin, (1, num_points)) \
             + delta * torch.rand(3, num_points)

    points = points.T.numpy()

    return points

if __name__ == '__main__':
    speed('cpu')
    # output()