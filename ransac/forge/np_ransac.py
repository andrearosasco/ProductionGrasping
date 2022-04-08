import multiprocessing

import numpy as np
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
from torch import nn

from utils.timer import Timer


def aux(points, eps):
    idx = np.random.choice(points.shape[0], 3, replace=False)
    subset = points[idx]
    # plane = numpy_svd(subset)
    v = np.cross(subset[1] - subset[0], subset[2] - subset[0])
    v = v / np.linalg.norm(v)
    p = - (v @ np.mean(subset, axis=0, keepdims=True).T)
    # v, p = plane[:3], plane[3:]

    distances = np.abs((points @ v) - p)
    keep = distances <= eps

    distances[~keep] = 0
    score = np.sum(keep)

    return score, np.concatenate([v, p])


def parallel_ransac(points, num_points, eps, iterations, pool):
    min_dist = 1000
    max_score = 0
    res = None

    scores, planes = zip(*pool.starmap(aux, [(points, eps) for _ in range(iterations)]))
    return planes[np.argmax(scores)]
    # for i in zip(*pool.imap_unordered(aux, range(10))):
    #     print(i)

    # for i in range(iterations):
    #     idx = np.random.choice(points.shape[0], 3, replace=False)
    #     subset = points[idx]
    #     # plane = numpy_svd(subset)
    #     v = np.cross(subset[1] - subset[0], subset[2] - subset[0])
    #     v = v / np.linalg.norm(v)
    #     p = - (v @ np.mean(subset, axis=0, keepdims=True).T)
    #     # v, p = plane[:3], plane[3:]
    #
    #     distances = np.abs((points @ v) - p)
    #     keep = distances <= eps
    #
    #     distances[~keep] = 0
    #     score = np.sum(keep)
    #
    #     # if dist < min_dist:
    #     #     min_dist = dist
    #     #     res = plane
    #     if score > max_score:
    #         max_score = score
    #         res = np.concatenate([v, p])

    # aux1 = PointCloud()
    # aux1.points = Vector3dVector(points[~np.isin(np.arange(0, points.shape[0]), idx) & ~keep])
    # aux1.paint_uniform_color([0, 1, 0])
    #
    # aux2 = plot_plane(*np.concatenate([v, p]))
    # aux2.paint_uniform_color([0, 0, 1])
    #
    # aux3 = PointCloud()
    # aux3.points = Vector3dVector(subset)
    # aux3.paint_uniform_color([1, 0, 0])
    #
    # aux4 = PointCloud()
    # aux4.points = Vector3dVector(points[keep])
    # aux4.paint_uniform_color([1, 1, 0])
    #
    # draw_geometries([aux1, aux2, aux3, aux4])

    return res


def sequential_ransac(points, num_points, eps, iterations):
    max_score = 0
    res = None

    for i in range(iterations):
        idx = np.random.choice(points.shape[0], 3, replace=False)
        subset = points[idx]

        v = np.cross(subset[1] - subset[0], subset[2] - subset[0])
        v = v / np.linalg.norm(v)
        p = - (v @ np.mean(subset, axis=0, keepdims=True).T)

        distances = np.abs((points @ v) - p)
        keep = distances <= eps

        distances[~keep] = 0
        score = np.sum(keep)

        if score > max_score:
            max_score = score
            res = np.concatenate([v, p])

    return res


def parallel_ransac2(points, num_points, eps, iterations):
    idx = np.random.randint(0, points.shape[0], size=iterations * 3)
    subsets = points[idx].reshape(iterations, 3, 3)
    v1 = subsets[:, 1] - subsets[:, 0]
    v2 = subsets[:, 2] - subsets[:, 0]
    p = subsets[:, 0]

    normals = np.cross(v1, v2)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    d = - np.sum(normals * p, axis=1, keepdims=True)

    distances = np.abs((normals @ points.T) - d)
    scores = np.sum(distances < eps, axis=1)

    best = np.argmax(scores)

    return np.concatenate([normals[best], d[best]])


def parallel_ransac3(points, subsets, eps):
    device = points.device

    v1 = subsets[:, 1] - subsets[:, 0]
    v2 = subsets[:, 2] - subsets[:, 0]
    p = subsets[:, 0]

    normals = cross_product(v1, v2)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)
    d = - torch.sum(normals * p, dim=1, keepdim=True)

    distances = torch.abs(normals @ points.transpose(0, 1) + d)

    scores = torch.sum(distances < eps, dim=1)

    return torch.concat([normals, d], dim=1), scores


def cross_product(v1, v2):
    return torch.stack([v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1],
                        v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2],
                        v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]], dim=1)


def plot_plane(a, b, c, d):
    xy = (np.random.rand(1000000, 2) - 0.5) * 2
    z = - ((a * xy[..., 0] + b * xy[..., 1] + d) / c)

    xy = xy[(-1 < z) & (z < 1)]
    z = z[(-1 < z) & (z < 1)]

    plane = np.concatenate([xy, z[..., None]], axis=1)

    aux = PointCloud()
    aux.points = Vector3dVector(plane)
    return aux


def main():
    # The sequential algorithm takes 9 fps while the parallel one,
    # with 8 threads, runs at 30 fps.
    points = np.load('./ransac/assets/test_pcd.npy')

    pool = multiprocessing.Pool(8)  # The thread pool must be an argument so that it's not re-initialized every time

    for _ in range(100):
        with Timer('ransac'):
            # for _ in range(6):
            # sequential_ransac(points, 3, 0.01, 2000)
            points = torch.tensor(points)
            idx = torch.randint(0, points.shape[0], size=[1000 * 3])
            subsets = points[idx].reshape(1000, 3, 3)
            plane = parallel_ransac3(points, subsets, 0.005)
            # plane = parallel_ransac(points, 3, 0.01, 2000, pool)
            # plane = sequential_ransac(points, 3, 0.01, 2000)

            # aux = PointCloud()
            # aux.points = Vector3dVector(points)
            # draw_geometries([aux, plot_plane(*plane)])

    print(1 / (Timer.timers['ransac'] / Timer.counters['ransac']))


if __name__ == '__main__':
    # speed('cpu')
    # output()
    main()
