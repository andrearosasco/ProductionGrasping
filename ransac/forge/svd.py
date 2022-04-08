import copy

import numpy as np
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
from torch import nn

from utils.timer import Timer

class Ransac(nn.Module):

    def __init__(self, n_points, eps, iterations):
        super().__init__()
        self.n_points = n_points
        self.eps = eps
        self.iterations = iterations

    def forward(self, x):
        return fit_plane(x)

def fit_plane(points, device='cpu'):
    A = torch.concat([points[..., :2], torch.ones([points.shape[0], 1], device=device)], dim=-1)
    b = points[..., 2:]

    # q, r = torch.linalg.qr(A, mode='reduced')
    q, r = givens_rotation(A)

    p = q.T @ b
    x = torch.inverse(r) @ p

    return x

def speed():
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

    for _ in range(1000):
        with Timer('fit_plane'):
            plane_model = fit_plane(points)

    for k in Timer.timers:
        print(k, Timer.timers[k] / Timer.counters[k], Timer.timers[k])


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

        plane_model = fit_plane(points, device='cpu')

        aux = PointCloud()
        aux.points = Vector3dVector(points)
        aux.paint_uniform_color([1, 0, 0])

        aux2 = plot_plane2(*plane_model)
        aux2.paint_uniform_color([0, 1, 0])

        draw_geometries([aux, aux2])

def build():
    ransac = Ransac(1000, 0.01 * 1.5, 1000)

    m = 20  # number of points
    delta = 0  # size of random displacement
    origin = torch.rand(3, 1)  # random origin for the plane
    basis = torch.rand(3, 2)  # random basis vectors for the plane
    coefficients = torch.rand(2, m)  # random coefficients for points on the plane

    # generate random points on the plane and add random displacement
    points = basis @ coefficients \
             + torch.tile(origin, (1, m)) \
             + delta * torch.rand(3, m)

    torch.onnx.export(ransac, points, './delete.onnx', input_names=['input'],
                      output_names=[f'output'], opset_version=11)


def plot_plane(a, b, c, d):
    xy = np.random.rand(1000, 2)
    z = - ((a * xy[..., 0] + b * xy[..., 1] + d) / c)

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

def givens_rotation(A):
    """
    QR-decomposition of rectangular matrix A using the Givens rotation method.
    """

    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
    n, m = A.shape
    Q = torch.eye(n)
    R = copy.deepcopy(A)

    rows, cols = torch.tril_indices(n, m, -1)
    for (row, col) in zip(rows, cols):
        # If the subdiagonal element is nonzero, then compute the nonzero
        # components of the rotation matrix
        if R[row, col] != 0:
            r = torch.sqrt(R[col, col]**2 + R[row, col]**2)
            c, s = R[col, col]/r, -R[row, col]/r

            # The rotation matrix is highly discharged, so it makes no sense
            # to calculate the total matrix product
            R[col], R[row] = R[col]*c + R[row]*(-s), R[col]*s + R[row]*c
            Q[:, col], Q[:, row] = Q[:, col]*c + Q[:, row]*(-s), Q[:, col]*s + Q[:, row]*c

    return Q[:, :m], R[:m]

def Hausholder(A):
    """
    QR-decomposition of a rectangular matrix A using the Householder reflection method.
    """

    # Initialization of the orthogonal matrix Q and the upper triangular matrix R
    n, m = A.shape
    Q = np.eye(n)
    R = np.copy(A)

    for k in range(m):
        v = np.copy(R[k:, k]).reshape((n-k, 1))
        v[0] = v[0] + np.sign(v[0]) * np.linalg.norm(v)
        v = v / np.linalg.norm(v)
        R[k:, k:] = R[k:, k:] - 2 * v @ v.T @ R[k:, k:]
        Q[k:] = Q[k:] - 2 * v @ v.T @ Q[k:]

    return Q[:m].T, R[:m]


# To check the solutions, we use the standard deviation of SME
def SME(A, b, x):
    return 1/max(b) * np.sqrt(1/len(b) * np.sum(abs(np.dot(A, x) - b) ** 2))


if __name__ == '__main__':
     output()

