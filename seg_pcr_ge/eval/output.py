import numpy as np
import torch
import tqdm
from open3d.cpu.pybind.geometry import PointCloud, TriangleMesh
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
import open3d as o3d
from scipy.spatial.transform import Rotation

from ransac.utils.dataset import TestSet
from ransac.utils.grasp_estimator import GraspEstimator as GE1
from seg_pcr_ge.delete import GraspEstimator as GE2
from shape_reconstruction.pytorch.utils.pose_generator import PoseGenerator
from shape_reconstruction.tensorrt.utils.decoder import Decoder
from shape_reconstruction.tensorrt.utils.inference import Infer
from utils.timer import Timer


class DataConfig:
    dataset_path = "data/ShapeNetCore.v2"
    partial_points = 2024
    multiplier_complete_sampling = 50
    implicit_input_dimension = 8192
    dist = [0.1, 0.4, 0.5]
    noise_rate = 0.01
    tolerance = 0.0
    train_samples = 10000
    val_samples = 1024

    n_classes = 1


def main():
    a = torch.zeros([1]).to('cuda')
    backbone = Infer('./shape_reconstruction/tensorrt/assets/pcr.engine')
    decoder = Decoder()
    grasp_estimator = GE2()
    pose_generator = PoseGenerator()

    res = []
    rotations = [Rotation.from_euler('yx', [r, 45], degrees=True) for r in range(0, 360, 10)]
    valid_set = TestSet(DataConfig, box_width=500, box_depth=500, box_height=200, rotations=rotations)
    # valid_set = BoxNet(DataConfig, n_samples=100)
    for i, partial in tqdm.tqdm(enumerate(list(valid_set))):

        fast_weights = backbone(partial.numpy())
        res = decoder(fast_weights)
        poses = grasp_estimator.find_poses(res * np.array([1, 1, -1]), 0.005, 5000)
        # poses2 = pose_generator.find_poses(res * np.array([1, 1, -1]), 0.01, iterations=1000)

        aux1 = PointCloud()
        aux1.points = Vector3dVector(res * np.array([1, 1, -1]))
        aux1.paint_uniform_color(np.random.rand(3))

        aux2 = PointCloud()
        aux2.points = Vector3dVector(partial.numpy() * np.array([1, 1, -1]))
        aux2.paint_uniform_color(np.random.rand(3))

        # for s, poses in [['new', poses1], ['old', poses2]]:
        #     print(s)
        c1, r1, c2, r2, planes, lines, vertices = poses
        right_hand = TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.1).\
            rotate(r1, center=[0, 0, 0]).translate(c1, relative=False)
        left_hand = TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.1). \
            rotate(r2, center=[0, 0, 0]).translate(c2, relative=False)

        # print(np.array([1, 0, 0]) @ r1)
        # if i >= 20:
        draw_geometries([aux1, aux2, right_hand, left_hand, TriangleMesh.create_coordinate_frame(size=0.5)] +
                        [TriangleMesh.create_coordinate_frame(origin=v, size=0.1) for v in vertices] +
                        [plot_line(line) for line in lines])
        pass

    for k in Timer.timers:
        print(1 / (Timer.timers[k] / Timer.counters[k]))

# def viz(points, R1, R2, c1, c2):
#     pc = PointCloud()
#     pc.points = Vector3dVector(points)
#     right_hand = TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.1). \
#         rotate(R2 @ R1, center=[0, 0, 0]).translate(c1, relative=False)
#     left_hand = TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=0.1). \
#         rotate(R2 @ R1, center=[0, 0, 0]).translate(c2, relative=False)
#
#     draw_geometries([pc, right_hand, left_hand, TriangleMesh.create_coordinate_frame(size=0.5)])
#     pass


def plot_plane(a, b, c, d):
    xy = (np.random.rand(1000000, 2) - 0.5) * 2
    z = - ((a * xy[..., 0] + b * xy[..., 1] + d) / c)

    xy = xy[(-1 < z) & (z < 1)]
    z = z[(-1 < z) & (z < 1)]

    plane = np.concatenate([xy, z[..., None]], axis=1)

    aux = PointCloud()
    aux.points = Vector3dVector(plane)
    return aux

def plot_line(line):
    t = (np.random.rand(1000000, 1) - 0.5) * 2
    l0, l = line

    aux = PointCloud()
    aux.points = Vector3dVector(l0 + t * l)
    return aux

if __name__ == '__main__':
    main()