import torch
from sklearn.cluster import DBSCAN

from shape_reconstruction.tensorrt.utils.inference import Infer as InferPcr
from utils.timer import Timer

a = torch.zeros([1]).to('cuda')
backbone = InferPcr('./shape_reconstruction/tensorrt/assets/pcr.engine')

from segmentation.tensorrt.utils.inference import Infer as InferSeg

model = InferSeg('./segmentation/tensorrt/assets/seg_int8.engine')

from shape_reconstruction.tensorrt.utils.decoder import Decoder

decoder = Decoder()

import copy

import numpy as np
import tqdm

import cv2

from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector, set_verbosity_level
from open3d.cpu.pybind.visualization import Visualizer, draw_geometries
from open3d.cpu.pybind.utility import Warning, Error

from utils.input import RealSense


def main():
    set_verbosity_level(Error)

    camera = RealSense()

    vis = Visualizer()
    vis.create_window('Pose Estimation')
    scene_pcd = PointCloud()
    part_pcd = PointCloud()
    pred_pcd = PointCloud()
    render_setup = False

    for _ in range(100000):
        with Timer('total'):
            # while True:
            rgb, depth = camera.read()

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            mask = model(rgb)
            mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)

            segmented_depth = copy.deepcopy(depth)
            segmented_depth[mask != 1] = 0

            # Adjust size
            distance = distance = segmented_depth[segmented_depth != 0].mean()
            if len(segmented_depth.nonzero()[0]) >= 4096:
                segmented_pc = RealSense.depth_pointcloud(segmented_depth)

                with Timer(name='downsample'):
                    # Downsample
                    idx = np.random.choice(segmented_pc.shape[0], 4096, replace=False)
                    downsampled_pc = segmented_pc[idx]

                with Timer(name='denoise'):
                  # Denoise
                    clustering = DBSCAN(eps=0.05, min_samples=10).fit(downsampled_pc) #0.1 10 are perfect but slow
                    close = clustering.labels_[downsampled_pc.argmax(axis=0)[2]]
                    denoised_pc = downsampled_pc[clustering.labels_ == close]

                # denoised_pc = downsampled_pc
                with Timer(name='adjust'):
                    if denoised_pc.shape[0] > 2024:
                        idx = np.random.choice(denoised_pc.shape[0], 2024, replace=False)
                        size_pc = denoised_pc[idx]
                    else:
                        print('Info: Partial Point Cloud padded')
                        diff = 2024 - denoised_pc.shape[0]
                        pad = np.zeros([diff, 3])
                        pad[:] = segmented_pc[0]
                        size_pc = np.vstack((denoised_pc, pad))

                with Timer(name='normalize'):
                    # Normalize
                    mean = np.mean(size_pc, axis=0)
                    var = np.sqrt(np.max(np.sum((size_pc - mean) ** 2, axis=1)))
                    normalized_pc = (size_pc - mean) / (var * 2)
                    normalized_pc[..., -1] = -normalized_pc[..., -1]

                with Timer(name='backbone'):
                    # Reconstruction
                    fast_weights = backbone(normalized_pc)
                with Timer(name='implicit function'):
                    res = decoder(fast_weights)
                    print(res.shape[0])
                if res.shape[0] > 10_000:
                    print('Warning: corrupted results. Probable cause: too much input noise')
                    mean = 0
                    var = 1
                    res = np.array([[0, 0, 0]])
                    size_pc = np.array([[0, 0, 0]])
            else:
                print('Warning: not enough input points. Skipping reconstruction')
                mean = 0
                var = 1
                res = np.array([[0, 0, 0]])
                size_pc = np.array([[0, 0, 0]])

            # Visualization
            draw_mask(rgb, mask, distance)

            scene_pc = RealSense.rgb_pointcloud(depth, rgb)

            part_pc = PointCloud()
            part_pc.points = Vector3dVector(size_pc + [0, 0, 1])
            part_pc.paint_uniform_color([0, 1, 0])
            pred_pc = PointCloud()
            pred_pc.points = Vector3dVector((res * np.array([1, 1, -1]) * (var * 2) + mean))
            pred_pc.paint_uniform_color([1, 0, 0])

            scene_pcd.clear()
            part_pcd.clear()
            pred_pcd.clear()

            scene_pcd += scene_pc
            part_pcd += part_pc
            pred_pcd += pred_pc

            if not render_setup:
                vis.add_geometry(scene_pcd)
                # vis.add_geometry(part_pcd)
                vis.add_geometry(pred_pcd)

                render_setup = True

            vis.update_geometry(scene_pcd)
            # vis.update_geometry(part_pcd)
            vis.update_geometry(pred_pcd)

            vis.poll_events()
            vis.update_renderer()

    for k in Timer.timers:
        print(f'{k} = {Timer.timers[k] / Timer.counters[k]} s')
        print(f'{k} = {1 / (Timer.timers[k] / Timer.counters[k])} fps')


def draw_pcs(*pcs):
    res = []
    for pc in pcs:
        o3d_pc = PointCloud()
        o3d_pc.points = Vector3dVector(pc)
        o3d_pc.paint_uniform_color(np.random.rand(3))
        res.append(o3d_pc)
    draw_geometries(res)


def draw_mask(rgb, mask, distance):
    overlay = copy.deepcopy(rgb)
    if np.any(mask == 1):
        overlay[mask == 1] = np.array([0, 0, 128])
    res1 = cv2.addWeighted(rgb, 1, overlay, 0.5, 0)

    res2 = copy.deepcopy(rgb)
    res2[mask == 0] = np.array([0, 0, 0])

    # cv2.putText(res2, f'Distance: {distance}', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    font = cv2.FONT_ITALIC
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(res2, f'Distance: {distance / 1000}',
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    cv2.imshow('Segmentation 1', cv2.cvtColor(res1, cv2.COLOR_RGB2BGR))
    cv2.imshow('Segmentation 2', cv2.cvtColor(res2, cv2.COLOR_RGB2BGR))

    cv2.waitKey(1)


import os, sys


class shush:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == '__main__':
    main()
