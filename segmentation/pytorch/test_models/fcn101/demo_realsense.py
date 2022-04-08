import copy
import time

import cv2
import scipy
from PIL import Image
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import Visualizer
from torch import nn
from torchvision import models

import open3d as o3d

import PIL
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.transforms import InterpolationMode

from utils.misc.bilateral_solver import apply_bilateral
from utils.misc.input import RealSense
from utils.model.wrappers import Segmentator


def main():
    model = models.segmentation.fcn_resnet101(pretrained=False)
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load('./checkpoints/sym4/latest'), strict=False)
    model.eval()

    model = Segmentator(model, device='cuda')

    camera = RealSense()
    # i = camera.intrinsics()

    visualizer = Visualizer()
    visualizer.create_window(width=1920, height=1080)

    o3d_partial = PointCloud()

    setup = False

    i = 15
    while (True):
        frame, depth = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Solo quando leggiamo con cv2
        #
        # frame = np.array(Image.open('test3_rgb.png'))
        # depth = np.array(Image.open('test3_depth.png')).astype(np.uint16)

        x = model.preprocess(frame).to(device='cuda')
        with torch.no_grad():
            y = model.model(x)['out']
        score, categories = model.postprocess(y)


        # dr = T.Compose([
        #     lambda x: torch.tensor(x).unsqueeze(0),
        #     T.Resize((256, 192), InterpolationMode.BILINEAR)])
        #
        # depth = dr(depth.astype(np.int32)).squeeze().numpy().astype(np.uint16)
        ####### Bilateral Solver START ######
        # trf = T.Compose([T.Resize((256, 192), InterpolationMode.BILINEAR),
        #                  # T.ToTensor()
        #                  ])

        # cv2.imshow('r', scipy.ndimage.sobel((0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]) / 255)[..., None].repeat(3, 2))

        # frame = PIL.Image.fromarray(frame)

        # score = torch.softmax(y, dim=1).squeeze()[1].cpu().detach().numpy()


        # categories = np.zeros_like(score)
        # categories[score > 0.5] = 1
        #
        # score[...] = 1

        # aux = apply_bilateral(torch.tensor(np.array(trf(frame))).cpu().numpy(),
        #                              categories,
        #                              score, thresh=0.8)
        # if not aux is None:
        #     categories = aux

        # categories = np.array(T.Resize((480, 640), interpolation=InterpolationMode.NEAREST)(
        #     T.ToPILImage()(categories.astype(np.float32)))).astype(int)

        ####### Bilateral Solver END ######
        # reference, target, confidence, threshold

        cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))
        depth[categories != 1] = 0.0
        cv2.imshow('segmented depth', cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))


        pc = RealSense.pointcloud(depth)

        o3d_aux = PointCloud()
        o3d_aux.points = Vector3dVector(pc)

        # o3d_aux = denoise(o3d_aux)

        o3d_partial.clear()
        o3d_partial += o3d_aux


        # overlay = depth np.array(frame), 1, overlay, 0.5, 0)

        # np.array(T.Resize((480, 640), interpolation=InterpolationMode.NEAREST)(
        #     T.ToPILImage()((score * 255).astype(np.uint8))))[..., np.newaxis].repeat(3, axis=2)

        # x = x.squeeze().permute(1, 2, 0).cpu().numpy()
        overlay = copy.deepcopy(frame)
        # categories = torch.argmax(y.squeeze(), dim=0).detach().cpu().numpy()
        if np.any(categories == 1):
            overlay[categories == 1] = np.array([0, 0, 128])
        res = cv2.addWeighted(frame, 1, overlay, 0.5, 0)

        # cv2.imshow('frame', cv2.cvtColor(res1, cv2.COLOR_RGB2BGR))
        cv2.imshow('resized', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

        if not setup:
            visualizer.add_geometry(o3d_partial)
            setup = True
        #
        visualizer.update_geometry(o3d_partial)
        visualizer.poll_events()
        visualizer.update_renderer()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
#

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def denoise(pcd: o3d.geometry.PointCloud):

    o3d.visualization.draw_geometries([pcd])

    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    o3d.visualization.draw_geometries([voxel_down_pcd])

    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                        std_ratio=2.0)
    display_inlier_outlier(voxel_down_pcd, ind)

    inlier_cloud = voxel_down_pcd.select_by_index(ind)

    cl, ind = inlier_cloud.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
    display_inlier_outlier(inlier_cloud, ind)

    # print("Radius oulier removal")
    # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    # display_inlier_outlier(voxel_down_pcd, ind)


if __name__ == '__main__':
    main()