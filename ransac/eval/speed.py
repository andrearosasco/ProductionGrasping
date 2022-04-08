import copy
import time

import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector


from ransac.trt_ransac import TrTRansac, BuildRansac
from utils.timer import Timer


def main():
    # The sequential algorithm takes 9 fps while the parallel one,
    # with 8 threads, runs at 30 fps.
    points = np.load('./ransac/assets/test_pcd.npy')
    ransac_points = copy.deepcopy(points)

    trt_ransac = TrTRansac()
    pt_ransac = BuildRansac()

    res = []

    for _ in range(1000):

        with Timer('ransac'):

            planes = trt_ransac(points, 0.01, 1000)

    print(1 / (Timer.timers['ransac'] / Timer.counters['ransac']))



if __name__ == '__main__':
    main()