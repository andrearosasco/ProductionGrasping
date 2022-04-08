import copy

import numpy as np
import torch
from torch import nn

from ransac.utils.inference import Runner


# This class starts the trt engine and run it 6 times to extract
# all of the cube faces
class TrTRansac:

    runner = None

    def __init__(self):
        if TrTRansac.runner is None:
            TrTRansac.runner = Runner('./ransac/assets/ransac_5000.engine')

    def __call__(self, points, eps, iterations):
        res_planes = np.zeros([6, 4])
        res_points = []
        i = 0

        # We copy points because we don't wanna modify the input.
        # aux_points:  has one less surface each iteration and it's used to generate the subsets on which we fit a plane
        # inp_points: aux_point + a number of random points such that the input to the engine has always the same size
        aux_points = copy.deepcopy(points)

        while len(res_points) != 6:
            if aux_points.shape[0] < 10:
                return None
            idx = np.random.randint(0, aux_points.shape[0], size=[iterations * 3])
            subsets = aux_points[idx].reshape(iterations, 3, 3)

            diff = 5981 - aux_points.shape[0]
            if diff < 0:
                inp_points = copy.deepcopy(aux_points)
                idx = np.random.choice(inp_points.shape[0], 5981, replace=False)
                inp_points = inp_points[idx]
            elif diff > 0:
                inp_points = np.concatenate([aux_points, np.random.random([diff, 3])])
            else:
                inp_points = copy.deepcopy(aux_points)

            scores, planes = TrTRansac.runner(inp_points, subsets, eps)
            planes = planes.reshape(iterations, 4)

            naninf = np.any(np.isinf(planes) | np.isnan(planes), axis=1) | np.all(planes == 0, axis=1)
            planes[naninf] = np.array([0, 0, 0, 0])
            scores[naninf] = 0

            remove_idxs = np.zeros([aux_points.shape[0]], dtype=np.bool)
            aux_scores = copy.deepcopy(scores)
            while i <= 2:
                if i ==0:
                    pass

                orth = np.all(np.isclose((planes @ res_planes.T), 0, rtol=0, atol=1e-1), axis=1)
                aux_scores[~orth] = 0

                if np.sum(aux_scores) == 0: # TODO remove points anyway?
                    break

                trt_plane = planes[np.argmax(aux_scores)]

                new_idxs = (np.abs(
                    np.concatenate([aux_points, np.ones([aux_points.shape[0], 1])], axis=1) @ trt_plane) < eps)
                remove_idxs += new_idxs

                new_plane = copy.deepcopy(trt_plane)[None]
                new_plane[..., :3] = new_plane[..., :3] / np.linalg.norm(new_plane[..., :3], axis=1, keepdims=True)

                res_planes[i] = new_plane
                i += 1

                new_points = copy.deepcopy(aux_points[new_idxs])
                res_points.append(new_points)

            while 3 <= i < 6:
                aux_scores = copy.deepcopy(scores)

                j = i - 3
                opp = np.any(np.isclose((planes @ res_planes[j:j+1].T), -1, rtol=0, atol=1e-1), axis=1) # between -1 and -.9
                orth = np.any(np.isclose((planes @ res_planes[np.delete(np.arange(6), j)].T), 0, rtol=0, atol=1e-1), axis=1)

                # np.sum(np.any(np.isclose((planes @ res_planes[j:j + 1].T), -1, rtol=0, atol=1e-1), axis=1) & np.any(
                #     np.isclose((planes @ res_planes[np.delete(np.arange(6), j)].T), -1, rtol=0, atol=1e-1), axis=1))

                aux_scores[~(opp & orth)] = 0

                if np.sum(aux_scores) == 0:
                    # trt_plane = planes[np.argmax(scores)]
                    # new_idxs = (np.abs(
                    #     np.concatenate([aux_points, np.ones([aux_points.shape[0], 1])], axis=1) @ trt_plane) < eps)
                    # remove_idxs += new_idxs
                    break

                trt_plane = planes[np.argmax(aux_scores)]

                new_idxs = (np.abs(
                    np.concatenate([aux_points, np.ones([aux_points.shape[0], 1])], axis=1) @ trt_plane) < eps)
                remove_idxs += new_idxs

                new_plane = copy.deepcopy(trt_plane)[None]
                new_plane[..., :3] = new_plane[..., :3] / np.linalg.norm(new_plane[..., :3], axis=1, keepdims=True)

                res_planes[i] = new_plane
                i += 1

                new_points = copy.deepcopy(aux_points[new_idxs])
                res_points.append(new_points)

            aux_points = aux_points[~remove_idxs]

        # Technically they are already normalized but since the engine can approximate values
        # to run faster, we normalize them again.
        return [res_planes, res_points]
