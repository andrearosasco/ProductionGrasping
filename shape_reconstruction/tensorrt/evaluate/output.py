from multiprocessing.connection import Client, Listener

import numpy as np
import torch
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.visualization import draw_geometries
import tqdm

from shape_reconstruction.pytorch.configs import ModelConfig
from shape_reconstruction.pytorch.model.PCRNetwork import PCRNetwork
from shape_reconstruction.tensorrt.utils.inference import Infer
from shape_reconstruction.tensorrt.utils.dataset import DataSet
from shape_reconstruction.tensorrt.utils.decoder import Decoder


def write(runtime):
    it = 22
    decoder = Decoder()

    # runtime = 'trt'
    # runtime = 'pt'

    data_loader = DataSet(iterations=it)

    if runtime == 'trt':
        ####  Setup TensorRT Engine
        backbone = Infer('./shape_reconstruction/tensorrt/assets/pcr_poli.engine')
        ####  Run Evaluation
        for i, x in tqdm.tqdm(enumerate(data_loader)):
            fast_weights = backbone(x['input'])

            res = decoder(fast_weights)
            np.save(f'./shape_reconstruction/tensorrt/assets/reconstructions/rec_trt{i}', res)

    if runtime == 'pt':
        #### Setup PyTorch Model

        pt_model = PCRNetwork.load_from_checkpoint('./shape_reconstruction/pytorch/checkpoint/final',
                                                   config=ModelConfig)

        pt_model = pt_model.to('cuda')
        pt_model.eval()

        ####  Run Evaluation
        for i, x in tqdm.tqdm(enumerate(data_loader)):
            _, fast_weights = pt_model(torch.tensor(x['input'].astype(np.float32), device='cuda'))
            res = decoder(fast_weights)

            np.save(f'./shape_reconstruction/tensorrt/assets/reconstructions/rec_pt{i}', res)

def read():
    def to_o3d(pc):
        res = PointCloud()
        res.points = Vector3dVector(pc)
        return res

    for i in range(22):
        rec_trt = np.load(f'./shape_reconstruction/tensorrt/assets/reconstructions/rec_trt{i}.npy')
        rec_pt = np.load(f'./shape_reconstruction/tensorrt/assets/reconstructions/rec_pt{i}.npy')
        part = np.load(f'./shape_reconstruction/tensorrt/assets/real_data/partial{i}.npy')

        pcs = [to_o3d(pc).paint_uniform_color(c) for pc, c in zip([rec_trt, rec_pt, part],
                                                                 [[1, 0, 0], [0, 1, 0], [0, 0, 1]])]

        draw_geometries(pcs)

if __name__ == '__main__':
    # write('trt')
    # write('ptr')
    read()
    # pred_pc = PointCloud()
    # pred_pc.points = Vector3dVector(res)
    # pred_pc.paint_uniform_color([0, 0, 1])
    #
    # part_pc = PointCloud()
    # part_pc.points = Vector3dVector(x['input'][0].astype(np.float32))
    # part_pc.paint_uniform_color([1, 0, 0])
    #
    # draw_geometries([pred_pc, part_pc])
