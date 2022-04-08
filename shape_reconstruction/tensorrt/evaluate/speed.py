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
from utils.timer import Timer


def main(runtime):
    it = 1000
    # model = PCRNetwork(ModelConfig)
    # model.to('cuda')
    a = torch.zeros([1]).to('cuda')

    # runtime = 'trt'
    # runtime = 'pt'

    data_loader = DataSet(iterations=it)

    if runtime == 'trt':
        ####  Setup TensorRT Engine
        backbone = Infer('./shape_reconstruction/tensorrt/assets/pcr_test.engine')
        ####  Run Evaluation
        for i, x in tqdm.tqdm(enumerate(data_loader)):
            with Timer('backbone'):
                fast_weights = backbone(x['input'])

            # res = decoder(fast_weights)
            # np.save(f'./shape_reconstruction/tensorrt/assets/reconstructions/rec_trt{i}', res)
        print(1 / (Timer.timers['backbone'] / Timer.counters['backbone']))

    if runtime == 'pt':
        #### Setup PyTorch Model

        pt_model = PCRNetwork.load_from_checkpoint('./shape_reconstruction/pytorch/checkpoint/final',
                                                   config=ModelConfig)

        pt_model = pt_model.to('cuda')
        pt_model.eval()

        ####  Run Evaluation
        with torch.no_grad():
            for i, x in tqdm.tqdm(enumerate(data_loader)):
                _, fast_weights = pt_model(torch.tensor(x['input'].astype(np.float32), device='cuda'))
            # res = decoder(fast_weights)

            # np.save(f'./shape_reconstruction/tensorrt/assets/reconstructions/rec_pt{i}', res)

if __name__ == '__main__':
    main('trt')
