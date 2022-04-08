from shape_reconstruction.tensorrt.utils.decoder import Decoder
decoder = Decoder()

from shape_reconstruction.tensorrt.utils.inference import Infer as InferPcr
backbone = InferPcr('./shape_reconstruction/tensorrt/assets/pcr.engine')




import numpy as np
import tqdm

# import cv2
#
#
# from utils.input import RealSense


def main():
    # camera = RealSense()


    for _ in tqdm.tqdm(range(1000)):
        # while True:
        # rgb, depth = camera.read()

        # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # mask = np.ones([192, 256, 1], dtype=np.int32)
        # mask = cv2.resize(mask, dsize=(640, 480), interpolation=cv2.INTER_NEAREST)
        #
        # segmented_depth = copy.deepcopy(depth)
        # segmented_depth[mask != 1] = 0
        #
        # segmented_pc = RealSense.depth_pointcloud(segmented_depth)

        segmented_pc = np.load('./shape_reconstruction/tensorrt/assets/real_data/partial0.npy')

        # Downsample
        # idx = np.random.choice(segmented_pc.shape[0], (int(segmented_pc.shape[0] * 0.05)), replace=False)
        # downsampled_pc = segmented_pc[idx]

        downsampled_pc = segmented_pc
        denoised_pc = downsampled_pc
        # Adjust size

        idx = np.random.choice(denoised_pc.shape[0], (2024), replace=False)
        size_pc = denoised_pc[idx]

        # Reconstruction
        fast_weights = backbone(size_pc)


if __name__ == '__main__':
    main()
