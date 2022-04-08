import torch
from torch.utils.data import DataLoader

from configs import DataConfig, TrainConfig
from utils.reproducibility import get_init_fn, get_generator, make_reproducible

from datasets.BoxNetPOVDepth import BoxNet as Dataset

if __name__ == '__main__':
    make_reproducible(TrainConfig.seed)

    training_set = Dataset(DataConfig, DataConfig.train_samples)

    dl = DataLoader(training_set,
                    batch_size=TrainConfig.mb_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=TrainConfig.num_workers,
                    pin_memory=True,
                    worker_init_fn=get_init_fn(TrainConfig.seed),
                    generator=get_generator(TrainConfig.seed))

    for _ in dl:
        pass