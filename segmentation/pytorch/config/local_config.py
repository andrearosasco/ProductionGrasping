from utils.framework.config import BaseConfig


class Config(BaseConfig):

    run = 'test_models.fcn101.train'

    class General:
        device = 'cpu'

    class Train:
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.0
        log_every = 10
        epoch = 10

    class Eval:
        wandb = True

    class Data:
        class Eval:
            mb_size = 1
            paths = {'ycb': '../../Downloads/YCB_Video_Dataset'}

        class Train:
            mb_size = 1

        num_worker = 0


if __name__ == '__main__':
    Config()