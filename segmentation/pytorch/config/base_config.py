from utils.framework.config import BaseConfig


class Config(BaseConfig):
    class General:
        device: str

    class Train:
        lr: float
        momentum: float
        weight_decay: float
        log_every: int
        epoch: int

        update_step: int

    class Eval:
        wandb: bool

    class Data:
        class Eval:
            mb_size: int

        class Train:
            mb_size: int

        num_worker: int

