ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import tyro
from src.args import Args
from src.training.trainer import Trainer

if __name__ == "__main__":
    args = tyro.cli(Args)
    trainer = Trainer(args)
    trainer.train()
