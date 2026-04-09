ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import tyro
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.args import Args
from src.training.trainer import Trainer

if __name__ == "__main__":
    args = tyro.cli(Args)
    trainer = Trainer(args)
    trainer.train()
