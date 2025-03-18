import pdb
import torch
import argparse
import numpy as np
from diffusion.diffmodel_init import create_gaussian_diffusion
import os
from utils.config.default import get_cfg_default
from utils.tools import *
from utils.logger import *
from utils.engine.builder import build_dataset, build_model
from trainer import Trainer
import models.PaD_TS
import dataset.stock
import dataset.sine
import dataset.fMRI
import dataset.energy


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    # PAD_TS 模型的默认配置
    cfg.MODEL.PAD_TS = CN()
    cfg.MODEL.PAD_TS.HIDDEN_SIZE = 128
    cfg.MODEL.PAD_TS.NUM_HEADS = 4
    cfg.MODEL.PAD_TS.N_ENCODER = 1
    cfg.MODEL.PAD_TS.N_DECODER = 3
    cfg.MODEL.PAD_TS.FEATURE_LAST = True
    cfg.MODEL.PAD_TS.DROPOUT = 0.0
    cfg.MODEL.PAD_TS.MLP_RATIO = 4.0


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed

    if args.model:
        cfg.MODEL.NAME = args.model

    if args.window:
        cfg.DATASET.WINDOW = args.window


def setup_cfg(args):

    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)

    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)

    dataset = build_dataset(cfg)

    model = build_model(cfg)
    diffusion = create_gaussian_diffusion(
        predict_xstart=cfg.DIFFUSION.PREDICT_XSTART,
        diffusion_steps=cfg.DIFFUSION.DIFFUSION_STEPS,
        noise_schedule=cfg.DIFFUSION.NOISE_SCHEDULE,
        loss=cfg.DIFFUSION.LOSS,
        rescale_timesteps=cfg.DIFFUSION.RESCALE_TIMESTEPS,
    )
    trainer = Trainer(cfg, model, diffusion, dataset)
    print("======Training======")
    trainer.train()
    print("======Done======")

    print("======Evaluating======")
    trainer.eval()
    print("======Done======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    # PaD-TS
    parser.add_argument("--model", type=str, default="", help="name of model")
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--window", type=int, default=24, help="window size")
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument("--period", type=str, default="train", help="train or test")
    args = parser.parse_args()
    main(args)
