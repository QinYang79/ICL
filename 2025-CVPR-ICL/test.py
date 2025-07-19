from prettytable import PrettyTable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    sub = '/home/qinyang/projects/ACVPR/RDE_bak/final_logs/CUHK-PEDES/20241113_011401_RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0+aug+aug_dual_fixed_0.5'
    parser.add_argument("--config_file", default=f'{sub}/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = False
    args.dataset_name = 'CUHK-PEDES'
    args.root_dir='/home/qinyang/projects/data'
    # CUHK-PEDES ICFG-PEDES RSTPReid UFine6926
    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    args.output_dir =sub
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    asss = ['best.pth','last.pth']
    for i in range(len(asss)):
        if os.path.exists(op.join(args.output_dir, asss[i])):
            model = build_model(args,num_classes)
            checkpointer = Checkpointer(model)
            checkpointer.load(f=op.join(args.output_dir, asss[i]))
            model = model.cuda()
            do_inference(model, test_img_loader, test_txt_loader)


