import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import configparser

from rich import print

from dataset_train import TrainDataset 
from dataset_validation import ValidationDataset 
from base_trainer import Trainer 
from model import MODEL

import gc 

gc.collect() 


from utils import numParams

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

def main(config, model, device):
    tr_dataset = TrainDataset(config)
    vl_dataset = ValidationDataset(config)
    tr_loader = DataLoader(tr_dataset,
                             batch_size=int(config['train']['batch_siz2']),
                             num_workers=4,
                             )
    vl_loader = DataLoader(vl_dataset,
                             batch_size=1,
                             num_workers=4,
                             )
    data = {'tr_loader': tr_loader, 'vl_loader': vl_loader}
    print("processing....1")
    model.cuda()
    print('The number of trainable parameters of the net is:%d' % (numParams(model)))

    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = float(config['optimizer']['lr']),
                                 weight_decay = float(config['optimizer']['weight_decay'])) 
    trainer = Trainer(config, data, device, model, optimizer)
    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="STODEC")
    parser.add_argument(
        "--configuration", default = '/data/hyunjoo/asd/project/config.ini', type=str, help="Configuration."
    )

    args = parser.parse_args()
    config = configparser.ConfigParser()    

    config_path = Path(args.configuration).absolute()
    config.read(config_path, encoding='utf-8') 
    
    device = torch.device("cuda")
    

    model = MODEL()
    main(config, model, device)