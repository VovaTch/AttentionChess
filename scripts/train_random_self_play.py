import argparse
import collections
import copy

import torch
import numpy as np
from clearml import Task
from clearml.backend_api import Session

import data_loaders.dataloader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.attchess as module_arch
from parse_config import ConfigParser
from trainer.trainer_s2_single import Trainer
from utils import prepare_device
from data_loaders.dataloader import collate_fn
from data_loaders.mcts import MCTS


# fix random seeds for reproducibility
SEED = 3407
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, args):
    
    if args.ml:
        """Handle ClearML integration"""
        
        task = Task.init(project_name=config['name'], task_name=config['task_name'])
        Session._session_initial_timeout = (15., 30.)
        
        config_dict = task.current_task().connect(config)
    
    logger = config.get_logger('train')

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, collate_fn=collate_fn)
    valid_data_loader = data_loader.split_validation()

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    torch.autograd.set_detect_anomaly(True)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    
    mcts_engine = MCTS(copy.deepcopy(model), copy.deepcopy(model), 100, device=device)
    data_loader.set_mcts(mcts_engine)

    trainer.train()


if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')  # Necessary for this to work; maybe it will run out of memory like that

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config/config_s3_random.json', type=str,
                      help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    parser.add_argument('--ml', action='store_true',
                      help='Activating ClearML flag.')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(parser, options)
    args = parser.parse_args()
    
    main(config, args)
