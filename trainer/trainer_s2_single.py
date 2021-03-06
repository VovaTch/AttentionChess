import copy
import gc

import numpy as np
import torch
from torchvision.utils import make_grid
from colorama import Fore

from base import BaseTrainer
from utils.util import inf_loop, MetricTracker
from data_loaders.mcts import MCTS


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, move_limit=125, clip_grad_norm=1e5, 
                 white_engine=None, black_engine=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(1)
        self.move_limit = move_limit
        self.clip_grad_norm = clip_grad_norm

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
        # Set up engines for black and white, checking if the play actually improves over time
        self.white_engine = white_engine
        self.black_engine = black_engine

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.model.aux_outputs_flag = True
        self.train_metrics.reset()
        self.logger.info(Fore.YELLOW + '\n-------------------------<<TRAINING>>-----------------------\n'
                         + Fore.RESET)

       # Superconvergence scheduler, must override regular ones
        if type(self.lr_scheduler) is torch.optim.lr_scheduler.OneCycleLR:
            lr_sceduler_copy = copy.deepcopy(self.lr_scheduler)

        for batch_idx, (board, quality, value, move_idx) in enumerate(self.data_loader):

            if self.device == 'cuda':
                torch.cuda.empty_cache()

            quality = quality.to(self.device)
            value = value.to(self.device)
            move_idx = move_idx.to(self.device)
            self.optimizer.zero_grad()

            _, output_quality, output_value, output_aux = self.model(board)
            loss_dict = self.criterion(output_quality, output_value, quality, value, move_idx, output_aux=output_aux)
            loss = sum([loss_dict[loss_type] * self.config['loss_weights'][loss_type]
                        for loss_type in self.config['loss_weights']])
            loss += sum([loss_dict[loss_type] * self.config['loss_weights'][loss_type[:-2]]
                        for loss_type in loss_dict.keys() if loss_type[:-2] in self.config['loss_weights']])
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output_quality, output_value, quality, 
                                                            value, move_idx, self.criterion))

            if (batch_idx + 1) % 10 == 0:
                self.logger.debug(Fore.GREEN + f'Train Epoch: {epoch} {self._progress(batch_idx)} '
                                               f'Loss: ' + Fore.CYAN + f'{loss.item():.6f}' + Fore.RESET)
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                
            # Superconvergence scheduler
            if type(self.lr_scheduler) is torch.optim.lr_scheduler.OneCycleLR:
                lr_sceduler_copy.step()

            if batch_idx == self.len_epoch:
                break
            
            # Set models in case of full self play; needed because there are 2 MCTS objects
            self.data_loader.dataset.mcts.good_model = copy.deepcopy(self.model)
            self.data_loader.dataset.mcts.evil_model = copy.deepcopy(self.model)
            if self.config['data_loader'] in ['FullSelfPlayLoader', 'PreEndingChessLoader']:
                self.data_loader.dataset.mcts_game.good_model = copy.deepcopy(self.model)
                self.data_loader.dataset.mcts_game.evil_model = copy.deepcopy(self.model)
                
            # Overwrite player specific engines if specified
            if self.white_engine is not None:
                self.data_loader.dataset.white_engine = self.white_engine
                self.data_loader.dataset.black_engine = copy.deepcopy(self.model)
            if self.black_engine is not None:
                self.data_loader.dataset.white_engine = copy.deepcopy(self.model)
                self.data_loader.dataset.black_engine = self.black_engine
            
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None and type(self.lr_scheduler) is not torch.optim.lr_scheduler.OneCycleLR:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.model.aux_outputs_flag = False
        self.valid_metrics.reset()
        with torch.no_grad():

            self.logger.info(Fore.YELLOW + '\n-------------------------<<EVALUATION>>-----------------------\n'
                             + Fore.RESET)

            for batch_idx, (board, quality, value, move_idx) in enumerate(self.valid_data_loader):

                quality = quality.to(self.device)
                value = value.to(self.device)
                move_idx = move_idx.to(self.device)

                _, output_quality, output_value = self.model(board)
                loss_dict = self.criterion(output_quality, output_value, quality, value, move_idx)
                loss = sum([loss_dict[loss_type] * self.config['loss_weights'][loss_type]
                            for loss_type in self.config['loss_weights']])

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output_quality, output_value, quality, value, 
                                                                move_idx,
                                                                self.criterion))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
