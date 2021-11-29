import copy

import numpy as np
import torch
from torchvision.utils import make_grid
import colorama

from base import BaseTrainer
from utils.util import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, move_limit=125):
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

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (board, turn, score) in enumerate(self.data_loader):

            board, score = board.to(self.device), score.to(self.device)
            board = board.squeeze(0)  # TODO: Temp solution, if I can train with larger batch sizes it can be good
            score = score.squeeze(0)

            if board.size()[0] >= self.move_limit:
                board = board[-self.move_limit:, :, :]
                score = score[-self.move_limit:, :, :]
                turn = turn[-self.move_limit:]

            self.optimizer.zero_grad()
            output = self.model(board, turn)
            loss = self.criterion(board, turn, predicted_logits=output, played_logits=score)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(board, turn, predicted_logits=output, played_logits=score))

            if batch_idx + 1 % 5:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            # self.writer.add_image('input', make_grid(board.unsqueeze(1).cpu().repeat(1, 3, 1, 1),
            #                                          nrow=8, normalize=True))

            if self.device == 'cuda':
                torch.cuda.empty_cache()

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (board, turn, score) in enumerate(self.data_loader):
                board, score = board.to(self.device), score.to(self.device)
                board = board.squeeze(0)  # TODO: Temp solution, if I can train with larger batch sizes it can be good
                score = score.squeeze(0)

                output = self.model(board, turn)
                loss = self.criterion(board, turn, predicted_logits=output, played_logits=score)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(board, turn, predicted_logits=output,
                                                                played_logits=score))
                # self.writer.add_image('input', make_grid(board.cpu(), nrow=8, normalize=True))

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
