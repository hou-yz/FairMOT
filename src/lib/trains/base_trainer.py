from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from lib.models.data_parallel import DataParallel
from lib.utils.utils import AverageMeter


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model = model
        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model = DataParallel(self.model, device_ids=gpus, chunk_sizes=chunk_sizes).to(device)
            self.loss = DataParallel(self.loss, device_ids=gpus, chunk_sizes=chunk_sizes).to(device)
        else:
            self.model = self.model.to(device)
            self.loss = self.loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar(opt.exp_id, max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            if isinstance(batch, list):
                batch1, batch2 = batch
                for k in batch1:
                    if k != 'meta':
                        batch1[k] = batch1[k].to(device=opt.device, non_blocking=True)
                batch = batch1
                for k in batch2:
                    if k != 'meta':
                        batch2[k] = batch2[k].to(device=opt.device, non_blocking=True)

                outputs1 = self.model(batch1['input'])
                outputs2 = self.model(batch2['input'])
                loss, loss_stats = self.loss(outputs1, outputs2, batch1, batch2)
                loss = loss.mean()
                output = outputs1[-1]
            else:
                for k in batch:
                    if k != 'meta':
                        batch[k] = batch[k].to(device=opt.device, non_blocking=True)

                outputs = self.model(batch['input'])
                loss, loss_stats = self.loss(outputs, batch)
                loss = loss.mean()
                output = outputs[-1]
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print('{}| {}'.format(opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
