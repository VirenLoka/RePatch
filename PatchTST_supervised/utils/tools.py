import logging
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_path: str) -> logging.Logger:
    """Return a Logger that writes DEBUG+ to log_path and WARNING+ to stderr."""
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter('%(levelname)s  %(message)s'))
    logger.addHandler(sh)

    return logger


# ── Model utilities ───────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_gradient_norm(model) -> float:
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5


# ── Learning-rate schedule ────────────────────────────────────────────────────

def adjust_learning_rate(optimizer, scheduler, epoch, args,
                         printout=True, logger=None):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8,
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3
                     else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10
                     else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15
                     else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25
                     else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5
                     else args.learning_rate * 0.1}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        return

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        if printout:
            _log = logger or logging.getLogger(__name__)
            _log.info(f'LR updated → {lr:.3e}')


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, logger=None):
        self.patience     = patience
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.best_epoch   = 0
        self.early_stop   = False
        self.val_loss_min = np.inf
        self.delta        = delta
        self._log         = logger or logging.getLogger(__name__)

    def __call__(self, val_loss, model, path, epoch=0):
        score = -val_loss
        saved = False
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self._save(val_loss, model, path)
            saved = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self._log.info(
                f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self._save(val_loss, model, path)
            self.counter    = 0
            saved = True
        return saved

    def _save(self, val_loss, model, path):
        if self.verbose:
            self._log.info(
                f'Val loss improved ({self.val_loss_min:.6f} → {val_loss:.6f}). '
                f'Saving checkpoint …')
        torch.save(model.state_dict(), path + '/checkpoint.pth')
        self.val_loss_min = val_loss


# ── Misc ──────────────────────────────────────────────────────────────────────

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def test_params_flop(model, x_shape):
    model_params = sum(p.numel() for p in model.parameters())
    print(f'INFO: Trainable parameter count: {model_params / 1e6:.2f}M')
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        print(f'{"Computational complexity:":<30}  {macs}')
        print(f'{"Number of parameters:":<30}  {params}')
