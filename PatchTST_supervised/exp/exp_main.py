from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import (EarlyStopping, adjust_learning_rate, visual,
                          test_params_flop, count_parameters, compute_gradient_norm)
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import sys
import time

import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Summary-block formatting helpers ─────────────────────────────────────────
_W = 68  # block width

def _hline(char='─'):
    return char * _W

def _row(label, value=''):
    """Left-aligned label, value aligned after a colon."""
    return f'  {label:<18}: {value}'
# ─────────────────────────────────────────────────────────────────────────────


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    # ── Validation / test pass ────────────────────────────────────────────────
    def vali(self, vali_data, vali_loader, criterion, desc='  Validating'):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    tqdm(vali_loader, desc=desc, leave=False,
                         unit='batch', file=sys.stderr, dynamic_ncols=True)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                    dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
                total_loss.append(loss)

        self.model.train()
        return float(np.average(total_loss))

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data,  vali_loader  = self._get_data(flag='val')
        test_data,  test_loader  = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)

        # ── Training configuration header ─────────────────────────────────────
        total_params, trainable_params = count_parameters(self.model)
        print(_hline('='))
        print(f'  TRAINING  |  {setting}')
        print(_hline('='))
        print(_row('Model',        self.args.model))
        print(_row('Parameters',
                   f'{trainable_params:,} trainable  /  {total_params:,} total'))
        print(_row('Device',       str(self.device)))
        print(_hline())
        print(_row('Data file',    self.args.data_path))
        print(_row('Train',
                   f'{len(train_data):,} samples  '
                   f'({train_steps} batches × bs={self.args.batch_size})'))
        print(_row('Val',          f'{len(vali_data):,} samples'))
        print(_row('Test',         f'{len(test_data):,} samples'))
        print(_row('seq / pred',   f'{self.args.seq_len} / {self.args.pred_len}'))
        print(_hline())
        print(_row('Epochs',       self.args.train_epochs))
        print(_row('Patience',     self.args.patience))
        print(_row('Batch size',   self.args.batch_size))
        print(_row('Learning rate', self.args.learning_rate))
        print(_row('LR schedule',  self.args.lradj))
        print(_row('Grad clip',    '1.0  (global L2 norm)'))
        print(_hline('='))
        print()
        # ─────────────────────────────────────────────────────────────────────

        time_now    = time.time()
        train_start = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim    = self._select_optimizer()
        criterion      = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer       = model_optim,
            steps_per_epoch = train_steps,
            pct_start       = self.args.pct_start,
            epochs          = self.args.train_epochs,
            max_lr          = self.args.learning_rate,
        )

        loss_history = []   # (train_loss, vali_loss) accumulated per epoch

        epoch_bar = tqdm(range(self.args.train_epochs), desc='Training',
                         unit='epoch', file=sys.stderr, dynamic_ncols=True)
        for epoch in epoch_bar:
            iter_count  = 0
            train_loss  = []
            grad_norms  = []
            clip_count  = 0

            self.model.train()
            epoch_time = time.time()

            batch_bar = tqdm(train_loader,
                             desc=f'  Epoch {epoch + 1}/{self.args.train_epochs}',
                             leave=False, unit='batch',
                             file=sys.stderr, dynamic_ncols=True)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(batch_bar):
                iter_count += 1
                model_optim.zero_grad()

                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                    dim=1).float().to(self.device)

                # ── forward pass ──────────────────────────────────────────────
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim   = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss    = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    f_dim   = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss    = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # per-100 iteration log (stdout → log file)
                if (i + 1) % 100 == 0:
                    print(f'\titers: {i + 1}, epoch: {epoch + 1}'
                          f' | loss: {loss.item():.7f}')
                    speed     = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now   = time.time()

                # ── backward + gradient check + clip ──────────────────────────
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)        # required before norm computation
                    raw_norm = compute_gradient_norm(self.model)
                    clipped  = float(torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1.0))
                    if clipped > 1.0:
                        clip_count += 1
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    raw_norm = compute_gradient_norm(self.model)
                    clipped  = float(torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1.0))
                    if clipped > 1.0:
                        clip_count += 1
                    model_optim.step()

                grad_norms.append(raw_norm)
                batch_bar.set_postfix(loss=f'{loss.item():.4f}',
                                      gnorm=f'{raw_norm:.3f}')

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1,
                                         self.args, printout=False)
                    scheduler.step()

            # ── end-of-epoch statistics ───────────────────────────────────────
            epoch_dur  = time.time() - epoch_time
            train_loss = float(np.average(train_loss))
            vali_loss  = self.vali(vali_data, vali_loader, criterion,
                                   desc='  Validating')
            test_loss  = self.vali(test_data,  test_loader,  criterion,
                                   desc='  Test eval ')

            # gradient statistics
            grad_arr = np.array(grad_norms)
            g_mean   = float(grad_arr.mean())
            g_max    = float(grad_arr.max())
            g_min    = float(grad_arr.min())
            g_std    = float(grad_arr.std())
            if g_mean > 10.0:
                g_status = '⚠  EXPLODING'
            elif g_mean < 1e-6:
                g_status = '⚠  VANISHING'
            else:
                g_status = 'OK'

            # overfit check
            loss_history.append((train_loss, vali_loss))
            overfit_ratio = vali_loss / (train_loss + 1e-10)
            if overfit_ratio >= 2.0:
                overfit_flag = '⚠  SEVERE'
            elif overfit_ratio >= 1.5:
                overfit_flag = '⚠  WARNING'
            else:
                overfit_flag = 'OK'

            # 5-epoch divergence trend: val rising while train falling
            if len(loss_history) >= 5:
                h = loss_history[-5:]
                if h[-1][0] < h[0][0] - 1e-6 and h[-1][1] > h[0][1] + 1e-6:
                    overfit_flag += '  [5-ep diverging trend]'

            current_lr = model_optim.param_groups[0]['lr']
            saved      = early_stopping(vali_loss, self.model, path)
            ckpt_tag   = '✓  saved — new best' if saved else '–  no improvement'

            # ── epoch summary block (stdout → log) ────────────────────────────
            print(_hline())
            print(f'  Epoch {epoch + 1}/{self.args.train_epochs}'
                  f'  |  {epoch_dur:.1f}s  |  LR: {current_lr:.3e}')
            print(_hline())
            print(_row('Loss',
                        f'train={train_loss:.6f}  '
                        f'val={vali_loss:.6f}  '
                        f'test={test_loss:.6f}'))
            print(_row('Overfit',
                        f'val/train={overfit_ratio:.3f}  [{overfit_flag}]'))
            print(_row('Gradient',
                        f'mean={g_mean:.4f}  max={g_max:.4f}  '
                        f'min={g_min:.4f}  std={g_std:.4f}  [{g_status}]'))
            print(_row('Grad clip',
                        f'{clip_count}/{train_steps} batches clipped'))
            print(_row('Checkpoint', ckpt_tag))
            print(_row('Patience',
                        f'{early_stopping.counter}/{self.args.patience}'))
            print(_hline())
            print()
            # ─────────────────────────────────────────────────────────────────

            # epoch bar postfix (stderr → terminal)
            epoch_bar.set_postfix(
                train    = f'{train_loss:.4f}',
                vali     = f'{vali_loss:.4f}',
                test     = f'{test_loss:.4f}',
                patience = f'{early_stopping.counter}/{self.args.patience}',
            )

            if early_stopping.early_stop:
                epoch_bar.write('Early stopping triggered.')
                print('Early stopping triggered.')
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f'LR updated → {scheduler.get_last_lr()[0]:.3e}')

        # ── final training summary (stdout → log) ─────────────────────────────
        total_secs = time.time() - train_start
        mins, secs = divmod(int(total_secs), 60)
        print(_hline('='))
        print('  TRAINING COMPLETE')
        print(_hline('='))
        print(_row('Best epoch',
                   f'{early_stopping.best_epoch}  '
                   f'(val loss = {early_stopping.val_loss_min:.6f})'))
        print(_row('Stopped at',
                   f'epoch {epoch + 1} / {self.args.train_epochs}'))
        print(_row('Total time',   f'{mins}m {secs:02d}s'))
        print(_row('Checkpoint',   path + '/checkpoint.pth'))
        print(_hline('='))
        print()
        # ─────────────────────────────────────────────────────────────────────

        self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
        return self.model

    # ── Test ──────────────────────────────────────────────────────────────────
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('Loading model from checkpoint ...')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds  = []
        trues  = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    tqdm(test_loader, desc='Testing', unit='batch',
                         file=sys.stderr, dynamic_ncols=True)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                    dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    gt  = np.concatenate((inp[0, :, -1],     batch_y[0, :, -1]), axis=0)
                    pd  = np.concatenate((inp[0, :, -1],     outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds  = np.array(preds).reshape(-1, preds[0].shape[-2],  preds[0].shape[-1])
        trues  = np.array(trues).reshape(-1, trues[0].shape[-2],  trues[0].shape[-1])
        inputx = np.array(inputx).reshape(-1, inputx[0].shape[-2], inputx[0].shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        # ── test results summary (stdout → log) ───────────────────────────────
        print(_hline('='))
        print('  TEST RESULTS')
        print(_hline('='))
        print(_row('MSE',  f'{mse:.6f}'))
        print(_row('MAE',  f'{mae:.6f}'))
        print(_row('RMSE', f'{rmse:.6f}'))
        print(_row('MAPE', f'{mape:.6f}'))
        print(_row('MSPE', f'{mspe:.6f}'))
        print(_row('RSE',  f'{rse:.6f}'))
        print(_row('Corr', f'{corr:.6f}'))
        print(_hline('='))
        print()
        # ─────────────────────────────────────────────────────────────────────

        with open('result.txt', 'a') as f:
            f.write(setting + '  \n')
            f.write(f'mse:{mse}, mae:{mae}, rse:{rse}\n\n')

        np.save(folder_path + 'pred.npy',    preds)
        np.save(folder_path + 'metrics.npy',
                np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        return

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len,
                                       batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp],
                                    dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)

                preds.append(outputs.detach().cpu().numpy())

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        return
