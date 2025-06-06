import torch
import os.path
from networks.netFactory import NetFactory
from config_multi_gpus import mypara
from torch.utils.data import DataLoader
import math
from dataprocess.LoadData import *
import time
import torch.nn as nn
import shutil
from torch.amp import GradScaler, autocast
from tools.nino_tools import *
from tools.my_tools import copy_code
import copy
from test_multi_gpus import func_pre
from tools.draw_functions import Plot
class lrwarm:
    def __init__(self, model_size, factor, warmup, lr_w, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
                self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5)))

class modelTrainer:
    def __init__(self, mypara):
        assert mypara.input_channal == mypara.output_channal
        self.mypara = mypara
        self.device = mypara.device
        self.lr_w = mypara.lr_w
        net_fac = NetFactory()
        self.mymodel = net_fac.net_factory(mypara)
        self.mymodel = torch.nn.DataParallel(self.mymodel.to(self.device), device_ids=mypara.device_ids) \
            if torch.cuda.is_available() else self.mymodel.to(self.device)
        if mypara.opt == 'adam':
            self.opt = torch.optim.Adam(self.mymodel.parameters(), lr=mypara.lr)
        else:
            self.opt = torch.optim.AdamW(self.mymodel.parameters(), lr=mypara.lr, weight_decay=mypara.weight_decay)
        factor = math.sqrt(mypara.d_size * mypara.warmup) * mypara.lr
        self.opt = lrwarm(mypara.d_size, factor, mypara.warmup, mypara.lr_w, optimizer=self.opt)
        self.sst_level = mypara.sst_level
        self.ninoweight = mypara.ninoweight[: self.mypara.output_length]

    def model_pred(self, dataloader):
        self.mymodel.eval()
        nino_preds = []
        nino_trues = []
        with torch.no_grad():
            for input_var, var_true in dataloader:
                with autocast('cuda'):
                    input_var = input_var.to(self.device)
                    var_true = var_true.to(self.device)
                    var_pred = self.mymodel(input_var.float().to(self.device), predictand=None, train=False,)
                    nino_pred = get_nino(mypara, var_pred)
                    nino_true = get_nino(mypara, var_true)
                    nino_trues.append(nino_true)
                    nino_preds.append(nino_pred)
            nino_preds = torch.cat(nino_preds, dim=0).to(self.device)
            nino_trues = torch.cat(nino_trues, dim=0).to(self.device)
            print(nino_preds.shape, nino_trues.shape)
            ninosc = pearson_correlation(mypara, nino_preds, nino_trues.float().to(self.device))
        self.mymodel.train()
        return (None, None, 0, 0, 0, ninosc,)

    def print_info(self, i_epoch=0, j=0, current_lr=0, loss_var=0, score=None):
        print( "-->epoch: {} batch:{}, lr:{}, loss_var:{:.2f}, score:{:.3f}\n".
            format(i_epoch, j, current_lr, loss_var, score))
    def train_model(self, data_train, dataset_eval, train_snapshot_path):
        path = train_snapshot_path
        torch.manual_seed(self.mypara.seeds)
        chk_path = os.path.join(path, 'Geoformer.pth')

        dataloader_train = DataLoader(data_train, batch_size=self.mypara.batch_size_train, shuffle=True, drop_last=True, num_workers=self.mypara.num_workers, pin_memory=True)
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.mypara.batch_size_eval, shuffle=True, drop_last=True, num_workers=self.mypara.num_workers, pin_memory=True)
        print(len(dataloader_eval))
        print(len(dataloader_train))

        count = 0
        best = -math.inf
        sv_ratio = 1
        scaler = GradScaler('cuda')
        flag = 0
        for i_epoch in range(self.mypara.num_epochs):
            print("==========" * 8)
            self.mymodel.train()
            current_lr = 0
            for j, (input_var, var_true) in enumerate(dataloader_train):
                loss_var, score = 0, 0
                with autocast('cuda'):
                    input_var = input_var.float().to(self.device)
                    var_true = var_true.float().to(self.device)
                    if sv_ratio > 0:
                        sv_ratio = max(sv_ratio - 2.5e-4, 0)
                    print(input_var.shape, var_true.shape)
                    # -------training for one batch
                    var_pred = self.mymodel(input_var, var_true, train=True, sv_ratio=sv_ratio)
                    nino_pred = get_nino(mypara, var_pred)
                    nino_true = get_nino(mypara, var_true)

                    self.opt.optimizer.zero_grad()

                    loss_var += nn.MSELoss()(var_pred, var_true)

                    if torch.isnan(loss_var):
                        flag = 1
                        break
                    score = pearson_correlation(mypara, nino_pred, nino_true.float().to(self.device))
                    scaler.scale(loss_var).backward()
                    scaler.unscale_(self.opt.optimizer)
                    self.opt.step()
                    scaler.update()


                if flag != 0:
                    break
                current_lr = self.opt.optimizer.param_groups[0]['lr']
                self.print_info(i_epoch, j, current_lr, loss_var, score)

                # ---------Intensive verification
                if (j + 1) % 200 == 0:
                    _, _, _, _, _, sceval = self.model_pred(dataloader=dataloader_eval)
                    self.print_info(score=sceval)
                    if sceval > best:
                        print("\nsc is increase from {:.3f} to {:.3f}   \nsaving model...\n".format(best, sceval))
                        torch.save(self.mymodel.module.state_dict(), chk_path, )
                        best = sceval
                        count = 0

                    torch.save(self.mymodel.module.state_dict(), os.path.join(path, f'Geoformer_{i_epoch}_{j}_{sceval}.pth'))

            # ----------after one epoch-----------
            if flag != 0:
                break
            _, _, lossvar_eval, lossnino_eval, comloss_eval, sceval = self.model_pred(dataloader=dataloader_eval)
            self.print_info(i_epoch=i_epoch,score=sceval)

            if sceval <= best:
                count += 1
                print("\nsc is not increase for {} epoch".format(count))
            else:
                count = 0
                print("\nsc is increase from {:.3f} to {:.3f}   \nsaving model...\n".format(best, sceval))
                torch.save(self.mymodel.module.state_dict(), chk_path)
                best = sceval
            torch.save(self.mymodel.module.state_dict(), os.path.join(path, f'Geoformer_{i_epoch}_{sceval}.pth'), )
            # ---------early stop
            if (count == self.mypara.patience) or (i_epoch == (self.mypara.num_epochs - 1)):
                print("\n-----!!!early stopping reached, max(sceval)= {:3f}!!!-----".format(best))
                break


if __name__ == "__main__":
    values, preds, trues = [], [], []
    sub_path = ''
    for i in range(mypara.train_num):
        start_time = time.time()
        temp_names = mypara.data_names
        temp_betas = mypara.beta
        copy_names = copy.deepcopy(temp_names)
        combined = list(zip(temp_betas, temp_names))
        sorted_combined = sorted(combined, key=lambda x: x[0])
        sorted_names = [name for _, name in sorted_combined]
        sorted_alphas = np.array([alpha for alpha, _ in sorted_combined])
        print(sorted_names)

        # mypara.data_names = sorted_names[mypara.start_file:mypara.end_file]
        # middle_file = (mypara.start_file + mypara.end_file)//2
        middle_file = mypara.start_file + 0
        # mypara.data_names = sorted_names[mypara.start_file:middle_file] + sorted_names[middle_file+1:mypara.end_file]
        mypara.data_names = sorted_names

        if (mypara.end_file-mypara.start_file) != len(sorted_names):
            sub_path = '/' + mypara.pre_save_name+'_' + sorted_names[middle_file] +'_' + f'{i}'
        else:
            sub_path = '/' + mypara.pre_save_name+'_' + f'{i}'

        train_snapshot_path = mypara.model_path + sub_path

        print(mypara.data_names)
        if mypara.tf_train == False:
            if not os.path.exists(train_snapshot_path):
                os.makedirs(train_snapshot_path)
            if os.path.exists(train_snapshot_path + '/code'):
                shutil.rmtree(train_snapshot_path + '/code')
        else:
            train_snapshot_path = mypara.load_path

        copy_code(train_snapshot_path)

        print("\nloading pre-train dataset...")
        traindataset = make_dataset_3(mypara, 'CMIP6')

        print("\nloading evaluation dataset...")
        evaldataset = make_dataset_3(mypara, 'SODA_ORAS5')

        print("len(traindataset), len(evaldataset): ", len(traindataset), len(evaldataset))
        # -------------------------------------------------------------
        trainer = modelTrainer(mypara)
        trainer.train_model(data_train=traindataset, dataset_eval=evaldataset, train_snapshot_path=train_snapshot_path)
        in_data_path = f'{mypara.data_path}/tauuv_thetao_GODAS__198001_202112_{mypara.mask_name}std_deg2_anomaly.npy'
        ori_data_path = f'{mypara.data_path}/tauuv_thetao_GODAS__198001_202112_360_ori_std_deg2_anomaly.npy'
        save_path = mypara.test_results + '/' + f'{sub_path}'

        chk_path = os.path.join(train_snapshot_path, 'Geoformer.pth')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        mypara.device_ids = [0, 1, 2, 3]
        func_pre(mypara=mypara,
                 adr_model=chk_path,
                 ori_data_path=ori_data_path,
                 in_data_path=in_data_path,
                 pattern_name=mypara.test_pattern,
                 save_path=save_path,
                 sub_pattern_name=mypara.test_pattern)
        mypara.device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        sst_pred = np.load(f'{save_path}/{mypara.test_pattern}sst_pred.npy')
        sst_true = np.load(f'{save_path}/{mypara.test_pattern}sst_true.npy')
        myplot = Plot(mypara)
        value, pred, true = myplot.calculate_values(sst_pred, sst_true)
        print(value.shape, pred.shape, true.shape)
        value, pred, true = value[:, None, :, ...], pred[:, None, :, ...], true[:, None, :, ...]
        np.save(f'{save_path}/{mypara.test_pattern}_value.npy', value)
        np.save(f'{save_path}/{mypara.test_pattern}_pred.npy', pred)
        np.save(f'{save_path}/{mypara.test_pattern}_true.npy', true)
        preds.append(pred)
        trues.append(true)
        values.append(value)

        myplot.draw_nino34(value, pred, true, mypara.test_pattern, save_path)
        mypara.data_names = copy_names
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time} seconds")

    myplot = Plot(mypara)
    values = np.concatenate(values, axis=1)
    preds = np.concatenate(preds, axis=1)
    trues = np.concatenate(trues, axis=1)

    save_path = mypara.test_results + '/' + f'{sub_path}'
    myplot.draw_nino34(values, preds, trues, mypara.test_pattern, save_path)

