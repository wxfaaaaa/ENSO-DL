from config_multi_gpus import mypara
import matplotlib.pylab as plt
import os
import time
from networks.netFactory import NetFactory
import torch
from torch.utils.data import DataLoader
import numpy as np
from dataprocess.LoadData import make_dataset_3
import warnings
from copy import deepcopy
from tools.data_tools import get_std
from tools.draw_functions import Plot
from tools.my_tools import copy_code
import copy
warnings.filterwarnings("ignore")

plt.rcParams['text.usetex'] = False

def func_pre(mypara, adr_model, ori_data_path, in_data_path, pattern_name, save_path=None, sub_pattern_name=''):
    copy_code(save_path)
    n_lev = mypara.sst_level + mypara.lev_range[1] - mypara.lev_range[0]
    lead_max = mypara.output_length
    dataCS = make_dataset_3(mypara, pattern_name, data_path=in_data_path, flag='test')
    ori_data = np.load(ori_data_path)[:, int(2-mypara.sst_level):,...].squeeze()
    ori_data_copy = deepcopy(ori_data)
    std_data, stds, _ = get_std(ori_data_copy, mypara)

    ori_data = np.nan_to_num(ori_data)
    if pattern_name == 'CMIP6':
        ori_data = ori_data[1380:1980]
        print(ori_data.shape)
    print("ori_data.shape:", ori_data.shape)
    print("stds:", stds)
    test_group = len(dataCS)

    dataloader_test = DataLoader(dataCS, batch_size=mypara.batch_size_eval, shuffle=False)
    net_fac = NetFactory()
    mymodel = torch.nn.DataParallel(net_fac.net_factory(mypara).to(mypara.device), device_ids=mypara.device_ids) \
            if torch.cuda.is_available() else net_fac.net_factory(mypara).to(mypara.device)
    mymodel.module.load_state_dict(torch.load(adr_model))
    mymodel.eval()

    var_pred = np.zeros([test_group, lead_max, n_lev,
                         mypara.lat_range[1] - mypara.lat_range[0],
                         mypara.lon_range[1] - mypara.lon_range[0], ])

    ii = 0
    iii = 0
    with torch.no_grad():
        for input_var, input_true in dataloader_test:
            out_var = mymodel(input_var.float().to(mypara.device), predictand=None, train=False)
            ii += out_var.shape[0]
            print(out_var.shape, sub_pattern_name, ii, iii)
            out_var = mymodel(input_var.float().to(mypara.device), predictand=None, train=False)
            if torch.cuda.is_available():
                var_pred[iii:ii] = out_var.cpu().detach().numpy()
            else:
                var_pred[iii:ii] = out_var.detach().numpy()
            iii = ii

    del (out_var,input_var,input_true)
    del mymodel, dataCS, dataloader_test

    # ---------------------------------------------------
    if pattern_name == 'CMIP6':
        len_data = int((2014-1970+1)*12)
        start_data = int((1970 - 1965) * 12)
    else:
        len_data = int((2021-1983+1)*12)
        start_data = int((1983 - 1980) * 12)
    print("len_data:", len_data)

    # Obs fields
    cut_var_true = ori_data[start_data:]  # 504-36 = 468
    var_pred = var_pred * stds[None, None, :, None, None]
    print(len_data, cut_var_true.shape)

    # Pred fields
    cut_var_pred = np.zeros([lead_max, len_data, var_pred.shape[2], var_pred.shape[3], var_pred.shape[4]])
    for i in range(lead_max):
        start = start_data - mypara.input_length - i
        end = start + len_data
        cut_var_pred[i] = (var_pred[start: end, i, ...])  # [len_data, lev, lat, lon]
        assert cut_var_pred.shape[1] == cut_var_true.shape[0]

    sst_pred = cut_var_pred[:, :, mypara.sst_level, ...]
    sst_true = cut_var_true[:, mypara.sst_level, ...]

    outputs = [sst_pred, sst_true]
    save_names = ['sst_pred.npy', 'sst_true.npy']

    for save_name, value in zip(save_names, outputs):
        var_save_path = os.path.join(save_path, sub_pattern_name + save_name)
        print(var_save_path)
        np.save(var_save_path, value)
    del sst_pred, sst_true, var_pred


if __name__ == '__main__':
    # mypara.device_ids = [0,1,2,3,4,5,6,7]
    mypara.device_ids = [0, 1, 2, 3]
    mypara.data_weight = 1.0
    mypara.device = torch.device(f"cuda:{mypara.device_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")
    pre_save_name = mypara.pre_save_name
    lead_max = mypara.output_length
    start_time = time.time()

    mypara.x_label = 'Beta Top Box'
    mypara.y_label = 'Beta Bottom Box'
    mypara.title_label = 'Corr_2'
    mypara.label_fontsize = 20
    mypara.title_fontsize = 24
    mypara.save_name = mypara.title_label

    # paths1 = [
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2432_mse____04171909_NorESM2-LM_2',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2432_mse____04171909_NorESM2-LM_1',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2432_mse____04171909_NorESM2-LM_0',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191150_NorESM2-LM_2',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191150_NorESM2-LM_1',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191150_NorESM2-LM_0',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191157_NorESM2-LM_2',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191157_NorESM2-LM_1',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191157_NorESM2-LM_0',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191151_NorESM2-LM_2',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191151_NorESM2-LM_1',
    #     '0_1380_360__False_0_7_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04191151_NorESM2-LM_0',
    # ]
    # paths2 = [
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_9',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_8',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_7',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_6',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_5',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_4',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_3',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_2',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_1',
    #     '0_1380_360__False_34_41_lr0.0015_3370_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____04181559_IPSL-CM6A-LR_0',
    # ]
    paths1_1 = [
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_9',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_8',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_7',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_6',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_5',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_4',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_3',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_2',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_1',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05111911_IPSL-CM6A-LR_0',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_9',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_8',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_7',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_6',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_5',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_4',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_3',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_2',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_1',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171925_IPSL-CM6A-LR_0',
    ]
    paths1_2 = [
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_9',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_8',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_7',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_6',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_5',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_4',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_3',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_2',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_1',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100659_CanESM5-1_0',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_9',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_8',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_7',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_6',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_5',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_4',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_3',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_2',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_1',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171823_CanESM5-1_0',
    ]
    paths1_3 = [
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_9',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_8',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_7',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_6',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_5',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_4',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_3',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_2',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_1',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100653_NESM3_0',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_9',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_8',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_7',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_6',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_5',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_4',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_3',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_2',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_1',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170017_NESM3_0',
    ]

    paths1_4 = [
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_9',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_8',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_7',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_6',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_5',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_4',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_3',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_2',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_1',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100657_CNRM-ESM2-1_0',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_9',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_8',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_7',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_6',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_5',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_4',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_3',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_2',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_1',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170028_CNRM-ESM2-1_0',
    ]

    paths1_5 = [
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_9',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_8',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_7',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_6',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_5',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_4',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_3',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_2',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_1',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05100649_INM-CM4-8_0',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_9',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_8',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_7',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_6',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_5',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_4',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_3',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_2',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_1',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170031_INM-CM4-8_0',
    ]

    paths1_6 = [
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_9',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_8',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_7',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_6',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_5',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_4',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_3',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_2',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_1',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091101_INM-CM5-0_0',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_9',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_8',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_7',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_6',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_5',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_4',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_3',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_2',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_1',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05170139_INM-CM5-0_0',
    ]

    paths1_7 = [
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_9',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_8',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_7',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_6',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_5',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_4',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_3',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_2',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_1',
        # '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05091053_MCM-UA-1-0_0',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_9',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_8',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_7',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_6',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_5',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_4',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_3',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_2',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_1',
        '0_1380_360__False_34_41_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05171757_MCM-UA-1-0_0',
    ]

    paths0_1 = [
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_9',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_8',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_7',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_6',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_5',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_4',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_3',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_2',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_1',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081549_NorESM2-LM_0',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_9',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_8',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_7',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_6',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_5',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_4',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_3',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_2',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_1',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151241_NorESM2-LM_0',
    ]

    paths0_2 = [
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_9',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_8',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_7',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_6',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_5',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_4',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_3',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_2',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_1',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081556_FGOALS-f3-L_0',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_9',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_8',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_7',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_6',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_5',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_4',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_3',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_2',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_1',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151250_FGOALS-f3-L_0',
    ]

    paths0_3 = [
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_9',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_8',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_7',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_6',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_5',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_4',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_3',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_2',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_1',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081551_CAMS-CSM1-0_0',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_9',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_8',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_7',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_6',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_5',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_4',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_3',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_2',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_1',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151246_CAMS-CSM1-0_0',
    ]
    paths0_4 =[
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_9',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_8',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_7',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_6',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_5',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_4',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_3',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_2',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_1',
       # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05070034_MRI-ESM2-0_0',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_9',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_8',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_7',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_6',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_5',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_4',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_3',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_2',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_1',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05151253_MRI-ESM2-0_0',
    ]

    paths0_5 = [
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_9',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_8',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_7',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_6',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_5',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_4',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_3',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_2',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_1',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05081557_KIOST-ESM_0',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_9',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_8',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_7',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_6',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_5',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_4',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_3',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_2',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_1',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160626_KIOST-ESM_0',
    ]

    paths0_6 = [
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_9',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_8',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_7',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_6',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_5',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_4',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_3',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_2',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_1',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090901_CESM2-WACCM-FV2_0',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_9',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_8',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_7',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_6',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_5',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_4',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_3',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_2',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_1',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160654_CESM2-WACCM-FV2_0',
    ]

    paths0_7 = [
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_9',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_8',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_7',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_6',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_5',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_4',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_3',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_2',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_1',
        # '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05090958_GISS-E2-1-G_0',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_9',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_8',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_7',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_6',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_5',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_4',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_3',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_2',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_1',
        '0_1380_360__False_0_7_lr0.0012_3000_1e-05_dp0.15_adamw_44_in12_out20_Geoformer_tauxy_2416_mse____05160712_GISS-E2-1-G_0',
    ]
    pattern = 'GODAS'
    mypara.test_pattern = pattern
    total_paths = [paths0_1, paths0_2, paths0_3, paths0_4, paths0_5, paths0_6, paths0_7, paths1_1, paths1_2, paths1_3, paths1_4, paths1_5, paths1_6, paths1_7]
    # total_paths = [paths1_5, paths1_6, paths1_7, paths0_1, paths0_2, ]

    temp_names = mypara.data_names
    temp_betas = mypara.beta

    copy_names = copy.deepcopy(temp_names)
    combined = list(zip(temp_betas, copy_names))
    #
    # # 按数据值从小到大排序
    sorted_combined = sorted(combined, key=lambda x: x[0])
    #
    # # 提取排序后的 name
    sorted_names = [name for _, name in sorted_combined]
    sorted_alphas = np.array([alpha for alpha, _ in sorted_combined])

    print(sorted_names)

    mypara.data_names = sorted_names[0:7] + sorted_names[-7:]
    # mypara.data_names = sorted_names[-2:-1]

    if pattern == 'CMIP6':
        pattern_names = mypara.data_names[:]
    else:
        pattern_names = ['']

    mypara.climate_name = 'IOD'
    myplot = Plot(mypara)


    sst_preds, sst_trues = [], []
    pattern_sst_preds, pattern_sst_trues, alphas = [], [], []
    # mean_values = [[] for _ in range(len(total_paths))]  # 正确初始化两个独立列表
    # mean_preds = [[] for _ in range(len(total_paths))]
    # mean_trues = [[] for _ in range(len(total_paths))]
    mean_values = []
    mean_preds = []
    mean_trues = []

    print(mean_values)
    for sub_pattern_name in pattern_names:
        if sub_pattern_name == '' and pattern == 'GODAS':
            sub_pattern_name = pattern
        pattern_values = []
        pattern_preds = []
        pattern_trues = []
        for idx, paths in enumerate(total_paths):
            values, preds, trues = [], [], []
            for sub_path in paths:
                if pattern == 'CMIP6':
                    mypara.data_len['CMIP6'] = [1380, 1980]
                    in_data_path = f'{mypara.data_path}/tauuv_thetao_{pattern}_{sub_pattern_name}_185001_201412_{mypara.mask_name}std_deg2_anomaly.npy'
                    ori_data_path = f'{mypara.data_path}/tauuv_thetao_{pattern}_{sub_pattern_name}_185001_201412_360_ori_std_deg2_anomaly.npy'
                else:
                    in_data_path = f'{mypara.data_path}/tauuv_thetao_GODAS__198001_202112_{mypara.mask_name}std_deg2_anomaly.npy'
                    ori_data_path = f'{mypara.data_path}/tauuv_thetao_GODAS__198001_202112_360_ori_std_deg2_anomaly.npy'

                save_path = os.path.join(mypara.test_results, f'{sub_path}')

                path = mypara.model_path + '/' + sub_path
                chk_path = os.path.join(path, 'Geoformer.pth')

                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                func_pre(mypara=mypara,
                         adr_model=chk_path,
                         ori_data_path=ori_data_path,
                         in_data_path=in_data_path,
                         pattern_name=pattern,
                         save_path=save_path,
                         sub_pattern_name=sub_pattern_name)

                sst_pred = np.load(f'{save_path}/{sub_pattern_name}sst_pred.npy')
                sst_true = np.load(f'{save_path}/{sub_pattern_name}sst_true.npy')


                value, pred, true = myplot.calculate_values(sst_pred, sst_true)

                value, pred, true = value[:, None, :, ...], pred[:, None, :, ...], true[:, None, :, ...]

                np.save(f'{save_path}/{sub_pattern_name}_{mypara.climate_name}_value.npy', value)
                np.save(f'{save_path}/{sub_pattern_name}_{mypara.climate_name}_pred.npy', pred)
                np.save(f'{save_path}/{sub_pattern_name}_{mypara.climate_name}_true.npy', true)

                value = np.load(f'{save_path}/{sub_pattern_name}_{mypara.climate_name}_value.npy') # (3, 3, 20) ----> (10, 3, 3, 20) ----> (14,20,3,3,20)
                pred = np.load(f'{save_path}/{sub_pattern_name}_{mypara.climate_name}_pred.npy')
                true = np.load(f'{save_path}/{sub_pattern_name}_{mypara.climate_name}_true.npy')
                myplot.draw_nino34(value, pred, true, sub_pattern_name, save_path)


    end_time = time.time()
    print(end_time - start_time)
