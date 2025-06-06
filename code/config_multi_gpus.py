
import numpy as np
import torch
from datetime import datetime
import math
from copy import deepcopy
import torch.nn.functional as F
class Mypara:
    def __init__(self):
        pass


mypara = Mypara()
mypara.device_ids = [0,1,2,3,4,5,6,7]
mypara.device = torch.device(f"cuda:{mypara.device_ids[0]}") if torch.cuda.is_available() else torch.device("cpu")
print(mypara.device, print(torch.cuda.is_available()))
mypara.batch_size_train = 24
mypara.batch_size_eval = 16
mypara.num_workers = 0
mypara.num_epochs = 20

mypara.lr = 6e-4
mypara.data_len = {'CMIP6': [0, 1380], 'ORAS5': [0,264], 'GODAS': [0,504], 'SODA': [0,1308], 'SODA_ORAS5': [0,1572]}
mypara.train_num = 10

mypara.warmup = 3000
mypara.lr_w = 1.0
mypara.mask_name = '360_'
mypara.opt = 'adamw'
mypara.weight_decay = 1e-5
mypara.data_weight = False
mypara.d_size = 256
mypara.nheads = 4
mypara.dim_feedforward = 512
mypara.dropout = 0.15
mypara.num_encoder_layers = 4
mypara.num_decoder_layers = 4
mypara.start_file = 0
mypara.end_file = 41
mypara.file_num = 41
mypara.tf_train = False

mypara.gap = 46

mypara.pear_alpha = 1
mypara.pear_norm = False
mypara.rmse_norm = False
mypara.mse_norm = False

mypara.mse_loss = True
mypara.mse_name = 'mse' if mypara.mse_loss else ''
mypara.l1_loss = False
mypara.l1_name = 'mae' if mypara.l1_loss else ''
mypara.rmse_loss = False
mypara.rmse_name = 'rmse' if mypara.rmse_loss else ''
mypara.nino_loss = False
mypara.nino_name = 'nino' if mypara.nino_loss else ''

# mypara.model_name = 'Point_Geoformer'
mypara.model_name = 'Geoformer'

mypara.x_label = 'Top Box'
mypara.y_label = 'Bottom Box'
mypara.title_label = 'Test'
mypara.label_fontsize = 20
mypara.title_fontsize = 24
mypara.save_name = 'Test'

mypara.need_attn = False

mypara.TFlr = 1.5e-5
mypara.early_stopping = True
mypara.specific_time = False
mypara.patience = 20

mypara.x_label = 'Top Box'
mypara.y_label = 'Bottom Box'
mypara.title_label = 'Test'
mypara.label_fontsize = 20
mypara.title_fontsize = 24
mypara.font_size = 14
mypara.save_name = 'Test'
mypara.climate_name = 'ENSO'

mypara.needtauxy = True
mypara.sst_level = 2 if mypara.needtauxy else 0
mypara.tauxy_name = 'tauxy' if mypara.needtauxy else ''

mypara.only_sst = False
mypara.input_channal = 7  # n_lev of 3D temperature
mypara.output_channal = 7
mypara.input_length = 12
mypara.output_length = 20
mypara.output_features = 1
mypara.lev_range = (1, 8)

mypara.train_group = 'all'
mypara.tauuv = '' if mypara.needtauxy else 'ntauuv'



# model
mypara.model_path = "../model"
mypara.data_path = '../../data'
mypara.test_results = "../test_results"


mypara.lon_range = (0, 180) if '360' in mypara.mask_name else (45, 165)
mypara.lat_range = (0, 51)
# nino34 region
mypara.lon_nino_relative = (95, 121) if '360' in mypara.mask_name else (49, 75)
mypara.lat_nino_relative = (15, 36)

# patch size
mypara.patch_size = (3, 4)
mypara.H0 = int((mypara.lat_range[1] - mypara.lat_range[0]) / mypara.patch_size[0])
mypara.W0 = int((mypara.lon_range[1] - mypara.lon_range[0]) / mypara.patch_size[1])
mypara.emb_spatial_size = mypara.H0 * mypara.W0
mypara.test_pattern = 'GODAS'

mypara.seeds = 1


# mypara.data_names =[
#         'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5','CanESM5-1', 'CanESM5-CanOE',
#     'CESM2', 'CESM2-WACCM', 'CESM2-WACCM-FV2', 'CIESM', 'CMCC-CM2-SR5', 'CMCC-ESM2','CNRM-CM6-1',
#     'CNRM-ESM2-1', 'E3SM-1-0', 'E3SM-1-1', 'EC-Earth3', 'EC-Earth3-CC','EC-Earth3-Veg', 'EC-Earth3-Veg-LR',
#     'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0', 'GFDL-CM4', 'GFDL-ESM4', 'GISS-E2-1-G','HadGEM3-GC31-LL',
#     'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MCM-UA-1-0', 'MIROC-ES2L', 'MIROC6',
#     'MRI-ESM2-0', 'NESM3', 'SAM0-UNICON', 'NorESM2-LM', 'UKESM1-0-LL',  'HadGEM3-GC31-MM',
#     ]
mypara.data_names = [
            'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5-1', 'CMCC-CM2-SR5', 'E3SM-1-1',
                  'EC-Earth3', 'EC-Earth3-CC', 'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0', 'INM-CM4-8', 'INM-CM5-0',
                  'NorESM2-LM',
        'CESM2-WACCM-FV2', 'CNRM-ESM2-1', 'GISS-E2-1-G', 'CMCC-ESM2', 'EC-Earth3-Veg', 'NESM3',
                  'EC-Earth3-Veg-LR', 'CanESM5', 'CIESM', 'GFDL-CM4', 'GFDL-ESM4',
                  'CNRM-CM6-1', 'MCM-UA-1-0', 'CanESM5-CanOE', 'SAM0-UNICON','E3SM-1-0',
                  'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'IPSL-CM6A-LR', 'KIOST-ESM', 'HadGEM3-GC31-MM', 'MRI-ESM2-0',
        'MIROC6', 'MIROC-ES2L',
        'CESM2', 'CESM2-WACCM']


now = datetime.now()
current_time = now.strftime("%m%d%H%M")

mypara.alpha = [
    0.18753173, -0.18862179,  0.04319507,  0.440747,    0.06792893,  0.67670929,
  0.16590307,  0.14261901, 0.54980814,  0.91446989,  0.09502151,  0.5345214,
 -0.00832802, -0.02802921,  0.48257755,  0.63610458,  0.19483274,  0.53573579,
  0.66918939, -0.02312484,  0.1588742,   0.52778548,  0.01039802,  0.51093074,
  0.20859946,  0.19292356,  0.28959032,  0.30927494, -0.03073625,  0.47398397,
  0.39417119,  0.17987085,  0.3806649,   0.17336711,  0.21038545, -0.03871647,
  0.53088706,  0.5036319,   0.80834796,  0.50476446,  0.36922875]

mypara.beta = [-0.17859618, -0.1471892,  -0.22933177, -0.32658712, -0.11864546, -0.20991332,
 -0.17581666, -0.14363591, -0.22431107, -0.32820681, -0.12358446, -0.16797017,
 -0.04324518, -0.03020757, -0.39491941, -0.27789851, -0.09999392, -0.26463492,
 -0.26156872, -0.21553457, -0.11851587, -0.21434143, -0.1311609,  -0.16386623,
 -0.12378838, -0.17023655, -0.16852477, -0.01441859, -0.12313315, -0.18311958,
 -0.15496726, -0.15832384, -0.25977054, -0.11920888, -0.28899838, -0.19026376,
 -0.30018795, -0.18169053, -0.19247375, -0.2392955,  -0.23303277]

mypara.beta_45 = [-0.33202918, -0.1901706,  -0.31923703, -0.43463726, -0.18314211, -0.50422172,
 -0.26225285, -0.28716426, -0.34295515, -0.45613424, -0.12533566, -0.36690838,
 -0.05333108, -0.06075879, -0.55194425, -0.3806207,  -0.29427783, -0.30215181,
 -0.46253408, -0.32330444, -0.21704104, -0.29910637, -0.33418009, -0.18783987,
 -0.29842103, -0.29528873, -0.34200256, -0.57027166, -0.31610093, -0.04271932,
 -0.22699846, -0.16278652, -0.13376754, -0.43249881, -0.42213333, -0.19549826,
 -0.41428762, -0.33923271, -0.42028077, -0.36142831, -0.41020844, -0.36101485,
 -0.44747047, -0.42545245, -0.43038285]

mypara.alpha_45 = [ 0.18753173, -0.18862179,  0.04319507,  0.440747,   0.06792893,  0.67670929,
  0.16590307,  0.14261901,  0.54980814,  0.91446989,  0.09502151,  0.5345214,
 -0.00832802, -0.02802921,  0.48257755,  0.63610458,  0.19483274,  0.53573579,
  0.66918939, -0.02312484,  0.1588742,   0.52778548,  0.65885826,  0.01039802,
  0.51093074,  0.20859946,  0.19292356,  0.06304098,  0.28959032, 0.30927494,
 -0.08721294,  0.43791281, -0.03073625,  0.47398397,  0.39417119, 0.17987085,
  0.3806649,  0.17336711,  0.21038545, -0.03871647,  0.53088706,  0.5036319,
  0.80834796,  0.50476446,  0.36922875]


mypara.time_map = {
        800:'195801_202408',
        3000:'185001_209912',
        3012:'185001_210012',
        536:'198001_202408'
    }

mypara.data_names_45 = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CanESM5-1', 'CMCC-CM2-SR5', 'E3SM-1-1',
                  'EC-Earth3', 'EC-Earth3-CC', 'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0', 'INM-CM4-8', 'INM-CM5-0',
                  'NorESM2-LM',
        'CESM2-WACCM-FV2', 'CNRM-ESM2-1', 'GISS-E2-1-G', 'CMCC-ESM2', 'EC-Earth3-Veg', 'NESM3',
                  'EC-Earth3-Veg-LR', 'CAS-ESM2-0', 'CanESM5', 'CIESM', 'GFDL-CM4', 'GFDL-ESM4',
        'NorESM2-MM',
                  'CNRM-CM6-1', 'MCM-UA-1-0', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5-CanOE', 'SAM0-UNICON','E3SM-1-0',
                  'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'IPSL-CM6A-LR', 'KIOST-ESM', 'HadGEM3-GC31-MM', 'MRI-ESM2-0',
        'MIROC6', 'MIROC-ES2L',
        'CESM2', 'CESM2-WACCM'
    ]

name = 'CMIP6'
start = mypara.data_len[name][0]
end = mypara.data_len[name][1]
mypara.pre_save_name = (f'{start}_{end}_{mypara.mask_name}_{mypara.data_weight}_{mypara.start_file}'
                        f'_{mypara.end_file}_lr{mypara.lr}_{mypara.warmup}_{mypara.weight_decay}_dp{mypara.dropout}_{mypara.opt}'
                        f'_{mypara.num_encoder_layers}{mypara.num_decoder_layers}'
                        f'_in{mypara.input_length}_out{mypara.output_length}_{mypara.model_name}_{mypara.tauxy_name}'
                        f'_{mypara.batch_size_train}{mypara.batch_size_eval}_{mypara.mse_name}_{mypara.l1_name}_{mypara.rmse_name}_'
                        f'{mypara.nino_name}_{current_time}')

mypara.ninoweight = torch.from_numpy(np.array([1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6) * np.log(np.arange(24) + 1))[:mypara.output_length]

"""
[0.00832802 0.01039802 0.02312484 0.02802921 0.03073625 0.03871647
 0.04319507 0.06792893 0.09502151 0.14261901 0.1588742  0.16590307
 0.17336711 0.17987085 0.18753173 0.18862179 0.19292356 0.19483274
 0.20859946 0.21038545 0.28959032 0.30927494 0.36922875 0.3806649
 0.39417119 0.440747   0.47398397 0.48257755 0.5036319  0.50476446
 0.51093074 0.52778548 0.53088706 0.5345214  0.53573579 0.54980814
 0.63610458 0.66918939 0.67670929 0.80834796 0.91446989]
[39.2214915  31.41341963 14.12495678 11.65346314 10.62710531  8.43665152
  7.5619131   4.80851628  3.43750973  2.29027929  2.05594971  1.96884461
  1.8840792   1.81595498  1.7417712   1.73170536  1.69309215  1.67650142
  1.56585911  1.55256633  1.12792916  1.05613913  0.8846477   0.85807062
  0.8286688   0.74109946  0.68913167  0.67685984  0.64856369  0.64710849
  0.63929872  0.61888282  0.61526714  0.61108379  0.60969861  0.59409336
  0.51349633  0.488109    0.48268491  0.40408015  0.35718767]
['CMCC-ESM2', 'FGOALS-g3', 'EC-Earth3-Veg', 'CNRM-CM6-1', 'INM-CM4-8', 'MRI-ESM2-0', 
'BCC-CSM2-MR', 'CanESM5', 'CIESM', 'CESM2', 'EC-Earth3-Veg-LR', 'CanESM5-CanOE', 
'MIROC-ES2L', 'KIOST-ESM', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'GFDL-ESM4', 'E3SM-1-1', 
'GFDL-CM4', 'MIROC6', 'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'MCM-UA-1-0', 
'IPSL-CM6A-LR', 'CAMS-CSM1-0', 'INM-CM5-0', 'CNRM-ESM2-1', 'SAM0-UNICON', 'UKESM1-0-LL', 
'FIO-ESM-2-0', 'FGOALS-f3-L', 'NESM3', 'CMCC-CM2-SR5', 'EC-Earth3', 'CESM2-WACCM', 
'E3SM-1-0', 'EC-Earth3-CC', 'CanESM5-1', 'NorESM2-LM', 'CESM2-WACCM-FV2']
"""


# std_name = 'std'
# for sub_pattern_name in mypara.data_names:
#     in_data_path = f'{mypara.data_path}/tauuv_thetao_CMIP6_{sub_pattern_name}_185001_201412_{mypara.mask_name}{std_name}_deg2_anomaly.npy'
#     save_data_path = f'{mypara.data_path}/tauuv_thetao_CMIP6_{sub_pattern_name}_185001_201412_{mypara.mask_name}20_360_{std_name}_deg2_anomaly.npy'
#     datain = np.load(in_data_path)
#     dataout = datain[:,:,70:121,...]
#     np.save(save_data_path, dataout)
#     print(datain.shape, dataout.shape)
#
# data_files = []
# save_files = []
#
# in_data_path = f'{mypara.data_path}/tauuv_thetao_SODA__187101_197912_{mypara.mask_name}{std_name}_deg2_anomaly.npy'
# save_data_path = f'{mypara.data_path}/tauuv_thetao_SODA__187101_197912_{mypara.mask_name}20_360_{std_name}_deg2_anomaly.npy'
# data_files.append(in_data_path)
# save_files.append(save_data_path)
#
# in_data_path = f'{mypara.data_path}/tauuv_thetao_ORAS5__195801_197912_{mypara.mask_name}{std_name}_deg2_anomaly.npy'
# save_data_path = f'{mypara.data_path}/tauuv_thetao_ORAS5__195801_197912_{mypara.mask_name}20_360_{std_name}_deg2_anomaly.npy'
# data_files.append(in_data_path)
# save_files.append(save_data_path)
#
#
# in_data_path = f'{mypara.data_path}/tauuv_thetao_GODAS__198001_202112_{mypara.mask_name}{std_name}_deg2_anomaly.npy'
# save_data_path = f'{mypara.data_path}/tauuv_thetao_GODAS__198001_202112_{mypara.mask_name}20_360_{std_name}_deg2_anomaly.npy'
# data_files.append(in_data_path)
# save_files.append(save_data_path)
#
# for file, save_file in zip(data_files, save_files):
#     datain = np.load(file)
#     dataout = datain[:,:,70:121,...]
#     np.save(save_file, dataout)
#     print(datain.shape, dataout.shape)
#
# exit()