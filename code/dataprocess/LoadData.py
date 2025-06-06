import numpy as np
import torch
from torch.utils.data import Dataset

def get_data_files(mypara, pattern_name):
    data_files = []
    mask_name = mypara.mask_name
    if pattern_name == 'CMIP6':
        for sub_pattern_name in mypara.data_names:
            in_data_path = f'{mypara.data_path}/tauuv_thetao_CMIP6_{sub_pattern_name}_185001_201412_{mask_name}std_deg2_anomaly.npy'
            data_files.append(in_data_path)
    elif pattern_name == 'SODA_ORAS5':
        in_data_path = f'{mypara.data_path}/tauuv_thetao_SODA__187101_197912_{mask_name}std_deg2_anomaly.npy'
        data_files.append(in_data_path)
        in_data_path = f'{mypara.data_path}/tauuv_thetao_ORAS5__195801_197912_{mask_name}std_deg2_anomaly.npy'
        data_files.append(in_data_path)
    elif pattern_name == 'GODAS':
        in_data_path = f'{mypara.data_path}/tauuv_thetao_GODAS__198001_202112_{mask_name}std_deg2_anomaly.npy'
        data_files.append(in_data_path)
    elif pattern_name == 'SODA':
        in_data_path = f'{mypara.data_path}/tauuv_thetao_SODA__187101_197912_{mask_name}std_deg2_anomaly.npy'
        data_files.append(in_data_path)
    elif pattern_name == 'ORAS5':
        in_data_path = f'{mypara.data_path}/tauuv_thetao_ORAS5__195801_197912_{mask_name}std_deg2_anomaly.npy'
        data_files.append(in_data_path)
    return data_files

class make_dataset_3(Dataset):
    """
    online reading dataset
    """
    def __init__(self, mypara, pattern_name='CMIP6', data_path=None, flag='train',):
        self.mypara = mypara
        self.needtauxy = mypara.needtauxy
        self.pattern_name = pattern_name
        self.lev_range = mypara.lev_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range
        self.input_length = mypara.input_length
        self.output_length = mypara.output_length
        self.mask_name = self.mypara.mask_name
        self.data_len = mypara.data_len[pattern_name]

        self.data_files = get_data_files(self.mypara, pattern_name)

        self.target_file_name = self.data_files

        if data_path is not None:
            self.target_file_name = [data_path]

        print(len(self.target_file_name))

        data_x, data_y, dataset = [], [], []

        index_len = 0
        for i in range(len(self.target_file_name)):
            train_file = self.target_file_name[i]
            print(train_file)
            data_in = np.load(train_file)

            actual_max = min(self.data_len[1], len(data_in))
            data_in = data_in[self.data_len[0]:actual_max]
            data_in = np.nan_to_num(data_in)
            print(data_in.shape)

            dataset.extend(data_in)
            start = index_len
            index_len += actual_max - self.data_len[0]
            end = index_len

            temp_x, temp_y = self.get_x_y_pairs(start, end, flag)
            data_x.extend(temp_x)
            data_y.extend(temp_y)

        self.data_x = np.array(data_x)
        self.data_y = np.array(data_y)
        self.dataset = torch.from_numpy(np.array(dataset))
        print(self.dataset.shape, self.data_x.shape, self.data_y.shape)

        del data_x, data_y, dataset

        if self.needtauxy == False:
            self.dataset = self.dataset[:, :, 2:, ...]
            print(self.dataset.shape)

    def get_x_y_pairs(self, start, end, flag):
        start, end = start + self.input_length, end - self.output_length if flag != 'test' else end
        dataX, dataY = [], []
        print(start, end)
        for i in range(start, end):
            dataX.append([i - self.input_length, i])
            if flag == 'test':
                dataY.append([i - self.input_length, i])
            else:
                dataY.append([i, i + self.output_length])
        return dataX, dataY


    def __getitem__(self, item):
        data_x = self.dataset[self.data_x[item,0]:self.data_x[item,1],...]
        data_y = self.dataset[self.data_y[item,0]:self.data_y[item,1],...]
        return data_x, data_y
    def __len__(self):
        return len(self.data_x)

