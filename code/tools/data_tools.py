import numpy as np

def erci_detrend_4d(data, deg):
    """
    对四维数据（时间×层数×纬度×经度）进行多项式去趋势处理，支持NaN值

    参数：
        data : 四维numpy数组，形状为 (时间, 层数, 纬度, 经度)
        deg  : 多项式拟合阶数（如线性趋势=1，二次趋势=2）

    返回：
        detrended_data : 去趋势后的四维数组，形状与输入一致
    """
    # 记录原始形状和NaN掩码
    t, z, y, x = data.shape
    nan_mask = np.isnan(data)

    # 重塑为二维数组：时间 × (层数×纬度×经度)
    reshaped_data = data.reshape((t, -1))
    nan_mask_reshaped = nan_mask.reshape((t, -1))

    # 时间轴坐标
    time_axis = np.arange(t)
    detrended_flat = np.full_like(reshaped_data, np.nan)

    # 对每个空间点（层数×纬度×经度组合）独立处理
    for i in range(reshaped_data.shape[1]):
        valid = ~nan_mask_reshaped[:, i]
        if np.sum(valid) >= deg + 1:  # 动态检查有效点数
            coeffs = np.polyfit(time_axis[valid], reshaped_data[valid, i], deg)
            trend = np.polyval(coeffs, time_axis)
            detrended_flat[valid, i] = reshaped_data[valid, i] - trend[valid]

    # 恢复四维形状
    return detrended_flat.reshape((t, z, y, x))

def erci_detrend(data, deg):
    array_shape = data.shape
    nan_mask = np.isnan(data)

    print(1, array_shape, data.shape, array_shape[0])
    # 将数组重塑为二维数组，其中第0维度作为样本，其余维度作为特征

    reshaped_data = np.reshape(data, (array_shape[0], -1))
    nan_mask_reshaped = np.reshape(nan_mask, (array_shape[0], -1))
    # 对每个样本拟合二次多项式
    x = np.arange(array_shape[0])
    detrended_data = np.full_like(reshaped_data, np.nan)
    for i in range(reshaped_data.shape[1]):
        valid_idx = ~nan_mask_reshaped[:, i]
        if np.sum(valid_idx) > 2:  # 需要至少3个点进行二次拟合
            fit_coeffs = np.polyfit(x[valid_idx], reshaped_data[valid_idx, i], deg)
            quadratic_trend = np.polyval(fit_coeffs, x)
            detrended_data[valid_idx, i] = reshaped_data[valid_idx, i] - quadratic_trend[valid_idx]
    # 将数据重新塑形回原始形状
    detrended_data = np.reshape(detrended_data, array_shape)
    return detrended_data

def get_std(thetao_tauuv_anomlay, mypara):
    """
    计算并标准化输入数据的标准差。

    参数:
    thetao_tauuv_anomlay (numpy array): 四维数组，形状为 [datalen, lev, lat, lon]。

    返回:
    numpy array: 标准化后的数据。
    """
    # 获取数据形状
    datalen, lev, _, _ = thetao_tauuv_anomlay.shape
    stds = []
    std_map = []
    lat_start, lat_end = mypara.lat_range[0], mypara.lat_range[1]
    lon_start, lon_end = mypara.lon_range[0], mypara.lon_range[1]
    lat = lat_end - lat_start
    lon = lon_end - lon_start
    for i in range(lev):
        # 提取当前层次的数据
        temp_data = thetao_tauuv_anomlay[:, i, lat_start:lat_end, lon_start:lon_end]  # [datalen, lat, lon]

        # 重塑数据为二维数组 [datalen, lat * lon]
        temp_data_reshaped = temp_data.reshape(datalen, -1)

        # 计算每个网格点的标准差
        temp_std = np.nanstd(temp_data_reshaped, axis=0)
        std_map.append(temp_std)

        # 计算平均标准差
        mean_std = np.nanmean(temp_std)

        stds.append(mean_std)
        # 标准化数据
        thetao_tauuv_anomlay[:, i, ...] /= mean_std
        # print(mean_std.shape, temp_std.shape, temp_data_reshaped.shape, temp_data.shape)
    stds = np.array(stds)
    std_map = np.array(std_map).reshape(lev, lat, lon)

    return thetao_tauuv_anomlay, stds, std_map

