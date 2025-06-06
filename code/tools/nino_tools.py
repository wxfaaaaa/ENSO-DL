import torch

def pearson_correlation(mypara, x: torch.Tensor, y: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    with torch.no_grad():
        # 输入检查
        assert x.shape == y.shape, "Input tensors must have the same shape."

        # 计算均值，并保持维度以便广播
        mean_x = torch.mean(x, dim=dim, keepdim=True)
        mean_y = torch.mean(y, dim=dim, keepdim=True)

        # 中心化张量
        x_centered = x - mean_x
        y_centered = y - mean_y

        # 计算协方差
        covariance = torch.sum(x_centered * y_centered, dim=dim)

        # 计算标准差
        std_x = torch.sqrt(torch.sum(x_centered ** 2, dim=dim) + eps)
        std_y = torch.sqrt(torch.sum(y_centered ** 2, dim=dim) + eps)

        # 计算皮尔逊相关系数
        pearson = covariance / (std_x * std_y)

        if len(mypara.ninoweight) == 1:
            return pearson.item()
        ones = torch.ones_like(pearson)
        pearson_norm = (ones + pearson) * 0.5

        if mypara.pear_norm:
            pearson = pearson_norm
        else:
            pearson = pearson * mypara.ninoweight.to(pearson.device)

        mse = torch.mean((pearson - pearson_norm) ** 2, dim=dim)
        rmse = mse.sqrt()

        if mypara.rmse_norm:
            loss = rmse
            loss_norm = 1 / (1 + loss)
        elif mypara.mse_norm:
            loss = mse
            loss_norm = 1 / (1 + loss)
        else:
            loss = mse
            loss_norm = loss
        sc = sum(mypara.pear_alpha * pearson + (1 - mypara.pear_alpha) * loss_norm).item()
        return sc

def cal_nino(mypara, datain):
    with torch.no_grad():
        assert len(datain.shape) == 5
        dataout = datain[
                  :,
                  :,
                  mypara.sst_level,
                  mypara.lat_nino_relative[0]: mypara.lat_nino_relative[1],
                  mypara.lon_nino_relative[0]: mypara.lon_nino_relative[1],
                  ].mean(dim=[2, 3])
        return dataout

def get_nino(mypara, var_data):
    if len((var_data.shape)) == 2:
        return var_data
    if len(var_data.shape) == 3:
        return var_data[:, :, 0]
    return cal_nino(mypara, var_data)
