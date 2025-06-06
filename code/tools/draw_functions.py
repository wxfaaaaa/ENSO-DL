import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from .my_tools import cal_ninoskill2, runmean, RunMean, GetDJF
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
from .data_tools import get_std
import random
import colorsys
import os
from copy import deepcopy
os.environ["OMP_NUM_THREADS"] = "1"

class RunMean():
    def __init__(self):
        pass
    def __call__(self, data, n_run=3):
        # data: 1983-2021, n_run: 3
        ll = data.shape[0]
        data_run = np.zeros([ll])
        for i in range(ll):
            if i < (n_run - 1):
                data_run[i] = np.nanmean(data[0: i + 1])
            else:
                data_run[i] = np.nanmean(data[i - n_run + 1: i + 1])
        return data_run

class GetDJF():
    def __init__(self):
        pass

    def __call__(self, data):
        assert len(data) % 12 == 0
        djf = []
        for i in range(0, len(data) - 12, 12):
            djf.append((data[i + 11] + data[i + 12] + data[i + 13]) / 3)
        return djf

def generate_distinct_colors(n, saturation=0.9, value=0.9):
    colors = []
    for i in range(n):
        hue = i / n  # 均匀分布色相 (0.0~1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


class Plot():
    def __init__(self, mypara):
        self.mypara = mypara
        self.nino_info = {'Niño34':{'start_lon':49, 'end_lon':75, 'start_lat':15, 'end_lat':36},
                     'Niño3':{'start_lon':59, 'end_lon':90, 'start_lat':15, 'end_lat':36},
                     'Niño4':{'start_lon':34, 'end_lon':60, 'start_lat':15, 'end_lat':36},
                    }

        self.nino_names = ['Niño34', 'Niño3','Niño4']
        self.climate_name = self.nino_names
        self.djf = True
        self.gap = mypara.gap
        if self.djf:
            self.mean_func = GetDJF()
        else:
            self.mean_func = RunMean()
        self.x_label = mypara.x_label
        self.y_label = mypara.y_label
        self.title_label = mypara.title_label
        self.label_fontsize = mypara.label_fontsize
        self.title_fontsize = mypara.title_fontsize
        self.font_size = mypara.font_size
        self.save_name = mypara.save_name
        self.pattern_name = mypara.test_pattern
        self.start_time = 1970 if self.pattern_name == 'CMIP6' else 1983
        self.end_time = 2014 if self.pattern_name == 'CMIP6' else 2021
        self.thick_line = 3
        self.thin_line = 1

    def get_lon_lat(self, name, gap):
        start_lon = self.nino_info[name]['start_lon'] + gap
        end_lon = self.nino_info[name]['end_lon'] + gap
        start_lat = self.nino_info[name]['start_lat']
        end_lat = self.nino_info[name]['end_lat']
        return start_lon, end_lon, start_lat, end_lat
    def calculate_values(self, sst_preds, sst_trues):
        values, preds, trues = [], [], []
        for name in self.climate_name:
            start_lon, end_lon, start_lat, end_lat = self.get_lon_lat(name, self.gap)
            sst_preds = np.array(sst_preds)
            sst_trues = np.array(sst_trues)
            x = np.nanmean(sst_preds[:, :, start_lat:end_lat, start_lon:end_lon], axis=(2, 3))
            y = np.nanmean(sst_trues[:, start_lat:end_lat, start_lon:end_lon], axis=(1, 2))
            corr, rmse, mae = [], [], []
            xx, yy = [], []
            for l in range(self.mypara.output_length):
                aa = self.mean_func(x[l])
                bb = self.mean_func(y)
                xx.append(aa)
                yy.append(bb)
                corr.append(pearsonr(aa, bb)[0])
                rmse.append(mean_squared_error(aa, bb)**0.5)
                mae.append(mean_absolute_error(aa, bb))
            values.append([corr, rmse, mae])
            preds.append(xx)
            trues.append(yy)
        values = np.array(values) # (3,3,lead)
        preds = np.array(preds) # (3,lead)
        trues = np.array(trues) # (3,lead)
        return values, preds, trues

    def draw_nino34(self, values, preds, trues, sub_pattern_name, load_path, flag=None):
        lead_max = self.mypara.output_length
        if flag == 'mean':
            colors = ['blue', 'green', 'purple', 'orange']
        else:
            colors = generate_distinct_colors(preds.shape[1]) if preds.shape[1] > 1 else ['b']

        for name, value, nino_pred, nino_true in zip(self.climate_name, values, preds, trues):
            corr, rmse, _ = value[:, 0], value[:, 1], value[:, 2]  # (n, lead)
            # print()
            mean_pred = np.mean(nino_pred, axis=0)  # (20, len)
            mean_true = np.mean(nino_true, axis=0)  # (20, len)
            mean_corr, mean_rmse = [], []
            for i in range(len(mean_pred)):
                mean_corr.append(pearsonr(mean_pred[i], mean_true[i])[0])
                mean_rmse.append(mean_squared_error(mean_pred[i], mean_true[i]) ** 0.5)

            # 确定需要绘制的lead索引
            selected_leads = [i for i in range(lead_max) if (i + 1) % 5 == 0]
            n_subplots = len(selected_leads)

            # 动态创建子图
            fig, axs = plt.subplots(n_subplots, 1, figsize=(25, 5 * n_subplots))
            if n_subplots == 1: axs = [axs]  # 处理单子图情况
            x = range(self.start_time, self.end_time, 1)

            temp_name = f'{sub_pattern_name} {self.start_time}-{self.end_time-1} DJF {name}'

            for ax_idx, lead_idx in enumerate(selected_leads):
                ax = axs[ax_idx]
                lead = lead_idx + 1
                # 绘制预测曲线
                for i, pred in enumerate(nino_pred):
                    color = colors[i]
                    linestyle = '--' if (flag == None) else '-'
                    linewidth = self.thin_line if (flag == None) else self.thick_line
                    label = None if (flag == None) else f'Corr={corr[i, lead_idx]:.3f}'
                    ax.plot(x, pred[lead_idx], color=color, linewidth=linewidth, linestyle=linestyle, label=label)

                # 绘制真实值和均值
                ax.plot(x, nino_true[0, lead_idx], 'r-', linewidth=self.thick_line, label='Actual')
                if flag != 'mean':
                    ax.plot(x, mean_pred[lead_idx], 'b-', linewidth=self.thick_line, label='Mean')

                ax.set_ylabel('Correlation', fontsize=self.label_fontsize)
                # 统一样式配置
                if flag == 'mean':
                    title_name = (f"{temp_name} Lead={lead} "
                                  f"CMax:{corr[:,lead_idx].max():.3f},CMin:{corr[:,lead_idx].min():.3f} "
                                  f"RMax:{rmse[:,lead_idx].max():.3f},RMin:{rmse[:,lead_idx].min():.3f}")
                else:
                    title_name = (f"{temp_name} Lead={lead} "
                                  f"Corr: {mean_corr[lead_idx]:.3f},CMax:{corr[:,lead_idx].max():.3f},CMin:{corr[:,lead_idx].min():.3f} "
                                  f"RMSE: {mean_rmse[lead_idx]:.3f},RMax:{rmse[:,lead_idx].max():.3f},RMin:{rmse[:,lead_idx].min():.3f}")
                ax.set_title(title_name, fontsize=self.title_fontsize)

                ax.set_xticks(x, [str(a) for a in x], fontsize=self.font_size)
                ax.tick_params(axis='y', labelsize=self.label_fontsize)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.axhline(0, color='black', linestyle='--', linewidth=self.thick_line)
                ax.legend(fontsize=self.label_fontsize, loc='upper right')

            plt.tight_layout()
            save_path = f"{load_path}/{temp_name}_ensemble_predict_actual_{len(nino_pred)}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
            print(f"图像已保存到 {save_path}")



