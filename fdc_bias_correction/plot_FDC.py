import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import argparse
import numpy as np
import os
import pandas as pd
import xarray as xr


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_nc", type=str)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--rch", type=int, default=0)
    return parser.parse_args()


args = parse_arguments()
input_file = args.input_nc
output_dir = args.output_dir
show = args.show
rch = args.rch
full = args.full
DS = xr.open_dataset(input_file)

rchid = int(DS["station_rchid"][rch].values)
filename = os.path.join(output_dir, "comparison_FDC_{}.png".format(rchid))

hourly_river_flow = DS["river_flow_rate"][:,rch].to_pandas().dropna()
daily_river_flow = hourly_river_flow.resample('D').mean().dropna()

fdc_hourly = np.sort(hourly_river_flow.values)[::-1]
fdc_daily = np.sort(daily_river_flow.values)[::-1]
p_hourly = np.array([x for x in np.arange(len(fdc_hourly))])/len(fdc_hourly) + 0.01
p_daily = np.array([x for x in np.arange(len(fdc_daily))])/len(fdc_daily) + 0.01

upper_prob = 0.99
lower_prob = 0.20
idx_1_h = np.abs(p_hourly - upper_prob).argmin() + 1
idx_0_h = np.abs(p_hourly - lower_prob).argmin() + 1
idx_1_d = np.abs(p_daily - upper_prob).argmin() + 1
idx_0_d = np.abs(p_daily - lower_prob).argmin() + 1

print(p_hourly)
print(fdc_hourly)
print(p_daily)
print(fdc_daily)
plt.figure(figsize=(16, 10))
plt.yticks(fontsize=12, alpha=.7)
plt.title("Flow Duration Curves (hourly and daily) for %s " % rchid, fontsize=22)
plt.grid(axis='both', alpha=.3)
if full:
    plt.plot(p_hourly, fdc_hourly, color='tab:red')
    plt.plot(p_daily, fdc_daily, color='tab:blue')
else:
    plt.plot(p_hourly[idx_0_h:idx_1_h], fdc_hourly[idx_0_h:idx_1_h], color='tab:red')
    plt.plot(p_daily[idx_0_d:idx_1_d], fdc_daily[idx_0_d:idx_1_d], color='tab:blue')

plt.legend(["Hourly FDC", "Daily FDC"])
plt.gca().spines["top"].set_alpha(0.0)
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)
plt.gca().spines["left"].set_alpha(0.3)
plt.savefig(filename)
if args.show:
    plt.show()
plt.close()


