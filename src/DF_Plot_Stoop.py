"""
This plot loads in (or compute) the results of energy drought detection by Method 3 (inspired by Stoop et al. 2024).

For questions, refer to benjamin.biewald@tennet.eu or bastien.cozian@rte-france.com
"""

#%%
# Import packages

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pylab as pylab
import matplotlib.transforms as mtransforms
from datetime import timedelta
import datetime as dtime
import time

from Dunkelflaute_function_library import get_zones, get_thresholds, detect_drought_Otero22, mask_data, get_f_score, get_df_timerange, lin_reg
from CREDIfunctions import Modified_Ordinal_Hour, Climatology_Hourly, Climatology_Hourly_Rolling, \
    Climatology_Hourly_Weekly_Rolling, get_CREDI_events, get_f_score_CREDI, get_f_score_CREDI_new, compute_timeline, get_correlation_CREDI

# Set parameters
path_to_data      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'
path_to_plot      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'
#path_to_data      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'  #'D:/Dunkelflaute/Data/'
#path_to_plot      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'      #'D:/Dunkelflaute/'

plot_format       = 'png'

# Countries (NUT0) of interest (must be a list of two letter abbreviations)
countries = ['DE','NL']

# Target Years
ty_pecd3 = 2033
ty_pecd4 = 2030

# ENS Data (which data to use?)
ens_dataset = 'ERAA23' # 'AO' or 'ERAA23'
# Adequacy Outlook Scenario
ao_scen = 'W.v8' # 'W.v8' or 'S.v6' for either wind or solar dominated scenario (only if ens_dataset='AO')

# Weighting of REP (standard 1, but for experimenting this can be changed)
w=1

#dt_colors = ['green', 'green', 'blue', 'blue', 'orange', 'orange']
#dt_colors = ['limegreen', 'violet', 'dodgerblue', 'darkgreen', 'darkmagenta', 'darkblue']
dt_colors = ['dodgerblue', 'tab:red', '#1e6f4c', 'skyblue', 'burlywood', '#8cc6ad']

# Percentile thresholds to investigate
#LWS_percs = np.linspace(0.0001,0.15,200)
#LWS_percs = np.linspace(0.0001, 0.15, 40)
#LWS_percs = np.array([0.0001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15])
#LWS_percs = np.concatenate([np.linspace(0, 0.02, 21)[1:], np.linspace(0.02, 0.04, 11)[1:], 
#                            np.linspace(0.04, 0.08, 11)[1:],  np.linspace(0.08, 0.16, 11)[1:]])
LWS_percs = np.concatenate([np.linspace(0, 0.02, 51)[1:], np.linspace(0.02, 0.04, 51)[1:], 
                            np.linspace(0.04, 0.08, 21)[1:],  np.linspace(0.08, 0.16, 11)[1:]])
#LWS_percs = np.asarray([0.005, 0.01, 0.02, 0.04])
RL_percs  = 1-LWS_percs
DD_percs  = 1-LWS_percs
        
zones_peon = zones = get_zones(countries,'PEON')  
zones_szon = zones = get_zones(countries,'SZON')  
zones_peof = zones = get_zones(countries,'PEOF')  

scen_Datespan = [42,51,51,51] # TODO: Automate the nr of years per scenario


# =================================================================
# Load hourly data
# =================================================================

if ens_dataset=='AO':
    data3_ENS_h = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_hourly.pkl')
elif ens_dataset=='ERAA23':
    data3_ENS_h = pd.read_pickle(path_to_data+'ERAA23_ENS_TY2033_hourly.pkl')
else:
    raise KeyError('ENS Dataset not existent!')
data3_dem_h = pd.read_pickle(path_to_data+'PEMMDB_demand_TY'+str(ty_pecd3)+'_hourly.pkl')
data4_dem_h = pd.read_pickle(path_to_data+'ETM_demand_TY'+str(ty_pecd4)+'_hourly.pkl')

data4_REP_h = pd.read_pickle(path_to_data+'PECD4_Generation_TY'+str(ty_pecd4)+'_national_hourly.pkl')
data3_REP_h = pd.read_pickle(path_to_data+'PECD3_Generation_TY'+str(ty_pecd3)+'_national_hourly.pkl')

# Weight REP for experimenting
start_date = '1982-01-01 00:00:00'
end_date   = '2016-12-31 00:00:00'
data3_cropped1 = data3_REP_h.query('Date>=@start_date and Date <= @end_date')
data4_cropped1 = data4_REP_h.query('Date>=@start_date and Date <= @end_date')
data3_gen_h = data3_cropped1[~((data3_cropped1.index.get_level_values(1).day == 29) & (data3_cropped1.index.get_level_values(1).month == 2))]
data4_gen_h = data4_cropped1[~((data4_cropped1.index.get_level_values(1).day == 29) & (data4_cropped1.index.get_level_values(1).month == 2))]

data3_RL_h = data3_dem_h - w*data3_gen_h
data4_RL_h = data4_dem_h - w*data4_gen_h

# TODO: Need to correct the Preprocessing for hourly data (perhaps daily data?) of demand 
# to account for 2016-12-31T01:00 à 2016-12-31T23:00. Currently, I believe that there is only 2016-12-31T00:00

# TODO: Use the calendar method of ERAA 2023: calendar of 2018, remove last day of year instead of Feb 29th in leap years.
# Why? -> 1) there is likely a 1-day shift in the ENS compared to energy variables once every four years
#         2) Needed to correctly implement the method based on Hourly & Weekly Rolling Window


# =================================================================
# Load daily data
# =================================================================

if ens_dataset=='AO':
    data3_ENS_d = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_daily.pkl')
elif ens_dataset=='ERAA23':
    data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_ENS_TY2033_daily.pkl')
else:
    raise KeyError('ENS Dataset not existent!')
data3_dem_d = pd.read_pickle(path_to_data+'PEMMDB_demand_TY'+str(ty_pecd3)+'_daily.pkl')
data4_dem_d = pd.read_pickle(path_to_data+'ETM_demand_TY'+str(ty_pecd4)+'_daily.pkl')

data4_REP_d = pd.read_pickle(path_to_data+'PECD4_Generation_TY'+str(ty_pecd4)+'_national_daily.pkl')
data3_REP_d = pd.read_pickle(path_to_data+'PECD3_Generation_TY'+str(ty_pecd3)+'_national_daily.pkl')

# Weight REP for experimenting
start_date = '1982-01-01 00:00:00'
end_date   = '2016-12-31 00:00:00'
data3_cropped1 = data3_REP_d.query('Date>=@start_date and Date <= @end_date')
data4_cropped1 = data4_REP_d.query('Date>=@start_date and Date <= @end_date')
data3_gen_d = data3_cropped1[~((data3_cropped1.index.get_level_values(1).day == 29) & (data3_cropped1.index.get_level_values(1).month == 2))]
data4_gen_d = data4_cropped1[~((data4_cropped1.index.get_level_values(1).day == 29) & (data4_cropped1.index.get_level_values(1).month == 2))]

data3_RL_d = data3_dem_d - w*data3_gen_d
data4_RL_d = data4_dem_d - w*data4_gen_d





















#%% 
# =================================================================
# Plot PDF of CREDI
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# Duration of CREDI events
PERIOD_length_days = 3
# Clustering period. If PERIOD_length_days < PERIOD_cluster_days, then there may be overlaps. We authorize up to 25% overlap.
PERIOD_cluster_days = 3

# percentile for peak F-score. Computed in `DF_Validation_Stoop.py`
#p_max = 0.01
p_max = 0.022
#p_max = 0.0356
#p_max = 0.044

zone = 'DE00'

ens_dataset = 'ERAA23'

figname = f"PDF_Stoop24_{ens_dataset}_ENS_{zone}_T{PERIOD_length_days}_Tc{PERIOD_cluster_days}_pmax_{int(p_max*1000)}e-3"

# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24

if ens_dataset=='AO':
    data3_ENS_d = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_daily.pkl')
elif ens_dataset=='ERAA23':
    data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_ENS_TY2033_daily.pkl')
else:
    raise KeyError('ENS Dataset not existent!')

# Generate masked data
ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
common_index = data3_RL_h.loc[('HIST')].index.intersection(ens_mask.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

rl3_thresh = np.quantile(rl3_event_values, q=1-p_max, interpolation="nearest")


# ---------------------------------------------
#  Plot PDF
# ---------------------------------------------
CREDI_events = np.asarray(rl3_event_values) / 1000 # MWh -> GWh
thresh = np.asarray(rl3_thresh) / 1000 # MWh -> GWh

print(f'Threshold = {thresh} (p = {p_max})')

binsize = 40

fig, ax = plt.subplots(1, 1, figsize=(5,5))

bins = np.linspace(np.min(CREDI_events), np.max(CREDI_events), binsize)
hist = np.histogram(CREDI_events, bins=bins)
x = hist[1][:-1] + np.diff(hist[1])[0]/2
ax.plot(x, hist[0], drawstyle='steps', label='RL', color=dt_colors[0])

ax.set_title(f'RL (PECD 3.1, {ens_dataset}, {zone}, T={PERIOD_length_days}, Tc={PERIOD_cluster_days}, p={p_max})')
ax.axvline(thresh, alpha=0.5, color='black')
#ax.set_xlim(0, np.max(CREDI_events))
ax.set_ylabel('Number of CREDI events')
ax.set_xlabel('CREDI [GWh]')
ax.legend(facecolor="white", loc='upper right', framealpha=1)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}PDF/{figname}.{plot_format}", dpi=300)
#plt.close()

print(f"Saved {path_to_plot}PDF/{figname}.{plot_format}")


















#%% 
# =================================================================
# Plot Correlation between ENS and Severity
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# Duration of CREDI events
PERIOD_length_days = 1
# Clustering period. If PERIOD_length_days < PERIOD_cluster_days, then there may be overlaps. We authorize up to 25% overlap.
PERIOD_cluster_days = 1

# percentile for peak F-score. Computed in `DF_Validation_Stoop.py`
p_max = 0.01
#p_max = 0.022
#p_max = 0.0356
#p_max = 0.044

zone = 'DE00'

ens_dataset = 'ERAA23'

figname = f"Correlation_ENSvsCREDI_Stoop24_{ens_dataset}_ENS_{zone}_T{PERIOD_length_days}_Tc{PERIOD_cluster_days}_pmax_{int(p_max*1000)}e-3"

# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24

if ens_dataset=='AO':
    data3_ENS_d = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_daily.pkl')
elif ens_dataset=='ERAA23':
    data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_ENS_TY2033_daily.pkl')
else:
    raise KeyError('ENS Dataset not existent!')

# Generate masked data
ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
common_index = data3_RL_h.loc[('HIST')].index.intersection(ens_mask.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

rl3_thresh = np.quantile(rl3_event_values, q=1-p_max, interpolation="nearest")

# Only true positives
rl3_DF_TP, rl3_sum_ENS_TP = get_correlation_CREDI(data3_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                            PERIOD_length_days=1, extreme_is_high=True, only_true_positive=True)
# With TP, FP, FN
rl3_DF_all, rl3_sum_ENS_all = get_correlation_CREDI(data3_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                            PERIOD_length_days=1, extreme_is_high=True, only_true_positive=False)


# ---------------------------------------------
#  Plot Correlation
# ---------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(5,5))

x = rl3_DF_all
y = rl3_sum_ENS_all
# Linear regression
df_reg, intercept, slope, r_value, p_value, reg_trend = lin_reg(x, y)
if p_value < 0.05:
    ls  = 'solid'
else:
    ls = 'dotted'
ax.scatter(x, y, color=dt_colors[3], label=f'RL (r={np.round(r_value, 2)})', alpha=1)
ax.plot(df_reg, c=dt_colors[3], linestyle=ls)
print(f' RL  (3.1) intercept={intercept}, slope={slope}, r_value={r_value}, p_value={p_value}, reg_trend={reg_trend}')


x = rl3_DF_TP
y = rl3_sum_ENS_TP
# Linear regression
df_reg, intercept, slope, r_value, p_value, reg_trend = lin_reg(x, y)
if p_value < 0.05:
    ls  = 'solid'
else:
    ls = 'dotted'
ax.scatter(x, y, color=dt_colors[0], label=f'RL only TP (r={np.round(r_value, 2)})', alpha=1)
ax.plot(df_reg, c=dt_colors[0], linestyle=ls)
print(f' RL  (3.1) intercept={intercept}, slope={slope}, r_value={r_value}, p_value={p_value}, reg_trend={reg_trend}')

ax.set_title(f'RL (PECD 3.1, {ens_dataset}, {zone}, T={PERIOD_length_days}, Tc={PERIOD_cluster_days}, p={p_max})')
ax.set_ylabel('Summed up ENS [GWh]')
ax.set_xlabel('CREDI [GWh]')
ax.legend(facecolor="white", loc='upper left', framealpha=1)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Correlation/{figname}.{plot_format}", dpi=300)
#plt.close()

print(f"Saved {path_to_plot}Correlation/{figname}.{plot_format}")











# %%
