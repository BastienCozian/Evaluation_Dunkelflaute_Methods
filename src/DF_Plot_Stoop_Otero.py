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
end_date   = '2016-12-31 23:00:00'
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
start_date = '1982-01-01'
end_date   = '2016-12-31'
data3_cropped1 = data3_REP_d.query('Date>=@start_date and Date <= @end_date')
data4_cropped1 = data4_REP_d.query('Date>=@start_date and Date <= @end_date')
data3_gen_d = data3_cropped1[~((data3_cropped1.index.get_level_values(1).day == 29) & (data3_cropped1.index.get_level_values(1).month == 2))]
data4_gen_d = data4_cropped1[~((data4_cropped1.index.get_level_values(1).day == 29) & (data4_cropped1.index.get_level_values(1).month == 2))]

data3_RL_d = data3_dem_d - w*data3_gen_d
data4_RL_d = data4_dem_d - w*data4_gen_d





















#%% 
# =================================================================
# Compute thresholds for Otero
# =================================================================

# Percentile thresholds & capacity reference year from Otero et al. 2022
# Paper used 0.1 and 0.9, validation yielded 0.01 and 0.99 / 0.98 as best fits
LWS_percentile = 0.01 # 0.01
RL_percentile  = 0.99 # 0.98
DD_percentile  = 0.99

# Load the preprocessed data (DataFrames as pickles)

lws3_thresh, lws3_sigma = get_thresholds(data3_REP_d.loc[('HIST')], LWS_percentile, start_date='1980-01-01', end_date='2019-12-31', empirical=True)
rl3_thresh,  rl3_sigma  = get_thresholds(data3_RL_d.loc[('HIST')],  RL_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)
dd3_thresh,  dd3_sigma  = get_thresholds(data3_dem_d.loc[('HIST')], DD_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)

# Detect Drought days and events
lws3_days, lws3_events = detect_drought_Otero22(['HIST'],  zones_szon, data3_REP_d, lws3_thresh, lws3_sigma)
rl3_days,  rl3_events  = detect_drought_Otero22(['HIST'],  zones_szon, data3_RL_d,  rl3_thresh,  rl3_sigma, below=False)
dd3_days,  dd3_events  = detect_drought_Otero22(['HIST'],  zones_szon, data3_dem_d, dd3_thresh,  dd3_sigma, below=False)

# Mask the data / Detect Drought days
#ens_mask  = mask_data(data3_ENS_d, 0, False, 2, 0)
lws3_mask = mask_data(data3_REP_d, lws3_thresh, True,  1, 0)
rl3_mask  = mask_data(data3_RL_d,  rl3_thresh,  False, 1, 0)
dd3_mask  = mask_data(data3_dem_d, dd3_thresh,  False, 1, 0)











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
PERIOD_length_days = 7
# Clustering period. If PERIOD_length_days < PERIOD_cluster_days, then there may be overlaps. We authorize up to 25% overlap.
PERIOD_cluster_days = 6

# percentile for peak F-score. Computed in `DF_Validation_Stoop.py`
#p_max = 0.01
#p_max = 0.022
#p_max = 0.0356
p_max = 0.044

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
# =================================================================
# Plot F-score for PECD3 - Otero vs. Stoop
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

zone = 'DE00'

ens_dataset = 'ERAA23'

figname = f"Fscore_Comparison_Otero-Stoop_{ens_dataset}_ENS_{zone}_T{PERIOD_length_days}"


# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()

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


# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

common_index = data3_RL_h.loc[('HIST')].index.intersection(ens_mask.index)

# Get CREDI events
lws3_CREDI_event, lws3_event_dates, \
    lws3_event_values = get_CREDI_events(data3_gen_h.loc[('HIST')][[zone]], zone, extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                         PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
dd3_CREDI_event, dd3_event_dates, \
    dd3_event_values = get_CREDI_events(data3_dem_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

p_list  = []
for p in range(len(LWS_percs)):

    ## Otero
    # Find capacity thresholds
    Otero_lws3_thresh, Otero_lws3_sigma = get_thresholds(data3_REP_d.loc[('HIST')][[zone]], LWS_percs[p], start_date='1980-01-01', end_date='2019-12-31', empirical=True)
    Otero_rl3_thresh,  Otero_rl3_sigma  = get_thresholds(data3_RL_d.loc[('HIST')][[zone]],  RL_percs[p] , start_date='1980-01-01', end_date='2019-12-31', empirical=True)
    Otero_dd3_thresh,  Otero_dd3_sigma  = get_thresholds(data3_dem_d.loc[('HIST')][[zone]], DD_percs[p] , start_date='1980-01-01', end_date='2019-12-31', empirical=True)

    # Mask the data / Detect Drought days
    Otero_lws3_mask = mask_data(data3_REP_d.loc[('HIST')][[zone]], Otero_lws3_thresh, True,  1, 0)
    Otero_rl3_mask  = mask_data(data3_RL_d.loc[('HIST')][[zone]],  Otero_rl3_thresh,  False, 1, 0)
    Otero_dd3_mask  = mask_data(data3_dem_d.loc[('HIST')][[zone]], Otero_dd3_thresh,  False, 1, 0)

    # Calculate F (compared to ENS)
    Otero_lws3_stat = get_f_score(ens_mask, Otero_lws3_mask, beta=1)
    Otero_rl3_stat  = get_f_score(ens_mask, Otero_rl3_mask,  beta=1)
    Otero_dd3_stat  = get_f_score(ens_mask, Otero_dd3_mask,  beta=1)

    ## Stoop
    # find capacity thresholds
    # Percentile is computed on the clustered events ("independent" events)
    # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
    Stoop_lws3_thresh = np.quantile(lws3_event_values, q=LWS_percs[p], interpolation="nearest")
    Stoop_rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")
    Stoop_dd3_thresh = np.quantile(dd3_event_values, q=DD_percs[p], interpolation="nearest")

    # Calculate F (compared to ENS)
    Stoop_lws3_stat = get_f_score_CREDI_new(data3_ENS_d[[zone]], lws3_event_dates, lws3_event_values, Stoop_lws3_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=False, beta=1)
    Stoop_rl3_stat = get_f_score_CREDI_new(data3_ENS_d[[zone]], rl3_event_dates, rl3_event_values, Stoop_rl3_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)
    Stoop_dd3_stat = get_f_score_CREDI_new(data3_ENS_d[[zone]], dd3_event_dates, dd3_event_values, Stoop_dd3_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)

    # Create Dataframe
    p_list.append( pd.concat([Stoop_lws3_stat, Stoop_rl3_stat, Stoop_dd3_stat, Otero_lws3_stat, Otero_rl3_stat, Otero_dd3_stat], 
                             keys=['LWS (Stoop)', 'RL (Stoop)', 'DD (Stoop)', 'LWS (Otero)', 'RL (Otero)', 'DD (Otero)'], 
                             names=['Drought type']))
    print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
stat_df  = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
stat_df.to_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")
stat_df.to_csv(f"{path_to_plot}Plot_data/{figname}_stats.csv", sep=';')


#%%
# ---------------------------------------------
#  Plot F / threshold (comparing to ENS / validation)
# ---------------------------------------------

# Load data
stat_df = pd.read_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")

x=stat_df.index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)


fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle('Stoop Method for '+zone+' using '+ens_dataset+' ENS')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
ax_big.plot(x, stat_df.loc[(x, 'RL (Stoop)',  'F'),(zone)], label='RL (Stoop)',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'DD (Stoop)',  'F'),(zone)], label='DD (Stoop)',  color=dt_colors[1], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'LWS (Stoop)',  'F'),(zone)], label='LWS (Stoop)',  color=dt_colors[2], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL (Otero)',  'F'),(zone)], label='RL (Otero)',  color=dt_colors[3], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'DD (Otero)',  'F'),(zone)], label='DD (Otero)',  color=dt_colors[4], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'LWS (Otero)',  'F'),(zone)], label='LWS (Otero)',  color=dt_colors[5], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Stoop)',  'TP'),(zone)], label='RL (Stoop)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Stoop)',  'TP'),(zone)], label='DD (Stoop)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Stoop)',  'TP'),(zone)], label='LWS (Stoop)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Otero)',  'TP'),(zone)], label='RL (Otero)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Otero)',  'TP'),(zone)], label='DD (Otero)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Otero)',  'TP'),(zone)], label='LWS (Otero)',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Stoop)',  'TN'),(zone)], label='RL (Stoop)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Stoop)',  'TN'),(zone)], label='DD (Stoop)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Stoop)',  'TN'),(zone)], label='LWS (Stoop)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Otero)',  'TN'),(zone)], label='RL (Otero)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Otero)',  'TN'),(zone)], label='DD (Otero)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Otero)',  'TN'),(zone)], label='LWS (Otero)',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Stoop)',  'FP'),(zone)], label='RL (Stoop)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Stoop)',  'FP'),(zone)], label='DD (Stoop)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Stoop)',  'FP'),(zone)], label='LWS (Stoop)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Otero)',  'FP'),(zone)], label='RL (Otero)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Otero)',  'FP'),(zone)], label='DD (Otero)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Otero)',  'FP'),(zone)], label='LWS (Otero)',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Stoop)',  'FN'),(zone)], label='RL (Stoop)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Stoop)',  'FN'),(zone)], label='DD (Stoop)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Stoop)',  'FN'),(zone)], label='LWS (Stoop)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Otero)',  'FN'),(zone)], label='RL (Otero)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Otero)',  'FN'),(zone)], label='DD (Otero)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Otero)',  'FN'),(zone)], label='LWS (Otero)',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Stoop)',  'PR'),(zone)], label='RL (Stoop)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Stoop)',  'PR'),(zone)], label='DD (Stoop)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Stoop)',  'PR'),(zone)], label='LWS (Stoop)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL (Stoop)',  'RE'),(zone)], label='RL (Stoop)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD (Stoop)',  'RE'),(zone)], label='DD (Stoop)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS (Stoop)',  'RE'),(zone)], label='LWS (Stoop)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('Recall (out of '+str(nr_of_pos[zone])+' in total)\n(How many ENS are identified as droughts?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.{plot_format}", dpi=300)
#plt.close()

print(f"Saved {path_to_plot}Validation/{figname}.{plot_format}")




















# %%

#%% Plot Correlation of ENS/Severity and ENS/Duration

# Units for plotting
unit = 'TWh'
scaling = 0.000001 # 'unit' in MWh (bc original data is in MWh)

# Load in Data
"""data4_REP_d = pd.read_pickle(path_to_data+'PECD4_Generation_TY'+str(ty_pecd4)+'_national_daily.pkl') * scaling
data3_REP_d = pd.read_pickle(path_to_data+'PECD3_Generation_TY'+str(ty_pecd3)+'_national_daily.pkl') * scaling
data4_RL_d  = pd.read_pickle(path_to_data+'PECD4_ETM_RL_TY'+str(ty_pecd4)+'_national_daily.pkl') * scaling
data3_RL_d  = pd.read_pickle(path_to_data+'PECD3_PEMMDB_RL_TY'+str(ty_pecd3)+'_national_daily.pkl') * scaling
data4_dem_d = pd.read_pickle(path_to_data+'ETM_demand_TY'+str(ty_pecd4)+'_daily.pkl') * scaling
data3_dem_d = pd.read_pickle(path_to_data+'PEMMDB_demand_TY'+str(ty_pecd3)+'_daily.pkl') * scaling
data4_CF_h = pd.read_pickle(path_to_data+'PECD4_CF_TY'+str(ty_pecd4)+'_national_hourly.pkl')
data3_CF_h = pd.read_pickle(path_to_data+'PECD3_CF_TY'+str(ty_pecd3)+'_national_hourly.pkl')

lws4_events = pd.read_pickle(path_to_data+'LWS4_TY'+str(ty_pecd4)+'_Thresh_'+str(LWS_percentile)+'_events.pkl')
rl4_events  = pd.read_pickle(path_to_data+'RL4_TY'+str(ty_pecd4)+'_Thresh_'+str(RL_percentile)+'_events.pkl')
dd4_events  = pd.read_pickle(path_to_data+'DD4_TY'+str(ty_pecd4)+'_Thresh_'+str(DD_percentile)+'_events.pkl')
#cfd4_events = pd.read_pickle(path_to_data+'CFD4_TY'+str(ty_pecd4)+'_Thresh_'+str(CF_thresh)+'_events.pkl')

lws3_events = pd.read_pickle(path_to_data+'LWS3_TY'+str(ty_pecd3)+'_Thresh_'+str(LWS_percentile)+'_events.pkl')
rl3_events  = pd.read_pickle(path_to_data+'RL3_TY'+str(ty_pecd3)+'_Thresh_'+str(RL_percentile)+'_events.pkl')
dd3_events  = pd.read_pickle(path_to_data+'DD3_TY'+str(ty_pecd3)+'_Thresh_'+str(DD_percentile)+'_events.pkl')
#cfd3_events = pd.read_pickle(path_to_data+'CFD3_TY'+str(ty_pecd3)+'_Thresh_'+str(CF_thresh)+'_events.pkl')

#cfd4_hours = pd.read_pickle(path_to_data+'CFD4_TY'+str(ty_pecd4)+'_Thresh_'+str(CF_thresh)+'_hours.pkl')
#cfd3_hours = pd.read_pickle(path_to_data+'CFD3_TY'+str(ty_pecd3)+'_Thresh_'+str(CF_thresh)+'_hours.pkl')"""

test = np.zeros((2,2,4,2)) # country, 3.1/4.1, Droughttype, Sev/Dur

def get_ENS_sums(data_ens_sum_d, df_event, zone, scale=1):
    # only one column of data_ens_sum_d for the specified country
    # only one scenario and one country of df_event
    #index_of_tp_days = ens_days.index.intersection(df_days.index)
    
    ens_sum = np.ones(df_event.loc[zone].shape[0])
    df_event_c = df_event.loc[zone]

    startdates = df_event_c['Startdate']
    enddates =   df_event_c['Startdate'] + (df_event_c['Duration']/scale).map(timedelta)

    for e in range(df_event_c.shape[0]):
        event = get_df_timerange(data_ens_sum_d,startdates.iloc[e],enddates.iloc[e])[zone]
        if event.empty:
            ens_sum[e] = np.nan
        else:
            ens_sum[e] = np.array(event.sum(axis=0))
    return ens_sum/1000 # MWh in GWh

for c in range(len(zones_szon)):
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    
    idx, idy = 0, 0
    #print(zones_szon[c]+' LWS (4.1) Sev R value: '+str(lin_reg(x, y)[3]))
    x=lws3_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    y=get_ENS_sums(data3_ENS_d, lws3_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[0], label='LWS (PECD 3.1)', alpha=0.5)
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[0], label='Lin Reg (PECD3.1)')
    test[c,1,0,0]=lin_reg(x, y)[3]
    #print(zones_szon[c]+' LWS (3.1) Sev R value: '+str(lin_reg(x, y)[3]))
    axs[idx, idy].set_title('LWS ('+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Summed up ENS [GWh]')
    axs[idx, idy].set_xlabel('Severity')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 1, 0
    #print(zones_szon[c]+' RL  (4.1) Sev R value: '+str(lin_reg(x, y)[3]))
    x=rl3_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    y=get_ENS_sums(data3_ENS_d, rl3_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[1], label='RL (PECD 3.1)', alpha=0.5)
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[1], label='Lin Reg (PECD3.1)')
    test[c,1,1,0]=lin_reg(x, y)[3]
    #print(zones_szon[c]+' RL  (3.1) Sev R value: '+str(lin_reg(x, y)[3]))
    axs[idx, idy].set_title('RL ('+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Summed up ENS [GWh]')
    axs[idx, idy].set_xlabel('Severity')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 0, 1
    x=dd3_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    y=get_ENS_sums(data3_ENS_d, dd3_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[2], label='DD (PECD 3.1)', alpha=0.5)
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[2], label='Lin Reg (PECD3.1)')
    test[c,1,2,0]=lin_reg(x, y)[3]
    #print(zones_szon[c]+' DD  (3.1) Sev R value: '+str(lin_reg(x, y)[3]))
    axs[idx, idy].set_title('DD ('+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Summed up ENS [GWh]')
    axs[idx, idy].set_xlabel('Severity')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    plt.tight_layout()
    #plt.savefig(path_to_plot+'Correlation/DF_CorrelationENSSeverity_'+ens_dataset+'_'+zones_szon[c]+'.'+plot_format,dpi=300)
    #plt.close()
    
    
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    
    idx, idy = 0, 0
    x=lws3_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=get_ENS_sums(data3_ENS_d, lws3_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[0], label='LWS (PECD 3.1)', alpha=0.5)
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[0], label='Lin Reg (PECD3.1)')
    #print(zones_szon[c]+' LWS (3.1) Dur R value: '+str(lin_reg(x, y)[3]))
    test[c,1,0,1]=lin_reg(x, y)[3]
    axs[idx, idy].set_title('LWS ('+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Summed up ENS [GWh]')
    axs[idx, idy].set_xlabel('Duration [d]')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 1, 0
    x=rl3_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=get_ENS_sums(data3_ENS_d, rl3_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[1], label='RL (PECD 3.1)', alpha=0.5)
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[1], label='Lin Reg (PECD3.1)')
    #print(zones_szon[c]+' RL  (3.1) Dur R value: '+str(lin_reg(x, y)[3]))
    test[c,1,1,1]=lin_reg(x, y)[3]
    axs[idx, idy].set_title('RL ('+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Summed up ENS [GWh]')
    axs[idx, idy].set_xlabel('Duration [d]')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 0, 1
    x=dd3_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=get_ENS_sums(data3_ENS_d, dd3_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[2], label='DD (PECD 3.1)', alpha=0.5)
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[2], label='Lin Reg (PECD3.1)')
    #print(zones_szon[c]+' DD  (3.1) Dur R value: '+str(lin_reg(x, y)[3]))
    test[c,1,2,1]=lin_reg(x, y)[3]
    axs[idx, idy].set_title('DD ('+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Summed up ENS [GWh]')
    axs[idx, idy].set_xlabel('Duration [d]')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 1, 1
    x=cfd3_events.loc[('HIST',zones_szon[c]),('Duration')]/24
    y=get_ENS_sums(data3_ENS_h, cfd3_events.loc[('HIST')],zones_szon[c], 24)
    axs[idx, idy].scatter(x, y, color=dt_colors[3], label='CFD (PECD 3.1)', alpha=0.5)
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[3], label='Lin Reg (PECD3.1)')
    test[c,1,3,1]=lin_reg(x, y)[3]
    #print(zones_szon[c]+' CFD (3.1) Dur R value: '+str(lin_reg(x, y)[3]))
    axs[idx, idy].set_title('CFD ('+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Summed up ENS [GWh]')
    axs[idx, idy].set_xlabel('Duration [d]')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    plt.tight_layout()
    #plt.savefig(path_to_plot+'Correlation/DF_CorrelationENSDuration_'+ens_dataset+'_'+zones_szon[c]+'.'+plot_format,dpi=300)
    #plt.close()
    
# %%
