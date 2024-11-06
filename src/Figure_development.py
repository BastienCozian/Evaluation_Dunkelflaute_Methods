"""
This file contains pieces of code used to generate figures.

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
import pickle

from Dunkelflaute_function_library import get_zones, get_thresholds, detect_drought_Otero22, mask_data, get_f_score, get_df_timerange, lin_reg, detect_drought_Li21, mask_df_by_entries
from CREDIfunctions import Modified_Ordinal_Hour, Climatology_Hourly, Climatology_Hourly_Rolling, \
    Climatology_Hourly_Weekly_Rolling, get_CREDI_events, get_f_score_CREDI, get_f_score_CREDI_new, compute_timeline, get_correlation_CREDI

# Set parameters
path_to_data      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'
path_to_plot      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'
#path_to_data      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'  #'D:/Dunkelflaute/Data/'
#path_to_plot      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'      #'D:/Dunkelflaute/'

plot_format       = 'png'

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

scen_Datespan = [42,51,51,51] # TODO: Automate the nr of years per scenario


# =================================================================
# Load hourly data
# =================================================================

data3_dem_h = pd.read_pickle(path_to_data+'PEMMDB_demand_TY'+str(ty_pecd3)+'_hourly.pkl')
data4_dem_h = pd.read_pickle(path_to_data+'ETM_demand_TY'+str(ty_pecd4)+'_hourly.pkl')

data4_REP_h = pd.read_pickle(path_to_data+'PECD4_Generation_TY'+str(ty_pecd4)+'_national_hourly.pkl')
data3_REP_h = pd.read_pickle(path_to_data+'PECD3_Generation_TY'+str(ty_pecd3)+'_national_hourly.pkl')

data3_CF_h = pd.read_pickle(path_to_data+'PECD3_CF_TY'+str(ty_pecd3)+'_national_hourly.pkl')

# Weight REP for experimenting
start_date = '1982-01-01 00:00:00'
end_date   = '2016-12-31 23:00:00'
data3_cropped1 = data3_REP_h.query('Date>=@start_date and Date <= @end_date')
data4_cropped1 = data4_REP_h.query('Date>=@start_date and Date <= @end_date')
data3_CF_crop = data3_CF_h.query('Date>=@start_date and Date <= @end_date')
data3_gen_h = data3_cropped1[~((data3_cropped1.index.get_level_values(1).day == 29) & (data3_cropped1.index.get_level_values(1).month == 2))]
data4_gen_h = data4_cropped1[~((data4_cropped1.index.get_level_values(1).day == 29) & (data4_cropped1.index.get_level_values(1).month == 2))]
data3_CF_h  = data3_CF_crop[~((data3_CF_crop.index.get_level_values(2).day == 29) & (data3_CF_crop.index.get_level_values(2).month == 2))]

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





"""
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

"""



















#%%
# =================================================================
# Plot F-score for PECD3 | Old ENS dataset
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset

ens_dataset = 'ERAA23_old'

figname = f"Validation_Stoop24_{ens_dataset}_ENS_{agg_zone}_T{PERIOD_length_days}"


# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24


# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_gen_h = pd.DataFrame()
df_agg_dem_h = pd.DataFrame()
df_agg_RL_h  = pd.DataFrame()
df_agg_old_ENS_d = pd.DataFrame()

df_agg_gen_h[agg_zone] = data3_gen_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_h[agg_zone] = data3_dem_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_RL_h[agg_zone]  = data3_RL_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_old_ENS_d[agg_zone] = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[zones_list].sum(axis=1)

# --- end of aggregation ----


ens_mask = mask_data(df_agg_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

common_index = df_agg_RL_h.index.intersection(ens_mask.index)

# Get CREDI events
lws3_CREDI_event, lws3_event_dates, \
    lws3_event_values = get_CREDI_events(df_agg_gen_h, agg_zone, extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                         PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
dd3_CREDI_event, dd3_event_dates, \
    dd3_event_values = get_CREDI_events(df_agg_dem_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

p_list  = []
for p in range(len(LWS_percs)):
    # find capacity thresholds
    # Percentile is computed on the clustered events ("independent" events)
    # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
    lws3_thresh = np.quantile(lws3_event_values, q=LWS_percs[p], interpolation="nearest")
    rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")
    dd3_thresh = np.quantile(dd3_event_values, q=DD_percs[p], interpolation="nearest")

    # Calculate F (compared to ENS)
    lws3_stat = get_f_score_CREDI_new(df_agg_old_ENS_d, lws3_event_dates, lws3_event_values, lws3_thresh, common_index, agg_zone, 
                                      PERIOD_length_days=1, extreme_is_high=False, beta=1)
    rl3_stat = get_f_score_CREDI_new(df_agg_old_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, agg_zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)
    dd3_stat = get_f_score_CREDI_new(df_agg_old_ENS_d, dd3_event_dates, dd3_event_values, dd3_thresh, common_index, agg_zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)

    """
    # Mask the data / Detect Drought days
    lws3_mask  = mask_data(lws3_CREDI_event[[zone]], lws3_thresh, True, 1, 0)
    rl3_mask  = mask_data(rl3_CREDI_event[[zone]], rl3_thresh, False, 1, 0)
    dd3_mask  = mask_data(dd3_CREDI_event[[zone]], dd3_thresh, False, 1, 0)

    # Calculate F (compared to ENS)
    lws3_stat  = get_f_score_CREDI(ens_mask, lws3_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    rl3_stat  = get_f_score_CREDI(ens_mask, rl3_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    dd3_stat  = get_f_score_CREDI(ens_mask, dd3_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    """


    # Create Dataframe
    p_list.append( pd.concat([lws3_stat, rl3_stat, dd3_stat], 
                             keys=['LWS3', 'RL3', 'DD3'],   names=['Drought type']))
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
fig.suptitle(f'Stoop Method for {agg_zone} using {ens_dataset} ENS (T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
ax_big.plot(x, stat_df.loc[(x, 'RL3',  'F'),(agg_zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'DD3',  'F'),(agg_zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'LWS3',  'F'),(agg_zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TP'),(agg_zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TP'),(agg_zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'TP'),(agg_zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[agg_zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TN'),(agg_zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TN'),(agg_zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'TN'),(agg_zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[agg_zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FP'),(agg_zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FP'),(agg_zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'FP'),(agg_zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[agg_zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FN'),(agg_zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FN'),(agg_zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'FN'),(agg_zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[agg_zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'PR'),(agg_zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'PR'),(agg_zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'PR'),(agg_zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[agg_zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'RE'),(agg_zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'RE'),(agg_zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'RE'),(agg_zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('Recall (out of '+str(nr_of_pos[agg_zone])+' in total)\n(How many ENS are identified as droughts?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.{plot_format}", dpi=300)
#plt.close()

print(f"Saved {path_to_plot}Validation/{figname}.{plot_format}")





















#%%
# =================================================================
# Plot F-score | Compare old and new dataset
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'ES00'
agg_zone = zone; zones_list = [zone]

#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']

ens_dataset = 'ERAA23'

figname = f"Validation_Stoop24_all_new_ENS_{agg_zone}_T{PERIOD_length_days}"


# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24



# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_gen_h = pd.DataFrame()
df_agg_dem_h = pd.DataFrame()
df_agg_RL_h  = pd.DataFrame()
df_agg_old_ENS_d = pd.DataFrame()

df_agg_gen_h[agg_zone] = data3_gen_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_h[agg_zone] = data3_dem_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_RL_h[agg_zone]  = data3_RL_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_old_ENS_d[agg_zone] = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[zones_list].sum(axis=1)

# --- end of aggregation ----

# TODO: Dirty -> I used an old piece of code, I should update that:
ens_mask = mask_data(df_agg_old_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

common_index = df_agg_RL_h.index.intersection(ens_mask.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

stat_df = dict()
for scenario_EVA in ['A', 'B']:
    stat_df[scenario_EVA] = dict()
    for FOS in range(1, 15+1):


        df_agg_ENS_fos_d = pd.DataFrame()
        df_agg_ENS_fos_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
        
        p_list  = []
        for p in range(len(LWS_percs)):
            # find capacity thresholds
            # Percentile is computed on the clustered events ("independent" events)
            # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
            rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")

            # Calculate F (compared to ENS)
            rl3_stat = get_f_score_CREDI_new(df_agg_ENS_fos_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, agg_zone, 
                                            PERIOD_length_days=1, extreme_is_high=True, beta=1)

            # Create Dataframe
            p_list.append( pd.concat([rl3_stat], keys=['RL'], names=['Drought type']))
            print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
        stat_df[scenario_EVA][FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])

# --- Old ENS dataset ---
p_list  = []
for p in range(len(LWS_percs)):
    # find capacity thresholds
    # Percentile is computed on the clustered events ("independent" events)
    # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
    rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")

    # Calculate F (compared to ENS)
    rl3_stat = get_f_score_CREDI_new(df_agg_old_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, agg_zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)
    # Create Dataframe
    p_list.append( pd.concat([rl3_stat], 
                             keys=['RL'],   names=['Drought type']))
    print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
stat_df_old  = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
pickle.dump(stat_df, open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "wb"))


#%%
# ---------------------------------------------
#  Plot F / threshold (comparing to ENS / validation)
# ---------------------------------------------

# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df['A'][1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for scenario_EVA in ['A', 'B']:
    quantiles_dict[scenario_EVA] = dict()
    min_dict[scenario_EVA] = dict()
    max_dict[scenario_EVA] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN', 'PR', 'RE']:
        quantiles_dict[scenario_EVA][metric] = np.quantile([stat_df[scenario_EVA][FOS].loc[(x, 'RL',  metric),(agg_zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[scenario_EVA][metric] = np.min([stat_df[scenario_EVA][FOS].loc[(x, 'RL',  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[scenario_EVA][metric] = np.max([stat_df[scenario_EVA][FOS].loc[(x, 'RL',  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)


fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle(f'Stoop Method for {agg_zone} using new ENS dataset (T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
metric = 'F'
ax_big.plot(x, stat_df_old.loc[(x, 'RL',  metric),(agg_zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
ax_big.fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
ax_big.plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
ax_big.plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
ax_big.plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
ax_big.fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
ax_big.plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
ax_big.plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
metric = 'TP'
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL',  metric),(agg_zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[agg_zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
metric = 'TN'
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL',  metric),(agg_zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[agg_zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
metric = 'FP'
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL',  metric),(agg_zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[agg_zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
metric = 'FN'
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL',  metric),(agg_zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[agg_zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
metric = 'PR'
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL',  metric),(agg_zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[agg_zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
metric = 'RE'
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL',  metric),(agg_zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('Recall (out of '+str(nr_of_pos[agg_zone])+' in total)\n(How many ENS are identified as droughts?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.{plot_format}", dpi=300)
#plt.close()

print(f"Saved {path_to_plot}Validation/{figname}.{plot_format}")





















#%%
# ==================================================================================
# Plot F-score | Stoop 23 (event-based method) | Compare RL, DD, RES for scenario B
# ==================================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'
agg_zone = zone; zones_list = [zone]

#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']

scenario_EVA = 'B'

figname = f"Validation_Stoop24_ENS_scenario{scenario_EVA}_{agg_zone}_T{PERIOD_length_days}"


# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24



# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_gen_h = pd.DataFrame()
df_agg_dem_h = pd.DataFrame()
df_agg_RL_h  = pd.DataFrame()
df_agg_old_ENS_d = pd.DataFrame()

df_agg_gen_h[agg_zone] = data3_gen_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_h[agg_zone] = data3_dem_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_RL_h[agg_zone]  = data3_RL_h.loc[('HIST')][zones_list].sum(axis=1)

# --- end of aggregation ----
df_ENS_example = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{'B'}_FOS{1}_daily.pkl')[zones_list[0]]
common_index = df_agg_RL_h.index.intersection(df_ENS_example.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
# Get CREDI events
lws3_CREDI_event, lws3_event_dates, \
    lws3_event_values = get_CREDI_events(df_agg_gen_h, agg_zone, extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                         PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
dd3_CREDI_event, dd3_event_dates, \
    dd3_event_values = get_CREDI_events(df_agg_dem_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

stat_df = dict()
for FOS in range(1, 15+1):

    df_agg_ENS_fos_d = pd.DataFrame()
    df_agg_ENS_fos_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    
    p_list  = []
    for p in range(len(LWS_percs)):
        # find capacity thresholds
        # Percentile is computed on the clustered events ("independent" events)
        # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
        rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")
        lws3_thresh = np.quantile(lws3_event_values, q=LWS_percs[p], interpolation="nearest")
        dd3_thresh = np.quantile(dd3_event_values, q=DD_percs[p], interpolation="nearest")

        # Calculate F (compared to ENS)
        rl3_stat = get_f_score_CREDI_new(df_agg_ENS_fos_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, agg_zone, 
                                         PERIOD_length_days=1, extreme_is_high=True, beta=1)
        lws3_stat = get_f_score_CREDI_new(df_agg_ENS_fos_d, lws3_event_dates, lws3_event_values, lws3_thresh, common_index, agg_zone,
                                          PERIOD_length_days=1, extreme_is_high=False, beta=1)
        dd3_stat = get_f_score_CREDI_new(df_agg_ENS_fos_d, dd3_event_dates, dd3_event_values, dd3_thresh, common_index, agg_zone, 
                                         PERIOD_length_days=1, extreme_is_high=True, beta=1)

        # Create Dataframe
        p_list.append( pd.concat([rl3_stat, lws3_stat, dd3_stat], keys=['RL', 'LWS', 'DD'],   names=['Drought type']))
        print('Done '+str(p+1)+'/'+str(len(LWS_percs)))

    stat_df[FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
    print(f'    Done FOS {FOS}')
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
pickle.dump(stat_df, open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "wb"))


##%%
# ---------------------------------------------
#  Plot F / threshold (comparing to ENS / validation)
# ---------------------------------------------

# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL', 'DD', 'LWS']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN', 'PR', 'RE']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)

# Percentile for peak F-score
ener_var = 'RL'
metric = 'F'
F_max = quantiles_dict[ener_var][metric][1].max()
p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
print(f"RL3 (T=1): F_max = {F_max}, p_max = {p_max}")

fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle(f'Stoop Method for {agg_zone} using new ENS dataset (T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
metric = 'F'
for ncolor, ener_var in enumerate(['RL', 'DD', 'LWS']):
    ax_big.plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    ax_big.fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    ax_big.plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    ax_big.plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
metric = 'TP'
for ncolor, ener_var in enumerate(['RL', 'DD', 'LWS']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
metric = 'TN'
for ncolor, ener_var in enumerate(['RL', 'DD', 'LWS']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
metric = 'FP'
for ncolor, ener_var in enumerate(['RL', 'DD', 'LWS']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
metric = 'FN'
for ncolor, ener_var in enumerate(['RL', 'DD', 'LWS']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
metric = 'PR'
for ncolor, ener_var in enumerate(['RL', 'DD', 'LWS']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
metric = 'RE'
for ncolor, ener_var in enumerate(['RL', 'DD', 'LWS']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('Recall (How many ENS are identified as droughts?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.{plot_format}", dpi=300)
#plt.close()

print(f"Saved {path_to_plot}Validation/{figname}.{plot_format}")


#%%
# ---------------------------------------------
#  Plot only F
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'
agg_zone = zone; zones_list = [zone]

#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']

scenario_EVA = 'B'

figname = f"Validation_Stoop24_ENS_scenario{scenario_EVA}_{agg_zone}_T{PERIOD_length_days}"

# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL', 'DD', 'LWS']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN', 'PR', 'RE']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)

# Percentile for peak F-score
ener_var = 'RL'
metric = 'F'
F_max = quantiles_dict[ener_var][metric][1].max()
p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
print(f"RL3 (T=1): F_max = {F_max}, p_max = {p_max}")

fig, axs = plt.subplots(1, 1, figsize=(5, 4))
axs.set_title(f'Method Stoop 2023 for {agg_zone} (T={PERIOD_length_days}d)')

# Event time series

metric = 'F'
for ncolor, ener_var in enumerate(['RL', 'DD', 'LWS']):
    axs.plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs.fill_between(x, min_dict[ener_var][metric], max_dict[ener_var][metric], color=dt_colors[ncolor], alpha=0.5)
    #axs.plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    #axs.plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs.set_ylabel('F-Score')
axs.set_xlabel("Percentile of top CREDI events")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs.legend(facecolor="white", loc='upper right', framealpha=1)
axs.grid(axis='y')


plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.pdf", dpi=300)
#plt.close()












































#%% 
# =================================================================
# Plot Timeline
# =================================================================
# Plot Timelines of DF Detection and ENS
# only HIST (obviously)
# whithout February 29th
# Grid plot: X=DoY, Y=Year, Colors: White=TN, Blue=FP, Red=FN, Green=TP
# For each droughttype individual panels

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 3
PERIOD_cluster_days = 3

#ens_dataset = 'ERAA23_old'; scenario_EVA = 'none' ; FOS = 0
ens_dataset = 'ERAA23'; scenario_EVA = 'B' ; FOS = 2

# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'
agg_zone = zone; zones_list = [zone]

#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']


# --- Percentile for peak F-score. Computed in `DF_Validation_Stoop.py`
# FOR DE00
#p_max = 0.01   # TODO: check with new data
#p_max = 0.022  # TODO: check with new data
#p_max = 0.0356 # TODO: check with new data
#p_max = 0.044  # TODO: check with new data
# New FOR DE00
#p_max = 0.014 # T=1
p_max = 0.0384 # T=3d
#p_max = 0.048 # T=5
#p_max = 0.088 # T=7

# FOR FR00
#p_max = 0.0044
#p_max = 0.026  # TODO: check with new data
#p_max = 0.0276 # TODO: check with new data
#p_max = 0.0204 # TODO: check with new data

# FOR CoreRegion
#p_max = 0.274 # T=1

# FOR CWE
#p_max = 0.014 # T=1

# FOR NO
#p_max = 0.0036 # T=1

# FOR IT
#p_max = 0.0284 # T=1

# For PL00
#p_max = 0.0084 # T=1




figname = f"Timeline_Stoop24_{ens_dataset}_Scenario{scenario_EVA}_FOS{FOS}_{agg_zone}_T{PERIOD_length_days}_Tc{PERIOD_cluster_days}_pmax_{int(p_max*1000)}e-3"

# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_RL_h  = pd.DataFrame()
df_agg_old_ENS_d = pd.DataFrame()

df_agg_RL_h[agg_zone]  = data3_RL_h.loc[('HIST')][zones_list].sum(axis=1)
df_agg_old_ENS_d[agg_zone] = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[zones_list].sum(axis=1)
# --- end of aggregation ----


# Create colormap
colors = ['whitesmoke', 'royalblue', 'red', 'limegreen']
cmap=ListedColormap(colors)
legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0],label='No DF, No ENS'),
                   Patch(facecolor=colors[1], edgecolor=colors[1],label='DF detected, but no ENS'),
                   Patch(facecolor=colors[2], edgecolor=colors[2],label='No DF, but ENS'),
                   Patch(facecolor=colors[3], edgecolor=colors[3],label='DF detected and ENS')]

df_agg_ENS_d = pd.DataFrame()
if ens_dataset=='AO':
    df_agg_ENS_d[agg_zone] = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_daily.pkl')[zones_list].sum(axis=1)
    X1= np.arange(364)
    X2= np.arange(364*24)
elif ens_dataset=='ERAA23_old':
    df_agg_ENS_d[agg_zone] = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[zones_list].sum(axis=1)
    X1= np.arange(365)
    X2= np.arange(365*24)
elif ens_dataset=='ERAA23':
    df_agg_ENS_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    X1= np.arange(365)
    X2= np.arange(365*24)
else:
    raise KeyError('ENS Dataset not existent!')
Y = np.arange(1982,2016+1) # common period of ENS & PECD

# Generate masked data
# TODO: Dirty -> I used an old piece of code, I should update that:
ens_mask = mask_data(df_agg_old_ENS_d, 0, False, 2, 0)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

rl3_thresh = np.quantile(rl3_event_values, q=1-p_max, interpolation="nearest")

df_mask_timeline = compute_timeline(df_agg_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, agg_zone, PERIOD_length_days, 
                                    extreme_is_high=True, start_date='1982-01-01', end_date='2016-12-31')


# Tranform detection masks into the right format
def detmask2matrix(detmask,X,Y):
    return np.reshape(detmask, (len(Y),len(X)))

fig, ax = plt.subplots(1, 1, figsize=(10, 4))

#axs.pcolormesh(X1, Y, detmask2matrix(rl3_detmask[zones_szon[c]],X1,Y), cmap=cmap)
ax.pcolormesh(X1, Y, detmask2matrix(df_mask_timeline[agg_zone],X1,Y), cmap=cmap)
ax.set_title(f'RL (PECD 3.1, {ens_dataset}, Scenario EVA {scenario_EVA}, FOS {FOS}, {agg_zone}, T={PERIOD_length_days}, Tc={PERIOD_cluster_days}, p={p_max})')
ax.set_ylabel('Year')
ax.set_xlabel('Day of Year')
ax.legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)


plt.tight_layout()
#plt.savefig(f"{path_to_plot}Validation/{figname}.{plot_format}", dpi=300)
print(f"Saved {path_to_plot}Validation/{figname}.{plot_format}")
#plt.close()
        
# TODO: Some masks are shorter? Why? In year with leap day the 31.12. is missing (and the leap day...)










