"""
This scripts does the validating of Method 3 (inspired by Stoop et al. 2024)
For several thresholds energy droughts are detected and evaluated using ENS data.
Results of the Evaluation are saved as pickles.
Plots are also created to visualize the "best-fittting" threshold.

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
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from datetime import timedelta
import datetime as dtime
import time
import pickle

from Dunkelflaute_function_library import get_zones, get_thresholds, detect_drought_Otero22, mask_data, get_f_score, get_df_timerange
from CREDIfunctions import Modified_Ordinal_Hour, Climatology_Hourly, Climatology_Hourly_Rolling, \
    Climatology_Hourly_Weekly_Rolling, get_CREDI_events, get_f_score_CREDI, get_f_score_CREDI_new, compute_timeline

# Set parameters
path_to_data      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'
path_to_plot      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'
#path_to_data      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'  #'D:/Dunkelflaute/Data/'
#path_to_plot      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'      #'D:/Dunkelflaute/'

plot_format       = 'png'

# Countries (NUT0) of interest (must be a list of two letter abbreviations)
#countries = ['DE','NL','FR']

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

#zones_peon = zones = get_zones(countries,'PEON')  
#zones_szon = zones = get_zones(countries,'SZON')  
#zones_peof = zones = get_zones(countries,'PEOF')  

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
# -> Solved: when loading the hourly demand data, the line `end_date   = '2016-12-31 00:00:00'` selected only the first hour of Jan 31st 2016.

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
# Plot F-score for PECD3
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

zone = 'NOS1'

ens_dataset = 'ERAA23'

figname = f"Validation_Stoop24_{ens_dataset}_ENS_{zone}_T{PERIOD_length_days}"


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
    data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')
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
    lws3_event_values = get_CREDI_events(data3_gen_h.loc[('HIST')], zone, extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                         PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
dd3_CREDI_event, dd3_event_dates, \
    dd3_event_values = get_CREDI_events(data3_dem_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
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
    lws3_stat = get_f_score_CREDI_new(data3_ENS_d, lws3_event_dates, lws3_event_values, lws3_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=False, beta=1)
    rl3_stat = get_f_score_CREDI_new(data3_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)
    dd3_stat = get_f_score_CREDI_new(data3_ENS_d, dd3_event_dates, dd3_event_values, dd3_thresh, common_index, zone, 
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
fig.suptitle(f'Stoop Method for {zone} using {ens_dataset} ENS (T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
ax_big.plot(x, stat_df.loc[(x, 'RL3',  'F'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'DD3',  'F'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'LWS3',  'F'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TP'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TP'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'TP'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TN'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TN'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'TN'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FP'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FP'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'FP'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FN'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FN'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'FN'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'PR'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'PR'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'PR'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'RE'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'RE'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'RE'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
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



















 #%%
# =================================================================
# Plot F-score for PECD3 and PECD4
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

zone = 'DE00'

ens_dataset = 'ERAA23'

figname = f"Validation_Stoop24_{ens_dataset}_ENS_{zone}_T{PERIOD_length_days}_compare_PECD3_PECD4"


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
    lws3_event_values = get_CREDI_events(data3_gen_h.loc[('HIST')], zone, extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                         PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
lws4_CREDI_event, lws4_event_dates, \
    lws4_event_values = get_CREDI_events(data4_gen_h.loc[('HIST')], zone, extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                         PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
rl4_CREDI_event, rl4_event_dates, \
    rl4_event_values = get_CREDI_events(data4_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days, 
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
dd3_CREDI_event, dd3_event_dates, \
    dd3_event_values = get_CREDI_events(data3_dem_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
dd4_CREDI_event, dd4_event_dates, \
    dd4_event_values = get_CREDI_events(data4_dem_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days, 
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

p_list  = []
for p in range(len(LWS_percs)):
    # find capacity thresholds
    # Percentile is computed on the clustered events ("independent" events)
    # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
    lws3_thresh = np.quantile(lws3_event_values, q=LWS_percs[p], interpolation="nearest")
    lws4_thresh = np.quantile(lws4_event_values, q=LWS_percs[p], interpolation="nearest")
    rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")
    rl4_thresh = np.quantile(rl4_event_values, q=RL_percs[p], interpolation="nearest")
    dd3_thresh = np.quantile(dd3_event_values, q=DD_percs[p], interpolation="nearest")
    dd4_thresh = np.quantile(dd4_event_values, q=DD_percs[p], interpolation="nearest")

    # Calculate F (compared to ENS)
    lws3_stat = get_f_score_CREDI_new(data3_ENS_d, lws3_event_dates, lws3_event_values, lws3_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=False, beta=1)
    lws4_stat = get_f_score_CREDI_new(data3_ENS_d, lws4_event_dates, lws4_event_values, lws4_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=False, beta=1)
    rl3_stat = get_f_score_CREDI_new(data3_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)
    rl4_stat = get_f_score_CREDI_new(data3_ENS_d, rl4_event_dates, rl4_event_values, rl4_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)
    dd3_stat = get_f_score_CREDI_new(data3_ENS_d, dd3_event_dates, dd3_event_values, dd3_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)
    dd4_stat = get_f_score_CREDI_new(data3_ENS_d, dd4_event_dates, dd4_event_values, dd4_thresh, common_index, zone, 
                                      PERIOD_length_days=1, extreme_is_high=True, beta=1)
    

    """
    # Mask the data / Detect Drought days
    lws3_mask  = mask_data(lws3_CREDI_event[[zone]], lws3_thresh, True, 1, 0)
    lws4_mask  = mask_data(lws4_CREDI_event[[zone]], lws4_thresh, True, 1, 0)
    rl3_mask  = mask_data(rl3_CREDI_event[[zone]], rl3_thresh, False, 1, 0)
    rl4_mask  = mask_data(rl4_CREDI_event[[zone]], rl4_thresh, False, 1, 0)
    dd3_mask  = mask_data(dd3_CREDI_event[[zone]], dd3_thresh, False, 1, 0)
    dd4_mask  = mask_data(dd4_CREDI_event[[zone]], dd4_thresh, False, 1, 0)

    # Calculate F (compared to ENS)
    lws3_stat  = get_f_score_CREDI(ens_mask, lws3_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    lws4_stat  = get_f_score_CREDI(ens_mask, lws4_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    rl3_stat  = get_f_score_CREDI(ens_mask, rl3_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    rl4_stat  = get_f_score_CREDI(ens_mask, rl4_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    dd3_stat  = get_f_score_CREDI(ens_mask, dd3_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    dd4_stat  = get_f_score_CREDI(ens_mask, dd4_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    """

    # Create Dataframe
    p_list.append( pd.concat([lws3_stat, lws4_stat, rl3_stat, rl4_stat, dd3_stat, dd4_stat], 
                             keys=['LWS3', 'LWS4', 'RL3', 'RL4', 'DD3', 'DD4'],   names=['Drought type']))
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
fig.suptitle(f'Stoop Method for {zone} using {ens_dataset} ENS (T={PERIOD_length_days}d)')


# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
ax_big.plot(x, stat_df.loc[(x, 'RL3',  'F'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL4',  'F'),(zone)], label='RL4',  color=dt_colors[3], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'DD3',  'F'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'DD4',  'F'),(zone)], label='DD4',  color=dt_colors[4], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'LWS3',  'F'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'LWS4',  'F'),(zone)], label='LWS4',  color=dt_colors[5], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TP'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'TP'),(zone)], label='RL4',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TP'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'TP'),(zone)], label='DD4',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'TP'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'TP'),(zone)], label='LWS4',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TN'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'TN'),(zone)], label='RL4',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TN'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'TN'),(zone)], label='DD4',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'TN'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'TN'),(zone)], label='LWS4',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FP'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'FP'),(zone)], label='RL4',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FP'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'FP'),(zone)], label='DD4',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'FP'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'FP'),(zone)], label='LWS4',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FN'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'FN'),(zone)], label='RL4',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FN'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'FN'),(zone)], label='DD4',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'FN'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'FN'),(zone)], label='LWS4',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'PR'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'PR'),(zone)], label='RL4',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'PR'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'PR'),(zone)], label='DD4',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'PR'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'PR'),(zone)], label='LWS4',  color=dt_colors[5], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'RE'),(zone)], label='RL3',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'RE'),(zone)], label='RL4',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'RE'),(zone)], label='DD3',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'RE'),(zone)], label='DD4',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'RE'),(zone)], label='LWS3',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'RE'),(zone)], label='LWS4',  color=dt_colors[5], alpha=0.8)
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




















 #%%
# =================================================================
# Plot F-score for HRW and (current) HWRW
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

zone = 'DE00'

ens_dataset = 'ERAA23'

figname = f"Validation_Stoop24_{ens_dataset}_ENS_{zone}_T{PERIOD_length_days}_compare_HRW_HWRW"


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
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31',
                                        climatology='HRW')

rl3_HWRW_CREDI_event, rl3_HWRW_event_dates, \
    rl3_HWRW_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                             PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31',
                                             climatology='HWRW')



p_list  = []
for p in range(len(LWS_percs)):
    # find capacity thresholds
    # Percentile is computed on the clustered events ("independent" events)
    # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
    rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")
    rl3_HWRW_thresh = np.quantile(rl3_HWRW_event_values, q=RL_percs[p], interpolation="nearest")

    rl3_stat      = get_f_score_CREDI_new(data3_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                          PERIOD_length_days=1, extreme_is_high=True, beta=1)
    rl3_HWRW_stat = get_f_score_CREDI_new(data3_ENS_d, rl3_HWRW_event_dates, rl3_HWRW_event_values, rl3_HWRW_thresh, common_index, zone, 
                                          PERIOD_length_days=1, extreme_is_high=True, beta=1)
    
    """
    # Mask the data / Detect Drought days
    rl3_mask = mask_data(rl3_CREDI_event[[zone]], rl3_thresh, False, 1, 0)
    rl3_HWRW_mask = mask_data(rl3_HWRW_CREDI_event[[zone]], rl3_HWRW_thresh, False, 1, 0)

    # Calculate F (compared to ENS)
    rl3_stat = get_f_score_CREDI(ens_mask, rl3_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    rl3_HWRW_stat = get_f_score_CREDI(ens_mask, rl3_HWRW_mask, zone, PERIOD_length_days, PERIOD_cluster_days, beta=1)
    """

    # Create Dataframe
    p_list.append( pd.concat([rl3_stat, rl3_HWRW_stat], 
                             keys=['RL3 HRW', 'RL3 HWRW'],   names=['Drought type']))
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
fig.suptitle(f'Stoop Method for {zone} using {ens_dataset} ENS (T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
ax_big.plot(x, stat_df.loc[(x, 'RL3 HRW',  'F'),(zone)], label='RL3 HRW',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL3 HWRW',  'F'),(zone)], label='RL3 HWRW',  color=dt_colors[3], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HRW',  'TP'),(zone)], label='RL3 HRW',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HWRW',  'TP'),(zone)], label='RL3 HWRW',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HRW',  'TN'),(zone)], label='RL3 HRW',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HWRW',  'TN'),(zone)], label='RL3 HWRW',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HRW',  'FP'),(zone)], label='RL3 HRW',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HWRW',  'FP'),(zone)], label='RL3 HWRW',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HRW',  'FN'),(zone)], label='RL3 HRW',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HWRW',  'FN'),(zone)], label='RL3 HWRW',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HRW',  'PR'),(zone)], label='RL3 HRW',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HWRW',  'PR'),(zone)], label='RL3 HWRW',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HRW',  'RE'),(zone)], label='RL3 HRW',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 HWRW',  'RE'),(zone)], label='RL3 HWRW',  color=dt_colors[3], alpha=0.8)
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






















 #%%
# =================================================================
# Plot F-score for different duration T
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

zone = 'DE00'

ens_dataset = 'ERAA23'

figname = f"Validation_Stoop24_{ens_dataset}_ENS_{zone}_compare_T"


# ---------------------------------------------
# Compute data for figure (~ 15s/percentile)
# ---------------------------------------------
start_time = time.time()

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
#PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

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

common_index = data3_RL_h.loc[('HIST')][[zone]].index.intersection(ens_mask.index)

PERIOD_length_days = 1
PERIOD_cluster_days = 1
# Get CREDI events
T1_CREDI_event, T1_event_dates, \
    T1_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 3
PERIOD_cluster_days = 3
# Get CREDI events
T3_CREDI_event, T3_event_dates, \
    T3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 5
PERIOD_cluster_days = 4
# Get CREDI events
T5_CREDI_event, T5_event_dates, \
    T5_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 7
PERIOD_cluster_days = 6
# Get CREDI events
T7_CREDI_event, T7_event_dates, \
    T7_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

N_event_T1 = []
N_event_T3 = []
N_event_T5 = []
N_event_T7 = []

p_list  = []
for p in range(len(LWS_percs)):
    # find capacity thresholds
    # Percentile is computed on the clustered events ("independent" events)
    # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
    T1_thresh = np.quantile(T1_event_values, q=RL_percs[p], interpolation="nearest")
    T3_thresh = np.quantile(T3_event_values, q=RL_percs[p], interpolation="nearest")
    T5_thresh = np.quantile(T5_event_values, q=RL_percs[p], interpolation="nearest")
    T7_thresh = np.quantile(T7_event_values, q=RL_percs[p], interpolation="nearest")

    # Calculate F (compared to ENS)
    T1_stat = get_f_score_CREDI_new(data3_ENS_d, T1_event_dates, T1_event_values, T1_thresh, common_index, zone, 
                                    PERIOD_length_days=1, extreme_is_high=True, beta=1)
    T3_stat = get_f_score_CREDI_new(data3_ENS_d, T3_event_dates, T3_event_values, T3_thresh, common_index, zone, 
                                    PERIOD_length_days=3, extreme_is_high=True, beta=1)
    T5_stat = get_f_score_CREDI_new(data3_ENS_d, T5_event_dates, T5_event_values, T5_thresh, common_index, zone, 
                                    PERIOD_length_days=5, extreme_is_high=True, beta=1)
    T7_stat = get_f_score_CREDI_new(data3_ENS_d, T7_event_dates, T7_event_values, T7_thresh, common_index, zone, 
                                    PERIOD_length_days=7, extreme_is_high=True, beta=1)

    """
    # Mask the data / Detect Drought days
    T1_mask = mask_data(T1_CREDI_event[[zone]], T1_thresh, False, 1, 0) # WARNING! No clustering was applied to T1_CREDI_event 
    T3_mask = mask_data(T3_CREDI_event[[zone]], T3_thresh, False, 1, 0)
    T5_mask = mask_data(T5_CREDI_event[[zone]], T5_thresh, False, 1, 0)
    T7_mask = mask_data(T7_CREDI_event[[zone]], T7_thresh, False, 1, 0)
    """
    # Count number of drought events (it changes with T)
    N_event_T1.append(np.sum(np.asarray(T1_event_values) > T1_thresh))
    N_event_T3.append(np.sum(np.asarray(T3_event_values) > T3_thresh))
    N_event_T5.append(np.sum(np.asarray(T5_event_values) > T5_thresh))
    N_event_T7.append(np.sum(np.asarray(T7_event_values) > T7_thresh))

    # Create Dataframe
    p_list.append( pd.concat([T1_stat, T3_stat, T5_stat, T7_stat], 
                             keys=['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)', 'RL3 (T=7)'],   names=['Drought type']))
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


# Percentile for peak F-score
F_max = stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)].max()
p_max = stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)] == F_max].index[0][0]
print(f"RL3 (T=1): F_max = {F_max}, p_max = {p_max}")
F_max = stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)].max()
p_max = stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)] == F_max].index[0][0]
print(f"RL3 (T=3): F_max = {F_max}, p_max = {p_max}")
F_max = stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)].max()
p_max = stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)] == F_max].index[0][0]
print(f"RL3 (T=5): F_max = {F_max}, p_max = {p_max}")
F_max = stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)].max()
p_max = stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)] == F_max].index[0][0]
print(f"RL3 (T=7): F_max = {F_max}, p_max = {p_max}")


fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle(f'Stoop Method for {zone} using {ens_dataset} ENS')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()


print(f"Number of events for T=1 at F-score peak: {N_event_T1[stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=3 at F-score peak: {N_event_T3[stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=5 at F-score peak: {N_event_T5[stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=7 at F-score peak: {N_event_T7[stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)].argmax()]}")

idx, idy = 1, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'TP'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'TP'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'TP'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'TP'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'TN'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'TN'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'TN'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'TN'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'FP'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'FP'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'FP'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'FP'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'FN'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'FN'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'FN'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'FN'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'PR'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'PR'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'PR'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'PR'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('Precision in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'RE'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'RE'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'RE'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'RE'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('Recall in total)\n(How many ENS are identified as droughts?)')
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
# Plot F-score for different duration T - x-axis = average CREDI 
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

zone = 'DE00'

ens_dataset = 'ERAA23'

figname = f"Validation_Stoop24_{ens_dataset}_ENS_{zone}_compare_T_xaxisCREDI"


# ---------------------------------------------
# Compute data for figure (~ 15s/percentile)
# ---------------------------------------------
start_time = time.time()

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
#PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

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

common_index = data3_RL_h.loc[('HIST')][[zone]].index.intersection(ens_mask.index)

PERIOD_length_days = 1
PERIOD_cluster_days = 1
# Get CREDI events
T1_CREDI_event, T1_event_dates, \
    T1_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 3
PERIOD_cluster_days = 3
# Get CREDI events
T3_CREDI_event, T3_event_dates, \
    T3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 5
PERIOD_cluster_days = 4
# Get CREDI events
T5_CREDI_event, T5_event_dates, \
    T5_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 7
PERIOD_cluster_days = 6
# Get CREDI events
T7_CREDI_event, T7_event_dates, \
    T7_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

N_event_T1 = []
N_event_T3 = []
N_event_T5 = []
N_event_T7 = []

T1_thresh_list = []
T3_thresh_list = []
T5_thresh_list = []
T7_thresh_list = []
p_list  = []
for p in range(len(LWS_percs)):
    # find capacity thresholds
    # Percentile is computed on the clustered events ("independent" events)
    # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
    T1_thresh = np.quantile(T1_event_values, q=RL_percs[p], interpolation="nearest")
    T3_thresh = np.quantile(T3_event_values, q=RL_percs[p], interpolation="nearest")
    T5_thresh = np.quantile(T5_event_values, q=RL_percs[p], interpolation="nearest")
    T7_thresh = np.quantile(T7_event_values, q=RL_percs[p], interpolation="nearest")

    # x-axis = average CREDI value.
    # divide CREDI by 24 hours and the number of days (chagne units MWh -> GW)
    T1_thresh_list.append(T1_thresh / (1000 * 24 * 1))
    T3_thresh_list.append(T3_thresh / (1000 * 24 * 3))
    T5_thresh_list.append(T5_thresh / (1000 * 24 * 5))
    T7_thresh_list.append(T7_thresh / (1000 * 24 * 7))

    # Calculate F (compared to ENS)
    T1_stat = get_f_score_CREDI_new(data3_ENS_d, T1_event_dates, T1_event_values, T1_thresh, common_index, zone, 
                                    PERIOD_length_days=1, extreme_is_high=True, beta=1)
    T3_stat = get_f_score_CREDI_new(data3_ENS_d, T3_event_dates, T3_event_values, T3_thresh, common_index, zone, 
                                    PERIOD_length_days=3, extreme_is_high=True, beta=1)
    T5_stat = get_f_score_CREDI_new(data3_ENS_d, T5_event_dates, T5_event_values, T5_thresh, common_index, zone, 
                                    PERIOD_length_days=5, extreme_is_high=True, beta=1)
    T7_stat = get_f_score_CREDI_new(data3_ENS_d, T7_event_dates, T7_event_values, T7_thresh, common_index, zone, 
                                    PERIOD_length_days=7, extreme_is_high=True, beta=1)

    """
    # Mask the data / Detect Drought days
    T1_mask = mask_data(T1_CREDI_event[[zone]], T1_thresh, False, 1, 0) # WARNING! No clustering was applied to T1_CREDI_event 
    T3_mask = mask_data(T3_CREDI_event[[zone]], T3_thresh, False, 1, 0)
    T5_mask = mask_data(T5_CREDI_event[[zone]], T5_thresh, False, 1, 0)
    T7_mask = mask_data(T7_CREDI_event[[zone]], T7_thresh, False, 1, 0)

    # Count number of drought events (it changes with T)
    N_event_T1.append((T1_mask==1).sum())
    N_event_T3.append((T3_mask==1).sum())
    N_event_T5.append((T5_mask==1).sum())
    N_event_T7.append((T7_mask==1).sum())
    """

    # Create Dataframe
    p_list.append( pd.concat([T1_stat, T3_stat, T5_stat, T7_stat], 
                             keys=['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)', 'RL3 (T=7)'],   names=['Drought type']))
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


# Percentile for peak F-score
F_max = stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)].max()
p_max = stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)] == F_max].index[0][0]
print(f"RL3 (T=1): F_max = {F_max}, p_max = {p_max}")
F_max = stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)].max()
p_max = stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)] == F_max].index[0][0]
print(f"RL3 (T=3): F_max = {F_max}, p_max = {p_max}")
F_max = stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)].max()
p_max = stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)] == F_max].index[0][0]
print(f"RL3 (T=5): F_max = {F_max}, p_max = {p_max}")
F_max = stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)].max()
p_max = stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)] == F_max].index[0][0]
print(f"RL3 (T=7): F_max = {F_max}, p_max = {p_max}")


fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle('Stoop Method for '+zone+' using '+ens_dataset+' ENS')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
ax_big.plot(T1_thresh_list, stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
ax_big.plot(T3_thresh_list, stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
ax_big.plot(T5_thresh_list, stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
ax_big.plot(T7_thresh_list, stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Threshold value (average CREDI value [GWh/h])")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

"""
print(f"Number of events for T=1 at F-score peak: {N_event_T1[stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=3 at F-score peak: {N_event_T3[stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=5 at F-score peak: {N_event_T5[stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=7 at F-score peak: {N_event_T7[stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)].argmax()]}")
"""

idx, idy = 1, 0
axs[idx, idy].plot(T1_thresh_list, stat_df.loc[(x, 'RL3 (T=1)',  'TP'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(T3_thresh_list, stat_df.loc[(x, 'RL3 (T=3)',  'TP'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(T5_thresh_list, stat_df.loc[(x, 'RL3 (T=5)',  'TP'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(T7_thresh_list, stat_df.loc[(x, 'RL3 (T=7)',  'TP'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Threshold value (average CREDI value [GWh/h])")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
axs[idx, idy].plot(T1_thresh_list, stat_df.loc[(x, 'RL3 (T=1)',  'TN'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(T3_thresh_list, stat_df.loc[(x, 'RL3 (T=3)',  'TN'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(T5_thresh_list, stat_df.loc[(x, 'RL3 (T=5)',  'TN'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(T7_thresh_list, stat_df.loc[(x, 'RL3 (T=7)',  'TN'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Threshold value (average CREDI value [GWh/h])")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
axs[idx, idy].plot(T1_thresh_list, stat_df.loc[(x, 'RL3 (T=1)',  'FP'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(T3_thresh_list, stat_df.loc[(x, 'RL3 (T=3)',  'FP'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(T5_thresh_list, stat_df.loc[(x, 'RL3 (T=5)',  'FP'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(T7_thresh_list, stat_df.loc[(x, 'RL3 (T=7)',  'FP'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Threshold value (average CREDI value [GWh/h])")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
axs[idx, idy].plot(T1_thresh_list, stat_df.loc[(x, 'RL3 (T=1)',  'FN'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(T3_thresh_list, stat_df.loc[(x, 'RL3 (T=3)',  'FN'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(T5_thresh_list, stat_df.loc[(x, 'RL3 (T=5)',  'FN'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(T7_thresh_list, stat_df.loc[(x, 'RL3 (T=7)',  'FN'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Threshold value (average CREDI value [GWh/h])")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
axs[idx, idy].plot(T1_thresh_list, stat_df.loc[(x, 'RL3 (T=1)',  'PR'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(T3_thresh_list, stat_df.loc[(x, 'RL3 (T=3)',  'PR'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(T5_thresh_list, stat_df.loc[(x, 'RL3 (T=5)',  'PR'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(T7_thresh_list, stat_df.loc[(x, 'RL3 (T=7)',  'PR'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('Precision in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Threshold value (average CREDI value [GWh/h])")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
axs[idx, idy].plot(T1_thresh_list, stat_df.loc[(x, 'RL3 (T=1)',  'RE'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(T3_thresh_list, stat_df.loc[(x, 'RL3 (T=3)',  'RE'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(T5_thresh_list, stat_df.loc[(x, 'RL3 (T=5)',  'RE'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(T7_thresh_list, stat_df.loc[(x, 'RL3 (T=7)',  'RE'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].set_ylabel('Recall in total)\n(How many ENS are identified as droughts?)')
axs[idx, idy].set_xlabel("Threshold value (average CREDI value [GWh/h])")
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
# Plot F-score for different duration T - explore large T
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

zone = 'DE00'

ens_dataset = 'ERAA23'

figname = f"Validation_Stoop24_{ens_dataset}_ENS_{zone}_compare_T_largeT"

"""
LWS_percs = np.linspace(0.001,0.30,200)
RL_percs  = 1-LWS_percs
DD_percs  = 1-LWS_percs
"""
# ---------------------------------------------
# Compute data for figure (~ 15s/percentile)
# ---------------------------------------------
start_time = time.time()

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
#PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

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

common_index = data3_RL_h.loc[('HIST')][[zone]].index.intersection(ens_mask.index)

PERIOD_length_days = 1
PERIOD_cluster_days = 1
# Get CREDI events
T1_CREDI_event, T1_event_dates, \
    T1_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 3
PERIOD_cluster_days = 3
# Get CREDI events
T3_CREDI_event, T3_event_dates, \
    T3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 5
PERIOD_cluster_days = 4
# Get CREDI events
T5_CREDI_event, T5_event_dates, \
    T5_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 7
PERIOD_cluster_days = 6
# Get CREDI events
T7_CREDI_event, T7_event_dates, \
    T7_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 60
PERIOD_cluster_days = 48
# Get CREDI events
TX_CREDI_event, TX_event_dates, \
    TX_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

N_event_T1 = []
N_event_T3 = []
N_event_T5 = []
N_event_T7 = []
N_event_TX = []

p_list  = []
for p in range(len(LWS_percs)):
    # find capacity thresholds
    # Percentile is computed on the clustered events ("independent" events)
    # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
    T1_thresh = np.quantile(T1_event_values, q=RL_percs[p], interpolation="nearest")
    T3_thresh = np.quantile(T3_event_values, q=RL_percs[p], interpolation="nearest")
    T5_thresh = np.quantile(T5_event_values, q=RL_percs[p], interpolation="nearest")
    T7_thresh = np.quantile(T7_event_values, q=RL_percs[p], interpolation="nearest")
    TX_thresh = np.quantile(TX_event_values, q=RL_percs[p], interpolation="nearest")

    # Calculate F (compared to ENS)
    T1_stat = get_f_score_CREDI_new(data3_ENS_d, T1_event_dates, T1_event_values, T1_thresh, common_index, zone, 
                                    PERIOD_length_days=1, extreme_is_high=True, beta=1)
    T3_stat = get_f_score_CREDI_new(data3_ENS_d, T3_event_dates, T3_event_values, T3_thresh, common_index, zone, 
                                    PERIOD_length_days=3, extreme_is_high=True, beta=1)
    T5_stat = get_f_score_CREDI_new(data3_ENS_d, T5_event_dates, T5_event_values, T5_thresh, common_index, zone, 
                                    PERIOD_length_days=5, extreme_is_high=True, beta=1)
    T7_stat = get_f_score_CREDI_new(data3_ENS_d, T7_event_dates, T7_event_values, T7_thresh, common_index, zone, 
                                    PERIOD_length_days=7, extreme_is_high=True, beta=1)
    TX_stat = get_f_score_CREDI_new(data3_ENS_d, TX_event_dates, TX_event_values, TX_thresh, common_index, zone, 
                                    PERIOD_length_days=60, extreme_is_high=True, beta=1)

    """
    # Mask the data / Detect Drought days
    T1_mask = mask_data(T1_CREDI_event[[zone]], T1_thresh, False, 1, 0) # WARNING! No clustering was applied to T1_CREDI_event 
    T3_mask = mask_data(T3_CREDI_event[[zone]], T3_thresh, False, 1, 0)
    T5_mask = mask_data(T5_CREDI_event[[zone]], T5_thresh, False, 1, 0)
    T7_mask = mask_data(T7_CREDI_event[[zone]], T7_thresh, False, 1, 0)
    """
    # Count number of drought events (it changes with T)
    N_event_T1.append(np.sum(np.asarray(T1_event_values) > T1_thresh))
    N_event_T3.append(np.sum(np.asarray(T3_event_values) > T3_thresh))
    N_event_T5.append(np.sum(np.asarray(T5_event_values) > T5_thresh))
    N_event_T7.append(np.sum(np.asarray(T7_event_values) > T7_thresh))
    N_event_TX.append(np.sum(np.asarray(TX_event_values) > TX_thresh))

    # Create Dataframe
    p_list.append( pd.concat([T1_stat, T3_stat, T5_stat, T7_stat, TX_stat], 
                             keys=['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)', 'RL3 (T=7)', 'RL3 (T=X)'],   names=['Drought type']))
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


# Percentile for peak F-score
F_max_T1 = stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)].max()
p_max_T1 = stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)] == F_max_T1].index[0][0]
print(f"RL3 (T=1): F_max = {F_max_T1}, p_max = {p_max_T1}")
F_max_T3 = stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)].max()
p_max_T3 = stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)] == F_max_T3].index[0][0]
print(f"RL3 (T=3): F_max = {F_max_T3}, p_max = {p_max_T3}")
F_max_T5 = stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)].max()
p_max_T5 = stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)] == F_max_T5].index[0][0]
print(f"RL3 (T=5): F_max = {F_max_T5}, p_max = {p_max_T5}")
F_max_T7 = stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)].max()
p_max_T7 = stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)] == F_max_T7].index[0][0]
print(f"RL3 (T=7): F_max = {F_max_T7}, p_max = {p_max_T7}")
F_max_TX = stat_df.loc[(x, 'RL3 (T=X)',  'F'),(zone)].max()
p_max_TX = stat_df.loc[(x, 'RL3 (T=X)',  'F'),(zone)][stat_df.loc[(x, 'RL3 (T=X)',  'F'),(zone)] == F_max_TX].index[0][0]
print(f"RL3 (T=X): F_max = {F_max_TX}, p_max = {p_max_TX}")


fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle('Stoop Method for '+zone+' using '+ens_dataset+' ENS')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
ax_big.axvline(p_max_T1, alpha=1, color=dt_colors[0], linestyle='dashed')
ax_big.axvline(p_max_T7, alpha=1, color=dt_colors[3], linestyle='dashed')
ax_big.axvline(p_max_TX, alpha=1, color=dt_colors[4], linestyle='dashed')
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
ax_big.plot(x, stat_df.loc[(x, 'RL3 (T=X)',  'F'),(zone)], label='RL3 (T=X)',  color=dt_colors[4], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()




print(f"Number of events for T=1 at F-score peak: {N_event_T1[stat_df.loc[(x, 'RL3 (T=1)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=3 at F-score peak: {N_event_T3[stat_df.loc[(x, 'RL3 (T=3)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=5 at F-score peak: {N_event_T5[stat_df.loc[(x, 'RL3 (T=5)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=7 at F-score peak: {N_event_T7[stat_df.loc[(x, 'RL3 (T=7)',  'F'),(zone)].argmax()]}")
print(f"Number of events for T=X at F-score peak: {N_event_TX[stat_df.loc[(x, 'RL3 (T=X)',  'F'),(zone)].argmax()]}")


idx, idy = 1, 0
axs[idx, idy].axvline(p_max_T1, alpha=1, color=dt_colors[0], linestyle='dashed')
axs[idx, idy].axvline(p_max_T7, alpha=1, color=dt_colors[3], linestyle='dashed')
axs[idx, idy].axvline(p_max_TX, alpha=1, color=dt_colors[4], linestyle='dashed')
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'TP'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'TP'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'TP'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'TP'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=X)',  'TP'),(zone)], label='RL3 (T=X)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
axs[idx, idy].axvline(p_max_T1, alpha=1, color=dt_colors[0], linestyle='dashed')
axs[idx, idy].axvline(p_max_T7, alpha=1, color=dt_colors[3], linestyle='dashed')
axs[idx, idy].axvline(p_max_TX, alpha=1, color=dt_colors[4], linestyle='dashed')
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'TN'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'TN'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'TN'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'TN'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=X)',  'TN'),(zone)], label='RL3 (T=X)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
axs[idx, idy].axvline(p_max_T1, alpha=1, color=dt_colors[0], linestyle='dashed')
axs[idx, idy].axvline(p_max_T7, alpha=1, color=dt_colors[3], linestyle='dashed')
axs[idx, idy].axvline(p_max_TX, alpha=1, color=dt_colors[4], linestyle='dashed')
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'FP'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'FP'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'FP'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'FP'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=X)',  'FP'),(zone)], label='RL3 (T=X)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
axs[idx, idy].axvline(p_max_T1, alpha=1, color=dt_colors[0], linestyle='dashed')
axs[idx, idy].axvline(p_max_T7, alpha=1, color=dt_colors[3], linestyle='dashed')
axs[idx, idy].axvline(p_max_TX, alpha=1, color=dt_colors[4], linestyle='dashed')
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'FN'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'FN'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'FN'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'FN'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=X)',  'FN'),(zone)], label='RL3 (T=X)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
axs[idx, idy].axvline(p_max_T1, alpha=1, color=dt_colors[0], linestyle='dashed')
axs[idx, idy].axvline(p_max_T7, alpha=1, color=dt_colors[3], linestyle='dashed')
axs[idx, idy].axvline(p_max_TX, alpha=1, color=dt_colors[4], linestyle='dashed')
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'PR'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'PR'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'PR'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'PR'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=X)',  'PR'),(zone)], label='RL3 (T=X)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].set_ylabel('Precision in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
axs[idx, idy].axvline(p_max_T1, alpha=1, color=dt_colors[0], linestyle='dashed')
axs[idx, idy].axvline(p_max_T7, alpha=1, color=dt_colors[3], linestyle='dashed')
axs[idx, idy].axvline(p_max_TX, alpha=1, color=dt_colors[4], linestyle='dashed')
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=1)',  'RE'),(zone)], label='RL3 (T=1)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=3)',  'RE'),(zone)], label='RL3 (T=3)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=5)',  'RE'),(zone)], label='RL3 (T=5)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=7)',  'RE'),(zone)], label='RL3 (T=7)',  color=dt_colors[3], alpha=0.8)
axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3 (T=X)',  'RE'),(zone)], label='RL3 (T=X)',  color=dt_colors[4], alpha=0.8)
axs[idx, idy].set_ylabel('Recall in total)\n(How many ENS are identified as droughts?)')
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

zone = 'AT00'

ens_dataset = 'ERAA23'

figname = f"Validation_Stoop24_all_new_ENS_{zone}_T{PERIOD_length_days}"


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

# TODO: Dirty -> I used an old piece of code, I should update that:
data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[[zone]]
ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

common_index = data3_RL_h.loc[('HIST')].index.intersection(ens_mask.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

stat_df = dict()
for scenario_EVA in ['A', 'B']:
    stat_df[scenario_EVA] = dict()
    for FOS in range(1, 15+1):

        data3_ENS_d = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[[zone]]
        
        p_list  = []
        for p in range(len(LWS_percs)):
            # find capacity thresholds
            # Percentile is computed on the clustered events ("independent" events)
            # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
            rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")

            # Calculate F (compared to ENS)
            rl3_stat = get_f_score_CREDI_new(data3_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                            PERIOD_length_days=1, extreme_is_high=True, beta=1)

            # Create Dataframe
            p_list.append( pd.concat([rl3_stat], 
                                    keys=['RL'],   names=['Drought type']))
            print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
        stat_df[scenario_EVA][FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
#stat_df.to_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")
#stat_df.to_csv(f"{path_to_plot}Plot_data/{figname}_stats.csv", sep=';')


#%%
# ---------------------------------------------
#  Plot F / threshold (comparing to ENS / validation)
# ---------------------------------------------

# Load data
#stat_df = pd.read_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")

x=stat_df['A'][1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

# Get values for old dataset
#stat_df_old =  pd.read_pickle(f"{path_to_plot}Plot_data/Validation_Stoop24_ERAA23_ENS_{zone}_T{PERIOD_length_days}_stats.pkl")

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for scenario_EVA in ['A', 'B']:
    quantiles_dict[scenario_EVA] = dict()
    min_dict[scenario_EVA] = dict()
    max_dict[scenario_EVA] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN', 'PR', 'RE']:
        quantiles_dict[scenario_EVA][metric] = np.quantile([stat_df[scenario_EVA][FOS].loc[(x, 'RL',  metric),(zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[scenario_EVA][metric] = np.min([stat_df[scenario_EVA][FOS].loc[(x, 'RL',  metric),(zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[scenario_EVA][metric] = np.max([stat_df[scenario_EVA][FOS].loc[(x, 'RL',  metric),(zone)] for FOS in range(1, 15+1)], axis=0)


fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle(f'Stoop Method for {zone} using new ENS dataset (T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
metric = 'F'
#ax_big.plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
ax_big.plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
ax_big.fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
ax_big.plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
ax_big.plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
ax_big.plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
ax_big.fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
ax_big.plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
ax_big.plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Threshold value (average CREDI value [GWh/h])")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
metric = 'TP'
#axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
metric = 'TN'
#axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
metric = 'FP'
#axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
metric = 'FN'
#axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
metric = 'PR'
#axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
metric = 'RE'
#axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['A'][metric][1], label='RL (Scenario A)',  color=dt_colors[1], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['A'][metric][0], quantiles_dict['A'][metric][2], color=dt_colors[1], alpha=0.5)
axs[idx, idy].plot(x, min_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, max_dict['A'][metric], linestyle='dashed', color=dt_colors[1], alpha=0.8)
axs[idx, idy].plot(x, quantiles_dict['B'][metric][1], label='RL (Scenario B)',  color=dt_colors[2], alpha=0.8)
axs[idx, idy].fill_between(x, quantiles_dict['B'][metric][0], quantiles_dict['B'][metric][2], color=dt_colors[2], alpha=0.5)
axs[idx, idy].plot(x, min_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
axs[idx, idy].plot(x, max_dict['B'][metric], linestyle='dashed', color=dt_colors[2], alpha=0.8)
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





















#%%
# =================================================================
# Plot F-score | Compare RL, DD, RES for scenario B
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

zone = 'FR00'

scenario_EVA = 'B'

figname = f"Validation_Stoop24_ENS_scenario{scenario_EVA}_{zone}_T{PERIOD_length_days}"


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

# TODO: Dirty -> I used an old piece of code, I should update that:
data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[[zone]]
ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

common_index = data3_RL_h.loc[('HIST')].index.intersection(ens_mask.index)

# Get CREDI events
lws3_CREDI_event, lws3_event_dates, \
    lws3_event_values = get_CREDI_events(data3_gen_h.loc[('HIST')], zone, extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                         PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
dd3_CREDI_event, dd3_event_dates, \
    dd3_event_values = get_CREDI_events(data3_dem_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')


stat_df = dict()
for FOS in range(1, 15+1):

    data3_ENS_d = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[[zone]]
    
    p_list  = []
    for p in range(len(LWS_percs)):
        # find capacity thresholds
        # Percentile is computed on the clustered events ("independent" events)
        # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
        lws3_thresh = np.quantile(lws3_event_values, q=LWS_percs[p], interpolation="nearest")
        rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")
        dd3_thresh = np.quantile(dd3_event_values, q=DD_percs[p], interpolation="nearest")

        # Calculate F (compared to ENS)
        lws3_stat = get_f_score_CREDI_new(data3_ENS_d, lws3_event_dates, lws3_event_values, lws3_thresh, common_index, zone, 
                                    PERIOD_length_days=1, extreme_is_high=False, beta=1)
        rl3_stat = get_f_score_CREDI_new(data3_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                        PERIOD_length_days=1, extreme_is_high=True, beta=1)
        dd3_stat = get_f_score_CREDI_new(data3_ENS_d, dd3_event_dates, dd3_event_values, dd3_thresh, common_index, zone, 
                                        PERIOD_length_days=1, extreme_is_high=True, beta=1)

        # Create Dataframe
        p_list.append( pd.concat([rl3_stat, lws3_stat, dd3_stat], keys=['RL', 'LWS', 'DD'],   names=['Drought type']))
        print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
    stat_df[FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])

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

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL', 'DD', 'LWS']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN', 'PR', 'RE']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], axis=0)

# Percentile for peak F-score
ener_var = 'RL'
metric = 'F'
F_max = quantiles_dict[ener_var][metric][1].max()
p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
print(f"RL3 (T=1): F_max = {F_max}, p_max = {p_max}")

fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle(f'Stoop Method for {zone} using scenario {scenario_EVA} (T={PERIOD_length_days}d)')

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
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
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
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
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
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
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
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
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
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
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



















#%%
# =================================================================
# Plot F-score | Only ENS present in at least N FOS scenarios
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

zone = 'DE00'

scenario_EVA = 'B'

figname = f"Validation_Stoop24_combineFOS_Scenario{scenario_EVA}_{zone}_T{PERIOD_length_days}"


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

# TODO: Dirty -> I used an old piece of code, I should update that:
data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[[zone]]
ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

common_index = data3_RL_h.loc[('HIST')].index.intersection(ens_mask.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')


data3_ENS_fos = []
for FOS in range(1, 15+1):
    data3_ENS_fos.append(pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[[zone]])
df_ENS_sum_fos = sum((data3_ENS_fos[FOS] > 0) * 1 for FOS in range(15))

stat_df = dict()
for nFOS in range(1, 15+1):

        df_ENS_in_nFOS = (df_ENS_sum_fos >= nFOS) * 1
        
        p_list  = []
        for p in range(len(LWS_percs)):
            # find capacity thresholds
            # Percentile is computed on the clustered events ("independent" events)
            # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
            rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")

            # Calculate F (compared to ENS)
            rl3_stat = get_f_score_CREDI_new(df_ENS_in_nFOS, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                            PERIOD_length_days=1, extreme_is_high=True, beta=1)

            # Create Dataframe
            p_list.append( pd.concat([rl3_stat], 
                                    keys=['RL'],   names=['Drought type']))
            print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
        stat_df[nFOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
#stat_df.to_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")
#stat_df.to_csv(f"{path_to_plot}Plot_data/{figname}_stats.csv", sep=';')


#%%
# ---------------------------------------------
#  Plot F / threshold (comparing to ENS / validation)
# ---------------------------------------------

# Load data
#stat_df = pd.read_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")


#nFOS_list = np.arange(1, 15+1)
nFOS_list = [1, 5, 8, 10, 15]

# Create a color gradient
colors = plt.cm.viridis(np.linspace(0, 1, 15))
# Normalize the colors for the colormap
norm = Normalize(vmin=0, vmax=15)
scalar_map = ScalarMappable(norm=norm, cmap='plasma')

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

# Get values for old dataset
stat_df_old =  pd.read_pickle(f"{path_to_plot}Plot_data/Validation_Stoop24_ERAA23_ENS_{zone}_T{PERIOD_length_days}_stats.pkl")


fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle(f'Stoop Method for {zone} using scenario {scenario_EVA} (T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
metric = 'F'
for nFOS in nFOS_list:
    ax_big.plot(x, stat_df[nFOS].loc[(x, 'RL',  metric),(zone)], color=scalar_map.to_rgba(nFOS-1), label=f'days with ENS>0 in at least {nFOS} FOS')
ax_big.plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Threshold value (average CREDI value [GWh/h])")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
metric = 'TP'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, stat_df[nFOS].loc[(x, 'RL',  metric),(zone)], color=scalar_map.to_rgba(nFOS-1))
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
metric = 'TN'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, stat_df[nFOS].loc[(x, 'RL',  metric),(zone)], color=scalar_map.to_rgba(nFOS-1))
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
metric = 'FP'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, stat_df[nFOS].loc[(x, 'RL',  metric),(zone)], color=scalar_map.to_rgba(nFOS-1))
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
metric = 'FN'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, stat_df[nFOS].loc[(x, 'RL',  metric),(zone)], color=scalar_map.to_rgba(nFOS-1))
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
metric = 'PR'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, stat_df[nFOS].loc[(x, 'RL',  metric),(zone)], color=scalar_map.to_rgba(nFOS-1))
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
metric = 'RE'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, stat_df[nFOS].loc[(x, 'RL',  metric),(zone)], color=scalar_map.to_rgba(nFOS-1))
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
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



























#%%
# =================================================================
# Plot F-score | Look at all FOS, take only days with ENS>0 in nFOS scenarios
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

zone = 'DE00'

scenario_EVA = 'B'

nFOS = 10 # out of 15 FOS scenarios

figname = f"Validation_Stoop24_atleast_{nFOS}FOS_Scenario{scenario_EVA}_{zone}_T{PERIOD_length_days}"


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

# TODO: Dirty -> I used an old piece of code, I should update that:
data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[[zone]]
ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

common_index = data3_RL_h.loc[('HIST')].index.intersection(ens_mask.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')


data3_ENS_fos = []
for FOS in range(1, 15+1):
    data3_ENS_fos.append(pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[[zone]])
df_ENS_sum_fos = sum((data3_ENS_fos[FOS] > 0) * 1 for FOS in range(15))
df_ENS_in_nFOS = (df_ENS_sum_fos >= nFOS) * 1


stat_df = dict()
for FOS in range(1, 15+1):

    df_ENS_fos = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[[zone]]
    # Take Only day with ENS>0 in at least nFOS to filter ENS that are likely weather induced
    df_ENS_d = df_ENS_fos * df_ENS_in_nFOS

    p_list  = []
    for p in range(len(LWS_percs)):
        # find capacity thresholds
        # Percentile is computed on the clustered events ("independent" events)
        # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
        rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")

        # Calculate F (compared to ENS)
        rl3_stat = get_f_score_CREDI_new(df_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                        PERIOD_length_days=1, extreme_is_high=True, beta=1)

        # Create Dataframe
        p_list.append( pd.concat([rl3_stat], keys=['RL'],   names=['Drought type']))
        print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
    stat_df[FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
#stat_df.to_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")
#stat_df.to_csv(f"{path_to_plot}Plot_data/{figname}_stats.csv", sep=';')


#%%
# ---------------------------------------------
#  Plot F / threshold (comparing to ENS / validation)
# ---------------------------------------------

# Load data
#stat_df = pd.read_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN', 'PR', 'RE']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], axis=0)

fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle(f'Stoop Method for {zone} for days with ENS>0 in at least {nFOS} FOS (Scenario {scenario_EVA}, T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
metric = 'F'
for ncolor, ener_var in enumerate(['RL']):
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
for ncolor, ener_var in enumerate(['RL']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
metric = 'TN'
for ncolor, ener_var in enumerate(['RL']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
metric = 'FP'
for ncolor, ener_var in enumerate(['RL']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
metric = 'FN'
for ncolor, ener_var in enumerate(['RL']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
metric = 'PR'
for ncolor, ener_var in enumerate(['RL']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
metric = 'RE'
for ncolor, ener_var in enumerate(['RL']):
    axs[idx, idy].plot(x, quantiles_dict[ener_var][metric][1], label=ener_var,  color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].fill_between(x, quantiles_dict[ener_var][metric][0], quantiles_dict[ener_var][metric][2], color=dt_colors[ncolor], alpha=0.5)
    axs[idx, idy].plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    axs[idx, idy].plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
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



























#%%
# =================================================================
# Plot F-score | a priori weather induced ENS, for a list of nFOS
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

zone = 'DE00'

scenario_EVA = 'B'

nFOS_list = [3, 6, 9, 12, 15] # out of 15 FOS scenarios

figname = f"Validation_Stoop24_list_FOS_Scenario{scenario_EVA}_{zone}_T{PERIOD_length_days}"


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

# TODO: Dirty -> I used an old piece of code, I should update that:
data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_old_ENS_TY2033_daily.pkl')[[zone]]
ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

common_index = data3_RL_h.loc[('HIST')].index.intersection(ens_mask.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')][[zone]], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')


data3_ENS_fos = []
for FOS in range(1, 15+1):
    data3_ENS_fos.append(pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[[zone]])
df_ENS_sum_fos = sum((data3_ENS_fos[FOS] > 0) * 1 for FOS in range(15))

stat_df = dict()
for nFOS in nFOS_list:
    stat_df[nFOS] = dict()
    df_ENS_in_nFOS = (df_ENS_sum_fos >= nFOS) * 1
    for FOS in range(1, 15+1):

        df_ENS_fos = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[[zone]]
        # Take Only day with ENS>0 in at least nFOS to filter ENS that are likely weather induced
        df_ENS_d = df_ENS_fos * df_ENS_in_nFOS

        p_list  = []
        for p in range(len(LWS_percs)):
            # find capacity thresholds
            # Percentile is computed on the clustered events ("independent" events)
            # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
            rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")

            # Calculate F (compared to ENS)
            rl3_stat = get_f_score_CREDI_new(df_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, zone, 
                                            PERIOD_length_days=1, extreme_is_high=True, beta=1)

            # Create Dataframe
            p_list.append( pd.concat([rl3_stat], keys=['RL'],   names=['Drought type']))
            print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
        stat_df[nFOS][FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
#stat_df.to_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")
#stat_df.to_csv(f"{path_to_plot}Plot_data/{figname}_stats.csv", sep=';')


#%%
# ---------------------------------------------
#  Plot F / threshold (comparing to ENS / validation)
# ---------------------------------------------

# Load data
#stat_df = pd.read_pickle(f"{path_to_plot}Plot_data/{figname}_stats.pkl")


# Create a color gradient
colors = plt.cm.viridis(np.linspace(0, 1, 15))
# Normalize the colors for the colormap
norm = Normalize(vmin=0, vmax=15)
scalar_map = ScalarMappable(norm=norm, cmap='plasma')

# Get values for old dataset
stat_df_old =  pd.read_pickle(f"{path_to_plot}Plot_data/Validation_Stoop24_ERAA23_ENS_{zone}_T{PERIOD_length_days}_stats.pkl")

x=stat_df[nFOS_list[0]][1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for nFOS in nFOS_list:
        quantiles_dict[ener_var][nFOS] = dict()
        min_dict[ener_var][nFOS] = dict()
        max_dict[ener_var][nFOS] = dict()
        for metric in ['F', 'TP', 'TN', 'FP', 'FN', 'PR', 'RE']:
            quantiles_dict[ener_var][nFOS][metric] = np.quantile([stat_df[nFOS][FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
            min_dict[ener_var][nFOS][metric] = np.min([stat_df[nFOS][FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], axis=0)
            max_dict[ener_var][nFOS][metric] = np.max([stat_df[nFOS][FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], axis=0)

fig, axs = plt.subplots(4, 2, figsize=(10,16))
fig.suptitle(f'Stoop Method for {zone} for days with ENS>0 in at least N FOS (Scenario {scenario_EVA}, T={PERIOD_length_days}d)')

# Event time series
idx, idy = 0, [0,1]
ax_big = plt.subplot(4, 1, 1)
metric = 'F'
ener_var = 'RL'
for nFOS in nFOS_list:
    ax_big.plot(x, quantiles_dict[ener_var][nFOS][metric][1], label=f'{ener_var} N={nFOS} (Q50)',  color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #ax_big.fill_between(x, quantiles_dict[ener_var][nFOS][metric][0], quantiles_dict[ener_var][nFOS][metric][2], color=scalar_map.to_rgba(nFOS-1), alpha=0.5)
    #ax_big.plot(x, min_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #ax_big.plot(x, max_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
ax_big.plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
ax_big.set_ylabel('F-Score')
ax_big.set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
axs[0,0].remove()
axs[0,1].remove()

idx, idy = 1, 0
metric = 'TP'
ener_var = 'RL'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, quantiles_dict[ener_var][nFOS][metric][1], label=f'{ener_var} N={nFOS}',  color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].fill_between(x, quantiles_dict[ener_var][nFOS][metric][0], quantiles_dict[ener_var][nFOS][metric][2], color=scalar_map.to_rgba(nFOS-1), alpha=0.5)
    #axs[idx, idy].plot(x, min_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].plot(x, max_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('True Positives (out of '+str(nr_of_pos[zone])+' in total)\n(DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 1, 1
metric = 'TN'
ener_var = 'RL'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, quantiles_dict[ener_var][nFOS][metric][1], label=f'{ener_var} N={nFOS}',  color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].fill_between(x, quantiles_dict[ener_var][nFOS][metric][0], quantiles_dict[ener_var][nFOS][metric][2], color=scalar_map.to_rgba(nFOS-1), alpha=0.5)
    #axs[idx, idy].plot(x, min_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].plot(x, max_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('True Negatives (out of '+str(nr_of_neg[zone])+' in total)\n(no DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 2, 0
metric = 'FP'
ener_var = 'RL'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, quantiles_dict[ener_var][nFOS][metric][1], label=f'{ener_var} N={nFOS}',  color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].fill_between(x, quantiles_dict[ener_var][nFOS][metric][0], quantiles_dict[ener_var][nFOS][metric][2], color=scalar_map.to_rgba(nFOS-1), alpha=0.5)
    #axs[idx, idy].plot(x, min_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].plot(x, max_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('False Positives (out of '+str(nr_of_neg[zone])+' in total)\n(DF detected, when no ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)

idx, idy = 2, 1
metric = 'FN'
ener_var = 'RL'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, quantiles_dict[ener_var][nFOS][metric][1], label=f'{ener_var} N={nFOS}',  color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].fill_between(x, quantiles_dict[ener_var][nFOS][metric][0], quantiles_dict[ener_var][nFOS][metric][2], color=scalar_map.to_rgba(nFOS-1), alpha=0.5)
    #axs[idx, idy].plot(x, min_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].plot(x, max_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('False Negatives (out of '+str(nr_of_pos[zone])+' in total)\n(No DF detected, when ENS)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 0
metric = 'PR'
ener_var = 'RL'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, quantiles_dict[ener_var][nFOS][metric][1], label=f'{ener_var} N={nFOS}',  color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].fill_between(x, quantiles_dict[ener_var][nFOS][metric][0], quantiles_dict[ener_var][nFOS][metric][2], color=scalar_map.to_rgba(nFOS-1), alpha=0.5)
    #axs[idx, idy].plot(x, min_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].plot(x, max_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
axs[idx, idy].set_ylabel('Precision (out of '+str(nr_of_pos[zone])+' in total)\n(How many detected droughts are ENS?)')
axs[idx, idy].set_xlabel("Percentile of top CREDI events labelled 'Dunkelflaute'")
axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=False))
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)

idx, idy = 3, 1
metric = 'RE'
ener_var = 'RL'
for nFOS in nFOS_list:
    axs[idx, idy].plot(x, quantiles_dict[ener_var][nFOS][metric][1], label=f'{ener_var} N={nFOS}',  color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].fill_between(x, quantiles_dict[ener_var][nFOS][metric][0], quantiles_dict[ener_var][nFOS][metric][2], color=scalar_map.to_rgba(nFOS-1), alpha=0.5)
    #axs[idx, idy].plot(x, min_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
    #axs[idx, idy].plot(x, max_dict[ener_var][nFOS][metric], linestyle='dashed', color=scalar_map.to_rgba(nFOS-1), alpha=0.8)
axs[idx, idy].plot(x, stat_df_old.loc[(x, 'RL3',  metric),(zone)], label='RL (old dataset)',  color=dt_colors[0], alpha=0.8)
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

PERIOD_length_days = 1
PERIOD_cluster_days = 1

# percentile for peak F-score
# for DE00 ! Can also change when changing the list of percentiles.
p_max = 0.01
#p_max = 0.022
#p_max = 0.0356
#p_max = 0.044

zone = 'DE00'

ens_dataset = 'ERAA23'

scenario_EVA = 'B'
fos = 2

figname = f"Timeline_Stoop24_{ens_dataset}_ENS_{zone}_T{PERIOD_length_days}_Tc{PERIOD_cluster_days}_pmax_{int(p_max*1000)}e-3"

# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24

# Create colormap
colors = ['whitesmoke', 'royalblue', 'red', 'limegreen']
cmap=ListedColormap(colors)
legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0],label='No DF, No ENS'),
                   Patch(facecolor=colors[1], edgecolor=colors[1],label='DF detected, but no ENS'),
                   Patch(facecolor=colors[2], edgecolor=colors[2],label='No DF, but ENS'),
                   Patch(facecolor=colors[3], edgecolor=colors[3],label='DF detected and ENS')]

if ens_dataset=='AO':
    data3_ENS_d = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_daily.pkl')
    X1= np.arange(364)
    X2= np.arange(364*24)
elif ens_dataset=='ERAA23_old':
    data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_ENS_TY2033_daily.pkl')
    X1= np.arange(365)
    X2= np.arange(365*24)
elif ens_dataset=='ERAA23':
    data3_ENS_d = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{fos}_daily.pkl')
    X1= np.arange(365)
    X2= np.arange(365*24)
else:
    raise KeyError('ENS Dataset not existent!')
Y = np.arange(1982,2017) # common period of ENS & PECD

# Generate masked data
ens_mask_d = mask_data(data3_ENS_d, 0, False, 2, 0)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(data3_RL_h.loc[('HIST')], zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

rl3_thresh = np.quantile(rl3_event_values, q=1-p_max, interpolation="nearest")

df_mask_timeline = compute_timeline(data3_ENS_d, rl3_event_dates, rl3_event_values, rl3_thresh, zone, PERIOD_length_days, 
                                    extreme_is_high=True, start_date='1982-01-01', end_date='2016-12-31')


# Tranform detection masks into the right format
def detmask2matrix(detmask,X,Y):
    return np.reshape(detmask, (len(Y),len(X)))

fig, ax = plt.subplots(1, 1, figsize=(10, 4))

#axs.pcolormesh(X1, Y, detmask2matrix(rl3_detmask[zones_szon[c]],X1,Y), cmap=cmap)
ax.pcolormesh(X1, Y, detmask2matrix(df_mask_timeline[zone],X1,Y), cmap=cmap)
ax.set_title(f'RL (PECD 3.1, {ens_dataset}, {zone}, T={PERIOD_length_days}, Tc={PERIOD_cluster_days}, p={p_max})')
ax.set_ylabel('Year')
ax.set_xlabel('Day of Year')
ax.legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)


plt.tight_layout()
#plt.savefig(f"{path_to_plot}Validation/{figname}.{plot_format}", dpi=300)
print(f"Saved {path_to_plot}Validation/{figname}.{plot_format}")
#plt.close()
        
# TODO: Some masks are shorter? Why? In year with leap day the 31.12. is missing (and the leap day...)
      

# %%
