"""
Figures in Supplementary Information

For questions: benjamin.biewald@tennet.eu or bastien.cozian@rte-france.com
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
from scipy import stats
import glob
import os

from Dunkelflaute_function_library import get_zones, get_thresholds, detect_drought_Otero22, mask_data, get_f_score, \
    get_df_timerange, lin_reg, detect_drought_Li21, mask_df_by_entries, detect_DF_Li21, mask_Li21, \
    pairwise_comparison, condorcet_ranking, get_daily_values, assign_SZON, get_daily_values_pecd
from CREDIfunctions import Modified_Ordinal_Hour, Climatology_Hourly, Climatology_Hourly_Rolling, \
    Climatology_Hourly_Weekly_Rolling, get_CREDI_events, get_f_score_CREDI, get_f_score_CREDI_new, \
    compute_timeline, get_correlation_CREDI, mask_CREDI

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








































#%%
# ===================================================================
# F-score | Stoop 23 | Compare different T | day-based approach
# ===================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].


scenario_EVA = 'B'

figname = f"Validation_Stoop24_ENS_scenario{scenario_EVA}_{agg_zone}_T1-5d"


# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()

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

PERIOD_length_days = 1
PERIOD_cluster_days = 1
# Get CREDI events
T1_CREDI_event, T1_event_dates, \
    T1_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 3
PERIOD_cluster_days = 3
# Get CREDI events
T3_CREDI_event, T3_event_dates, \
    T3_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

PERIOD_length_days = 5
PERIOD_cluster_days = 4
# Get CREDI events
T5_CREDI_event, T5_event_dates, \
    T5_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                       PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

stat_df = dict()
for FOS in range(1, 15+1):

    df_agg_ENS_fos_d = pd.DataFrame()
    df_agg_ENS_fos_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    
    p_list  = []
    for p in range(len(LWS_percs)):
        # Find capacity thresholds
        # Percentile is computed on the clustered events ("independent" events)
        # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
        T1_thresh = np.quantile(T1_event_values, q=RL_percs[p], interpolation="nearest")
        T3_thresh = np.quantile(T3_event_values, q=RL_percs[p], interpolation="nearest")
        T5_thresh = np.quantile(T5_event_values, q=RL_percs[p], interpolation="nearest")

        # Mask the data / Detect Drought days
        T1_mask = mask_CREDI(T1_event_dates, T1_event_values, T1_thresh, PERIOD_length_days=1, zone=agg_zone, extreme_is_high=True)
        T3_mask = mask_CREDI(T3_event_dates, T3_event_values, T3_thresh, PERIOD_length_days=3, zone=agg_zone, extreme_is_high=True)
        T5_mask = mask_CREDI(T5_event_dates, T5_event_values, T5_thresh, PERIOD_length_days=5, zone=agg_zone, extreme_is_high=True)
        ENS_fos_mask = mask_data(df_agg_ENS_fos_d, 0, False, 2, 0)

        # Calculate F
        T1_stat  = get_f_score(ENS_fos_mask, T1_mask,  beta=1)
        T3_stat  = get_f_score(ENS_fos_mask, T3_mask,  beta=1)
        T5_stat  = get_f_score(ENS_fos_mask, T5_mask,  beta=1)

        # Create Dataframe
        p_list.append( pd.concat([T1_stat, T3_stat, T5_stat], keys=['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)'],   names=['Drought type']))
        print('Done '+str(p+1)+'/'+str(len(LWS_percs)))

    stat_df[FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
    print(f'    Done FOS {FOS}')
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
pickle.dump(stat_df, open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "wb"))


#%%
# ---------------------------------------------
#  Plot only F
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].


scenario_EVA = 'B'

figname = f"Validation_Stoop24_ENS_scenario{scenario_EVA}_{agg_zone}_T1-5d"


# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)


# Percentile for peak F-score
for ener_var in ['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)']:
    metric = 'F'
    F_max = quantiles_dict[ener_var][metric][1].max()
    p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
    print(f"{ener_var}: F_max = {F_max}, p_max = {p_max}")



fig, axs = plt.subplots(1, 1, figsize=(5, 4))

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')

axs.set_title(zone_name, loc='left')
axs.set_title(f"Method Stoop'23", loc='right')

# Event time series

metric = 'F'
#for ncolor, ener_var, label in zip(range(4), ['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)', 'RL3 (T=7)'], ['Residual load (T=1d)', 'Residual load (T=3d)', 'Residual load (T=5d)', 'Residual load (T=7d)']):
for ncolor, ener_var, label in zip(range(3), ['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)'], ['Residual load (T=1d)', 'Residual load (T=3d)', 'Residual load (T=5d)']):
    axs.plot(x, quantiles_dict[ener_var][metric][1], label=label,  color=dt_colors[ncolor], alpha=0.8)
    axs.fill_between(x, min_dict[ener_var][metric], max_dict[ener_var][metric], color=dt_colors[ncolor], alpha=0.5)
    #axs.plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    #axs.plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs.set_ylabel('F-Score')
axs.set_xlabel("Percentile of top CREDI events")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs.legend(facecolor="white", loc='upper right', framealpha=1)
axs.grid(axis='y')
axs.set_ylim((0, 0.5))


plt.tight_layout()


plt.savefig(f"{path_to_plot}Validation/{figname}.pdf", dpi=300)



















#%%
# =================================================================
# Correlation for CREDI | T = 3, 5, 7 days
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------


scenario_EVA = 'B'

# --- T = 3 ---
#PERIOD_length_days = 3; PERIOD_cluster_days = 3

#zone = 'DE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0144
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0124
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0092
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0044
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.02
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.0052
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']; p_max = 0.019
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']; p_max = 0.008
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04']; p_max = 0.0084 # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']; p_max = 0.054
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']; p_max = 0.0228
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00']; p_max = 0.0152 # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00']; p_max = 0.024 # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00']; p_max = 0.06 # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

# --- T = 5 ---
PERIOD_length_days = 5; PERIOD_cluster_days = 4

#zone = 'DE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0204
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.002
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0044
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0032
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0116
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.008
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']; p_max = 0.0184
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']; p_max = 0.0144
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04']; p_max = 0.006 # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']; p_max = 0.0224
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']; p_max = 0.0188
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00']; p_max = 0.0196 # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00']; p_max = 0.0248 # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00']; p_max = 0.05 # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].



figname = f"Correlation_ENSvsCREDI_ENS_Scenario{scenario_EVA}_{agg_zone}_T{PERIOD_length_days}_Tc{PERIOD_cluster_days}_pmax_{int(p_max*1000)}e-3"

# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24

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



# Generate masked data
# TODO: Dirty -> I used an old piece of code, I should update that:
ens_mask = mask_data(df_agg_old_ENS_d, 0, False, 2, 0)
common_index = df_agg_RL_h.index.intersection(ens_mask.index)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')

rl3_thresh = np.quantile(rl3_event_values, q=1-p_max, interpolation="nearest")

rl3_DF_all = dict()
rl3_sum_ENS_all = dict()

for FOS in range(1, 15+1):

    df_agg_ENS_fos_d = pd.DataFrame()
    df_agg_ENS_fos_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    
    """
    # Only true positives
    rl3_DF_TP, rl3_sum_ENS_TP = get_correlation_CREDI(df_agg_ENS_fos_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, 
                                                    agg_zone, PERIOD_length_days=1, extreme_is_high=True, only_true_positive=True)
    """

    # With TP, FP, FN
    rl3_DF_all[FOS], rl3_sum_ENS_all[FOS] = get_correlation_CREDI(df_agg_ENS_fos_d, rl3_event_dates, rl3_event_values, rl3_thresh, common_index, 
                                                                  agg_zone, PERIOD_length_days=1, extreme_is_high=True, only_true_positive=False)



# ---------------------------------------------
# Plot only RL
# ---------------------------------------------

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    print('Warning: Set the title!')


fig, ax = plt.subplots(1, 1, figsize=(5,5))

r_value = dict()
rho_value = dict()

for nFOS in range(1, 16+1): # use 16 = 1 to correctly put the scatter plot on top with its label first

    FOS = nFOS

    if nFOS == 16:
        FOS = 1

    x = rl3_DF_all[FOS]
    y = rl3_sum_ENS_all[FOS]

    # Linear regression
    df_reg, intercept, slope, r_value[FOS], p_value, reg_trend = lin_reg(x, y)
    # Spearman rank-order correlation coefficient, removing NaNs from ENS list
    rho_value[FOS], p_spearman = stats.spearmanr(x[~np.isnan(y)], y[~np.isnan(y)])
    
    """
    if p_value < 0.05:
        ls  = 'solid'
    else:
        ls = 'dotted'
    """

    ls  = 'solid'
    if nFOS == 1 or nFOS == 16:
        color = dt_colors[0]
        alpha = 1
        marker = 'x'
        if nFOS == 1:
            label = f'outage scenario n°{FOS}'
        else:
            label = ''
    elif FOS == 2:
        color = 'grey'
        label = 'other outage scenarios'
        alpha = 0.4
        marker = '.'
    else:
        color = 'grey'
        label = ''
        alpha = 0.4
        marker = '.'

    ax.scatter(x, y, color=color, label=label, alpha=alpha, marker=marker)
    if nFOS == 1:
        ax.plot(df_reg, c=color, linestyle=ls)

    print(f'RL (3.1), FOS {FOS}: intercept={intercept}, slope={slope}, r_value={r_value[FOS]}, p_value={p_value}, reg_trend={reg_trend}')

r_q50 = np.quantile([r_value[FOS] for FOS in range(1, 15+1)], 0.5)
r_min = np.min([r_value[FOS] for FOS in range(1, 15+1)])
r_max = np.max([r_value[FOS] for FOS in range(1, 15+1)])
rho_q50 = np.quantile([rho_value[FOS] for FOS in range(1, 15+1)], 0.5)
rho_min = np.min([rho_value[FOS] for FOS in range(1, 15+1)])
rho_max = np.max([rho_value[FOS] for FOS in range(1, 15+1)])

#ax.set_title(f'RL (PECD 3.1), Scenario {scenario_EVA}, {agg_zone}, T={PERIOD_length_days}, Tc={PERIOD_cluster_days}, p={p_max})\n r={np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}] (Q50 [min, max])')
ax.set_title(zone_name+f' ($T={PERIOD_length_days}d$)', loc='left')
ax.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')

ax.set_ylabel('Total ENS [GWh]')
ax.set_xlabel('CREDI [GWh]')
ax.legend(facecolor="white", loc='upper left', framealpha=1)

plt.tight_layout()
plt.savefig(f"{path_to_plot}Correlation/{figname}.pdf")
#plt.close()


















# %%
# =================================================================
# Year ranking | Cumulative RL
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

scenario_EVA = 'B'

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.014
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.002
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0064
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0068
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.014

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.002
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']; p_max = 0.0104
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']; p_max = 0.0064
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04']; p_max = 0.002  # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']; p_max = 0.0064
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']; p_max = 0.0168
agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00']; p_max = 0.0172 # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00']; p_max = 0.0316  # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00']; p_max = 0.056  # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

figname = f"YearRank_CumulativeRL_ENS_Scenario{scenario_EVA}_{agg_zone}_all"

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_RL_d  = pd.DataFrame()

df_agg_RL_d[agg_zone]  = data3_RL_d.loc[('HIST')][zones_list].sum(axis=1)


# --- Simple Residual load metric ---

# Summed-up RL
df_RL_annual = df_agg_RL_d.groupby(df_agg_RL_d.index.year).sum()[agg_zone] * 1e-6

"""
# Cummulative annomaly with HWRW
ds_data = df_agg_RL_h.to_xarray()
ds_clim_HWRW, MOH = Climatology_Hourly_Weekly_Rolling(ds_data, RollingWindow=9)
ds_anom = ds_data.groupby(MOH) - ds_clim_HWRW
df_anom = ds_anom.to_pandas()
df_RL_anom_annual = df_anom.groupby(df_anom.index.year).sum()[agg_zone] * 1e-6
"""

# --- Annual metrics for FOS ---

df_ENS_annual_sum = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_annual_max = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_hours = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_days = pd.DataFrame(index=range(1982, 2016+1))

for FOS in range(1, 15+1):
    
    df_FOS_d = pd.DataFrame()
    df_FOS_h = pd.DataFrame()
    df_FOS_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    df_FOS_h[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_hourly.pkl')[zones_list].sum(axis=1)

    # Annual sum of ENS
    df_ENS_annual_sum[FOS] = df_FOS_d.groupby(df_FOS_d.index.year).sum()

    # Yearly maximum of daily ENS
    df_daily_FOS = df_FOS_d.groupby([df_FOS_d.index.year, df_FOS_d.index.month, df_FOS_d.index.day]).sum()
    time_index = pd.date_range(start='1982-01-01', end='2016-12-31', freq='d')
    time_index = time_index[~((time_index.day == 29) & (time_index.month == 2))]
    df_max_annual_FOS = pd.DataFrame(index=time_index)
    df_max_annual_FOS[agg_zone] = df_daily_FOS[agg_zone].values
    df_ENS_annual_max[FOS] = df_max_annual_FOS.groupby(df_max_annual_FOS.index.year).max()

    # Hour of ENS per year
    df_ENS_hours[FOS] = (df_FOS_h > 0).groupby(df_FOS_h.index.year).sum()

    # Number of days with ENS > 0
    df_ENS_days[FOS] = (df_FOS_d > 0).groupby(df_FOS_d.index.year).sum()



# -------------
#  Correlations
# -------------

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')
    
"""
x_list  = [df_RL_annual,                        df_RL_annual,                       df_RL_anom_annual,                          df_RL_anom_annual]
x_labels= ['Cumulative residual load [TWh]',    'Cumulative residual load [TWh]',   'Cumulative residual load anomaly [TWh]',   'Cumulative residual load anomaly [TWh]']
y_list  = [df_ENS_annual_sum,                   df_ENS_annual_max,                  df_ENS_annual_sum,                          df_ENS_annual_max]
y_labels= ['Summed-up ENS [MWh]',               'Max daily ENS [MWh]',              'Summed-up ENS [MWh]',                      'Max daily ENS [MWh]']
axs_idx = [(0,0), (0,1), (1,0), (1,1)]
"""

x_list  = [df_RL_annual,                        df_RL_annual]
x_labels= ['Cumulative residual load [TWh]',    'Cumulative residual load [TWh]']
y_list  = [df_ENS_annual_sum,                   df_ENS_annual_max]
y_labels= ['Total ENS [MWh]',               'Max daily ENS [MWh]']
axs_idx = [0, 1]

x_list  = [df_RL_annual,                        df_RL_annual,                       df_RL_annual,                       df_RL_annual]
x_labels= ['Cumulative residual load [TWh]',    'Cumulative residual load [TWh]',   'Cumulative residual load [TWh]',   'Cumulative residual load [TWh]']
y_list  = [df_ENS_annual_sum,                   df_ENS_annual_max,                  df_ENS_hours,                       df_ENS_days]
y_labels= ['Total ENS [MWh]',                   'Max daily ENS [MWh]',                  'Hours of ENS [h]',         'Days with ENS>0 [d]']
axs_idx = [0, 1, 2, 3]

fig, axs = plt.subplots(1, 4, figsize=(20,5))

idx = 0
#print(agg_zone+' RL  (4.1) Sev R value: '+str(lin_reg(x, y)[3]))
r_value = dict()
rho_value = dict()

for x, x_label, y, y_label, idx in zip(x_list, x_labels, y_list, y_labels, axs_idx):

    for nFOS in range(1, 16+1): # use 16 = 1 to correctly put the scatter plot on top with its label first

        FOS = nFOS

        if nFOS == 16:
            FOS = 1

        if nFOS == 1 or nFOS == 16:
            color = dt_colors[1]
            alpha = 1
            marker = '^'
            if nFOS == 1:
                label = f'outage scenario n°{FOS}'
            else:
                label = ''
        elif FOS == 2:
            color = 'grey'
            label = 'other outage scenarios'
            alpha = 0.4
            marker = '.'
        else:
            color = 'grey'
            label = ''
            alpha = 0.4
            marker = '.'

        #x = Otero_rl3_year['Severity (adapted)']
        #y = df_sumENS[FOS]

        # Linear regression
        df_reg, intercept, slope, r_value[FOS], p_value, reg_trend = lin_reg(x, y[FOS])
        # Spearman rank-order correlation coefficient
        rho_value[FOS], p_spearman = stats.spearmanr(x, y[FOS])

        axs[idx].scatter(x, y[FOS], color=color, label=label, alpha=alpha, marker=marker)
        if FOS == 1:
            axs[idx].plot(df_reg, c=color)
        axs[idx].set_ylabel(y_label)
        axs[idx].set_xlabel(x_label)
        axs[idx].legend(loc='upper left')

    # Show worst years for FOS=1 on the figure
    FOS = 1
    n_extreme_years_FOS = 5
    list_extreme_years_FOS = y[FOS].sort_values().index.to_list()[::-1][:n_extreme_years_FOS]
    for year in list_extreme_years_FOS:
        axs[idx].text(x.loc[year] * 0.997, y[FOS].loc[year], year, ha='right', color=dt_colors[1])

    # ---  Year ranking
    # Rank the years based on ENS.
    # We use a Condorcet ranking to rank the years based on the ranking of all the 15 outages scenarios (FOS)
    rankings_years_FOS = [y[FOS].sort_values().index.to_list()[::-1] for FOS in range(1, 15+1)]
    ranking_year_ENS, total_wins_ENS = condorcet_ranking(pairwise_comparison(rankings_years_FOS))
    # Rank based on dunkelflaute
    rankings_years_DF = x.sort_values().index.to_list()[::-1]
    print(f"{y_label} vs. {x_label}")
    print(f"    Most extreme years (ENS): {ranking_year_ENS[:5]}")
    print(f"    Most extreme years (dunkelflaute): {rankings_years_DF[:5]}")

    r_q50 = np.quantile([r_value[FOS] for FOS in range(1, 15+1)], 0.5)
    r_min = np.min([r_value[FOS] for FOS in range(1, 15+1)])
    r_max = np.max([r_value[FOS] for FOS in range(1, 15+1)])
    rho_q50 = np.quantile([rho_value[FOS] for FOS in range(1, 15+1)], 0.5)
    rho_min = np.min([rho_value[FOS] for FOS in range(1, 15+1)])
    rho_max = np.max([rho_value[FOS] for FOS in range(1, 15+1)])
    #axs.set_title(f'{ener_var}, {agg_zone} (Scenario {scenario_EVA}, p={p_max})\n r={np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}] (Q50 [min, max])')
    
    axs[idx].set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')
    axs[idx].set_title(zone_name, loc='left')

fig.tight_layout()
plt.savefig(f"{path_to_plot}Correlation/{figname}.pdf")




















# %%
# =================================================================
# Year ranking | Otero 
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------


scenario_EVA = 'B'

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.014
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.002
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0064
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0068
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.014

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.002
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']; p_max = 0.0104
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']; p_max = 0.0064
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04']; p_max = 0.002  # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']; p_max = 0.0064
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']; p_max = 0.0168
agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00']; p_max = 0.0172 # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00']; p_max = 0.0316  # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00']; p_max = 0.056  # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

figname = f"YearRank_Otero_ENS_Scenario{scenario_EVA}_{agg_zone}_all"


# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_RL_d  = pd.DataFrame()

df_agg_RL_d[agg_zone]  = data3_RL_d.loc[('HIST')][zones_list].sum(axis=1)

# --- Compute dunkelflaute ---

## Otero
# Find capacity thresholds
Otero_rl3_thresh,  Otero_rl3_sigma  = get_thresholds(df_agg_RL_d, 1-p_max , start_date='1982-01-01', end_date='2016-12-31', empirical=True)

Otero_rl3_days, Otero_rl3_events = detect_drought_Otero22(['HIST'],  [agg_zone], pd.concat([df_agg_RL_d], keys=['HIST'], names=['Scenario']), 
                                                          Otero_rl3_thresh, Otero_rl3_sigma, below=False)

# --- Compute Annual metrics ---

Otero_rl3_events_copy = Otero_rl3_events.copy()

# Date list, in order to get the index of each day without Feb 29th.
# If Startdate = 2004-02-28 and Duration = 2, then Enddate = 2004-03-01 instead of 2004-02-29.
date_range = pd.date_range(start='1982-01-01', end='2016-12-31')
date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]

Enddate_list = []

# Add the 'Enddate' column
for _, row in Otero_rl3_events_copy.iterrows():
    Enddate_list.append(date_range[date_range.get_loc(row['Startdate']) + row['Duration'] - 1])
Otero_rl3_events_copy['Enddate'] = Enddate_list

year_list = []
year_weight = []
for _, row in Otero_rl3_events_copy.iterrows():

    if row['Startdate'].year != row['Enddate'].year:
        year_list.append(row['Startdate'].year)
        year_end = pd.to_datetime(f'{row['Startdate'].year}-12-31')
        year_weight.append((year_end - row['Startdate']).days / row['Duration'])

    else:
        year_list.append(row['Startdate'].year)
        year_weight.append(1)
    
Otero_rl3_events_copy['Year'] = year_list
Otero_rl3_events_copy['Year weight'] = year_weight

Otero_rl3_year = pd.DataFrame(np.zeros((2016-1982+1, 2)), index=range(1982, 2016+1), columns=['Duration', 'Severity (adapted)'])

for _, row in Otero_rl3_events_copy.iterrows():

    if row['Year weight'] != 1:
        print(row['Year weight'])
        Otero_rl3_year.loc[row['Year'],'Severity (adapted)'] += row['Severity (adapted)'] * row['Year weight']
        Otero_rl3_year.loc[row['Year'] + 1,'Severity (adapted)'] += row['Severity (adapted)'] * (1 - row['Year weight'])
        Otero_rl3_year.loc[row['Year'],'Duration'] += row['Duration'] * row['Year weight']
        Otero_rl3_year.loc[row['Year'] + 1,'Duration'] += row['Duration'] * (1 - row['Year weight'])

    else:
        Otero_rl3_year.loc[row['Year'],'Severity (adapted)'] += row['Severity (adapted)']
        Otero_rl3_year.loc[row['Year'],'Duration'] += row['Duration'] * row['Year weight']


# --- Annual metrics for FOS ---

df_ENS_annual_sum = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_annual_max = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_hours = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_days = pd.DataFrame(index=range(1982, 2016+1))

for FOS in range(1, 15+1):

    df_FOS_d = pd.DataFrame()
    df_FOS_h = pd.DataFrame()
    df_FOS_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    df_FOS_h[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_hourly.pkl')[zones_list].sum(axis=1)

    # Annual sum of ENS
    df_ENS_annual_sum[FOS] = df_FOS_d.groupby(df_FOS_d.index.year).sum()

    # Yearly maximum of daily ENS
    df_daily_FOS = df_FOS_d.groupby([df_FOS_d.index.year, df_FOS_d.index.month, df_FOS_d.index.day]).sum()
    time_index = pd.date_range(start='1982-01-01', end='2016-12-31', freq='d')
    time_index = time_index[~((time_index.day == 29) & (time_index.month == 2))]
    df_max_annual_FOS = pd.DataFrame(index=time_index)
    df_max_annual_FOS[agg_zone] = df_daily_FOS[agg_zone].values
    df_ENS_annual_max[FOS] = df_max_annual_FOS.groupby(df_max_annual_FOS.index.year).max()

    # Hour of ENS per year
    df_ENS_hours[FOS] = (df_FOS_h > 0).groupby(df_FOS_h.index.year).sum()

    # Number of days with ENS > 0
    df_ENS_days[FOS] = (df_FOS_d > 0).groupby(df_FOS_d.index.year).sum()



# -------------
#  Correlations
# -------------

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')


x_list  = [Otero_rl3_year['Severity (adapted)'],    Otero_rl3_year['Severity (adapted)'],   Otero_rl3_year['Severity (adapted)'],    Otero_rl3_year['Severity (adapted)'],   
           Otero_rl3_year['Duration'],              Otero_rl3_year['Duration'],             Otero_rl3_year['Duration'],             Otero_rl3_year['Duration']]
x_labels= ['Total Severity',                    'Total Severity',                       'Total Severity',           'Total Severity',
           'Total dunkelflaute duration [d]',   'Total dunkelflaute duration [d]',      'Total dunkelflaute duration [d]',  'Total dunkelflaute duration [d]']
y_list  = [df_ENS_annual_sum,                   df_ENS_annual_max,                      df_ENS_hours,               df_ENS_days,
           df_ENS_annual_sum,                   df_ENS_annual_max,                      df_ENS_hours,               df_ENS_days]
y_labels= ['Total ENS [MWh]',                   'Max daily ENS [MWh]',                  'Hours of ENS [h]',         'Days with ENS>0 [d]',
           'Total ENS [MWh]',                   'Max daily ENS [MWh]',                  'Hours of ENS [h]',         'Days with ENS>0 [d]']
axs_idx = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]

fig, axs = plt.subplots(2, 4, figsize=(20, 10))

idx = 0
#print(agg_zone+' RL  (4.1) Sev R value: '+str(lin_reg(x, y)[3]))
r_value = dict()
rho_value = dict()

for x, x_label, y, y_label, idx in zip(x_list, x_labels, y_list, y_labels, axs_idx):

    for nFOS in range(1, 16+1): # use 16 = 1 to correctly put the scatter plot on top with its label first

        FOS = nFOS

        if nFOS == 16:
            FOS = 1

        if nFOS == 1 or nFOS == 16:
            color = dt_colors[1]
            alpha = 1
            marker = '^'
            if nFOS == 1:
                label = f'outage scenario n°{FOS}'
            else:
                label = ''
        elif FOS == 2:
            color = 'grey'
            label = 'other outage scenarios'
            alpha = 0.4
            marker = '.'
        else:
            color = 'grey'
            label = ''
            alpha = 0.4
            marker = '.'

        #x = Otero_rl3_year['Severity (adapted)']
        #y = df_sumENS[FOS]

        # Linear regression
        df_reg, intercept, slope, r_value[FOS], p_value, reg_trend = lin_reg(x, y[FOS])
        # Spearman rank-order correlation coefficient
        rho_value[FOS], p_spearman = stats.spearmanr(x, y[FOS])

        axs[idx].scatter(x, y[FOS], color=color, label=label, alpha=alpha, marker=marker)
        if FOS == 1:
            axs[idx].plot(df_reg, c=color)
        axs[idx].set_ylabel(y_label)
        axs[idx].set_xlabel(x_label)
        axs[idx].legend(loc='upper left')

    # Show worst years for FOS=1 on the figure
    FOS = 1
    n_extreme_years_FOS = 5
    list_extreme_years_FOS = y[FOS].sort_values().index.to_list()[::-1][:n_extreme_years_FOS]
    for year in list_extreme_years_FOS:
        axs[idx].text(x.loc[year] * 0.98, y[FOS].loc[year], year, ha='right', color=dt_colors[1])

    # ---  Year ranking
    # Rank the years based on ENS.
    # We use a Condorcet ranking to rank the years based on the ranking of all the 15 outages scenarios (FOS)
    rankings_years_FOS = [y[FOS].sort_values().index.to_list()[::-1] for FOS in range(1, 15+1)]
    ranking_year_ENS, total_wins_ENS = condorcet_ranking(pairwise_comparison(rankings_years_FOS))
    # Rank based on dunkelflaute
    rankings_years_DF = x.sort_values().index.to_list()[::-1]
    print(f"{y_label} vs. {x_label}")
    print(f"    Most extreme years (ENS): {ranking_year_ENS[:5]}")
    print(f"    Most extreme years (dunkelflaute): {rankings_years_DF[:5]}")

    r_q50 = np.quantile([r_value[FOS] for FOS in range(1, 15+1)], 0.5)
    r_min = np.min([r_value[FOS] for FOS in range(1, 15+1)])
    r_max = np.max([r_value[FOS] for FOS in range(1, 15+1)])
    rho_q50 = np.quantile([rho_value[FOS] for FOS in range(1, 15+1)], 0.5)
    rho_min = np.min([rho_value[FOS] for FOS in range(1, 15+1)])
    rho_max = np.max([rho_value[FOS] for FOS in range(1, 15+1)])
    #axs.set_title(f'{ener_var}, {agg_zone} (Scenario {scenario_EVA}, p={p_max})\n r={np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}] (Q50 [min, max])')
    
    axs[idx].set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')
    axs[idx].set_title(zone_name, loc='left')

fig.tight_layout()
plt.savefig(f"{path_to_plot}Correlation/{figname}.pdf")




















# %%
# =================================================================
# Year ranking | Stoop
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

scenario_EVA = 'B'

PERIOD_length_days = 1
PERIOD_cluster_days = 1

# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.014
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0176
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0064
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]; p_max = 0.004
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0124
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.0036
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']; p_max = 0.0108
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']; p_max = 0.0112
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']; p_max = 0.0256

# Agregated zones
agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00']; p_max = 0.0132 # Luxembourg is not in demand dataset
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.0036
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04']; p_max = 0.0024 # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']; p_max = 0.0316

#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00']; p_max = 0.0208 # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00']; p_max = 0.05  # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

figname = f"YearRank_Stoop_ENS_Scenario{scenario_EVA}_{agg_zone}_T{PERIOD_length_days}_Tc{PERIOD_cluster_days}_all"

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_RL_h  = pd.DataFrame()

df_agg_RL_h[agg_zone]  = data3_RL_h.loc[('HIST')][zones_list].sum(axis=1)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
    
# Find capacity thresholds
# Percentile is computed on the clustered events ("independent" events)
# interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
rl3_thresh = np.quantile(rl3_event_values, q=1-p_max, interpolation="nearest")

idx_above_thresh = np.sum(rl3_event_values > rl3_thresh)
DF_values = rl3_event_values[:idx_above_thresh]
DF_dates = rl3_event_dates[:idx_above_thresh]


# Same result structure as for Otero'22
# Note: we add the 'Duration' column for consistency, but duration is fixed to define CREDI events
Stoop_events = pd.DataFrame()

date_range = pd.date_range(start='1982-01-01', end='2016-12-31', freq='d')
date_range = date_range[~((date_range.day == 29) & (date_range.month == 2))]

Startdate_list = []
Enddate_list = []
Duration_list = []
CREDI_date = []

for date, val in zip(DF_dates, DF_values):
    # Ensure
    Enddate_list.append(date_range[date_range.get_loc(date) - 1])
    Startdate_list.append(date_range[date_range.get_loc(date) - PERIOD_length_days])
    Duration_list.append(PERIOD_length_days)
    CREDI_date.append(val)

Stoop_events['Startdate'] = Startdate_list
Stoop_events['Enddate'] = Enddate_list
Stoop_events['Duration'] = Duration_list
Stoop_events['CREDI'] = CREDI_date


year_list = []
year_weight = []
for _, row in Stoop_events.iterrows():

    if row['Startdate'].year != row['Enddate'].year:
        year_list.append(row['Startdate'].year)
        year_end = pd.to_datetime(f'{row['Startdate'].year}-12-31')
        year_weight.append((year_end - row['Startdate']).days / row['Duration'])

    else:
        year_list.append(row['Startdate'].year)
        year_weight.append(1)
    
Stoop_events['Year'] = year_list
Stoop_events['Year weight'] = year_weight

Stoop_year = pd.DataFrame(np.zeros((2016-1982+1, 2)), index=range(1982, 2016+1), columns=['Duration', 'CREDI'])

for _, row in Stoop_events.iterrows():

    if row['Year weight'] != 1:
        print(row['Year weight'])
        Stoop_year.loc[row['Year'],'CREDI'] += row['CREDI'] * row['Year weight']
        Stoop_year.loc[row['Year'] + 1,'CREDI'] += row['CREDI'] * (1 - row['Year weight'])
        Stoop_year.loc[row['Year'],'Duration'] += row['Duration'] * row['Year weight']
        Stoop_year.loc[row['Year'] + 1,'Duration'] += row['Duration'] * (1 - row['Year weight'])

    else:
        Stoop_year.loc[row['Year'],'CREDI'] += row['CREDI']
        Stoop_year.loc[row['Year'],'Duration'] += row['Duration'] * row['Year weight']



# --- Annual metrics for FOS ---

df_ENS_annual_sum = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_annual_max = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_hours = pd.DataFrame(index=range(1982, 2016+1))
df_ENS_days = pd.DataFrame(index=range(1982, 2016+1))

for FOS in range(1, 15+1):

    df_FOS_d = pd.DataFrame()
    df_FOS_h = pd.DataFrame()
    df_FOS_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    df_FOS_h[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_hourly.pkl')[zones_list].sum(axis=1)

    # Annual sum of ENS
    df_ENS_annual_sum[FOS] = df_FOS_d.groupby(df_FOS_d.index.year).sum()

    # Yearly maximum of daily ENS
    df_daily_FOS = df_FOS_d.groupby([df_FOS_d.index.year, df_FOS_d.index.month, df_FOS_d.index.day]).sum()
    time_index = pd.date_range(start='1982-01-01', end='2016-12-31', freq='d')
    time_index = time_index[~((time_index.day == 29) & (time_index.month == 2))]
    df_max_annual_FOS = pd.DataFrame(index=time_index)
    df_max_annual_FOS[agg_zone] = df_daily_FOS[agg_zone].values
    df_ENS_annual_max[FOS] = df_max_annual_FOS.groupby(df_max_annual_FOS.index.year).max()

    # Hour of ENS per year
    df_ENS_hours[FOS] = (df_FOS_h > 0).groupby(df_FOS_h.index.year).sum()

    # Number of days with ENS > 0
    df_ENS_days[FOS] = (df_FOS_d > 0).groupby(df_FOS_d.index.year).sum()



# -------------
#  Correlations
# -------------

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')

x_list  = [Stoop_year['CREDI'],                 Stoop_year['CREDI'],                Stoop_year['CREDI'],                Stoop_year['CREDI'],
           Stoop_year['Duration'],              Stoop_year['Duration'],             Stoop_year['Duration'],             Stoop_year['Duration']]
x_labels= ['Total 1-day CREDI values [MWh]',    'Total 1-day CREDI values [MWh]',   'Total 1-day CREDI values [MWh]',   'Total 1-day CREDI values [MWh]',
           'Total dunkelflaute duration',       'Total dunkelflaute duration',      'Total dunkelflaute duration',      'Total dunkelflaute duration']
y_list  = [df_ENS_annual_sum,                   df_ENS_annual_max,                  df_ENS_hours,                       df_ENS_days, 
           df_ENS_annual_sum,                   df_ENS_annual_max,                  df_ENS_hours,                       df_ENS_days]
y_labels= ['Total ENS [MWh]',                   'Max daily ENS [MWh]',              'Hours of ENS [h]',                 'Days with ENS>0 [d]', 
           'Total ENS [MWh]',                   'Max daily ENS [MWh]',              'Hours of ENS [h]',                 'Days with ENS>0 [d]']
axs_idx = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]

fig, axs = plt.subplots(2, 4, figsize=(20,10))

idx = 0
#print(agg_zone+' RL  (4.1) Sev R value: '+str(lin_reg(x, y)[3]))
r_value = dict()
rho_value = dict()

for x, x_label, y, y_label, idx in zip(x_list, x_labels, y_list, y_labels, axs_idx):

    
    for nFOS in range(1, 16+1): # use 16 = 1 to correctly put the scatter plot on top with its label first

        FOS = nFOS

        if nFOS == 16:
            FOS = 1

        if nFOS == 1 or nFOS == 16:
            color = dt_colors[1]
            alpha = 1
            marker = '^'
            if nFOS == 1:
                label = f'outage scenario n°{FOS}'
            else:
                label = ''
        elif FOS == 2:
            color = 'grey'
            label = 'other outage scenarios'
            alpha = 0.4
            marker = '.'
        else:
            color = 'grey'
            label = ''
            alpha = 0.4
            marker = '.'

        #x = Otero_rl3_year['Severity (adapted)']
        #y = df_sumENS[FOS]

        # Linear regression
        df_reg, intercept, slope, r_value[FOS], p_value, reg_trend = lin_reg(x, y[FOS])
        # Spearman rank-order correlation coefficient
        rho_value[FOS], p_spearman = stats.spearmanr(x, y[FOS])

        axs[idx].scatter(x, y[FOS], color=color, label=label, alpha=alpha, marker=marker)
        if FOS == 1:
            axs[idx].plot(df_reg, c=color)
        axs[idx].set_ylabel(y_label)
        axs[idx].set_xlabel(x_label)
        axs[idx].legend(loc='upper left')

    # Show worst years for FOS=1 on the figure
    FOS = 1
    n_extreme_years_FOS = 5
    list_extreme_years_FOS = y[FOS].sort_values().index.to_list()[::-1][:n_extreme_years_FOS]
    for year in list_extreme_years_FOS:
        axs[idx].text(x.loc[year] * 0.98, y[FOS].loc[year], year, ha='right', color=dt_colors[1])

    r_q50 = np.quantile([r_value[FOS] for FOS in range(1, 15+1)], 0.5)
    r_min = np.min([r_value[FOS] for FOS in range(1, 15+1)])
    r_max = np.max([r_value[FOS] for FOS in range(1, 15+1)])
    rho_q50 = np.quantile([rho_value[FOS] for FOS in range(1, 15+1)], 0.5)
    rho_min = np.min([rho_value[FOS] for FOS in range(1, 15+1)])
    rho_max = np.max([rho_value[FOS] for FOS in range(1, 15+1)])
    #axs.set_title(f'{ener_var}, {agg_zone} (Scenario {scenario_EVA}, p={p_max})\n r={np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}] (Q50 [min, max])')
    
    axs[idx].set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')
    axs[idx].set_title(zone_name, loc='left')

fig.tight_layout()
plt.savefig(f"{path_to_plot}Correlation/{figname}.pdf")
































# %%
# =================================================================
# Plot F-score | Otero 22 | Compare EVA scenarios A and B
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

figname = f"Validation_Otero22_ENS_scenario_A-B_{agg_zone}"



# ---------------------------------------------
# Compute data for figure
# ---------------------------------------------
start_time = time.time()


# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_gen_d = pd.DataFrame()
df_agg_dem_d = pd.DataFrame()
df_agg_RL_d  = pd.DataFrame()
df_agg_old_ENS_d = pd.DataFrame()

df_agg_gen_d[agg_zone] = data3_gen_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_d[agg_zone] = data3_dem_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_RL_d[agg_zone]  = data3_RL_d.loc[('HIST')][zones_list].sum(axis=1)

# --- end of aggregation ----

stat_df = dict()
for FOS in range(1, 15+1):

    scenario_EVA = 'A'
    df_agg_ENS_fos_d_A = pd.DataFrame()
    df_agg_ENS_fos_d_A[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    
    scenario_EVA = 'B'
    df_agg_ENS_fos_d_B = pd.DataFrame()
    df_agg_ENS_fos_d_B[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    

    p_list  = []
    for p in range(len(LWS_percs)):
        ## Otero
        # Find capacity thresholds
        Otero_rl3_thresh,  Otero_rl3_sigma  = get_thresholds(df_agg_RL_d,  RL_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)

        # Mask the data / Detect Drought days
        Otero_rl3_mask  = mask_data(df_agg_RL_d,  Otero_rl3_thresh,  False, 1, 0)

        ENS_fos_mask_A    = mask_data(df_agg_ENS_fos_d_A, 0, False, 2, 0)
        ENS_fos_mask_B    = mask_data(df_agg_ENS_fos_d_B, 0, False, 2, 0)

        # Calculate F
        Otero_rl3_stat_A  = get_f_score(ENS_fos_mask_A, Otero_rl3_mask,  beta=1)
        Otero_rl3_stat_B  = get_f_score(ENS_fos_mask_B, Otero_rl3_mask,  beta=1)

        # Create Dataframe
        p_list.append( pd.concat([Otero_rl3_stat_A, Otero_rl3_stat_B], keys=['RL A', 'RL B'],   names=['Drought type']))
        print('Done '+str(p+1)+'/'+str(len(LWS_percs)))

    stat_df[FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
    print(f'    Done FOS {FOS}')
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
pickle.dump(stat_df, open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "wb"))


#%%
# ---------------------------------------------
#  Plot only F
# ---------------------------------------------

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#gg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].



figname = f"Validation_Otero22_ENS_scenario_A-B_{agg_zone}"


# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL A', 'RL B']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)

# Percentile for peak F-score
ener_var = 'RL A'
metric = 'F'
F_max = quantiles_dict[ener_var][metric][1].max()
p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
print(f"RL3 (T=1): F_max = {F_max}, p_max = {p_max}")

fig, axs = plt.subplots(1, 1, figsize=(5, 4))

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')

axs.set_title(zone_name, loc='left')
axs.set_title(f"Method Otero'22", loc='right')


metric = 'F'
for ncolor, ener_var, label in zip(range(3), ['RL B', 'RL A'], ['Residual load (EVA B)', 'Residual load (EVA A)']):
    axs.plot(x, quantiles_dict[ener_var][metric][1], label=label,  color=dt_colors[ncolor], alpha=0.8)
    axs.fill_between(x, min_dict[ener_var][metric], max_dict[ener_var][metric], color=dt_colors[ncolor], alpha=0.5)
    #axs.plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    #axs.plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs.set_ylabel('F-Score')
axs.set_xlabel("Percentile of daily energy values")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs.legend(facecolor="white", loc='upper right', framealpha=1)
axs.grid(axis='y')
axs.set_ylim((0, 0.5))

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.pdf", dpi=300)
#plt.close()




















# %%
# =================================================================
# Plot F-score | Otero 22 | Filter out non weather days
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

nFOS_list = [3, 6, 9, 12, 15] # out of 15 FOS scenarios

scenario_EVA = 'B'
figname = f"Validation_Otero22_NFOS_ENS_Scenario{scenario_EVA}_{agg_zone}"



# ---------------------------------------------
# Compute data for figure
# ---------------------------------------------
start_time = time.time()


# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_gen_d = pd.DataFrame()
df_agg_dem_d = pd.DataFrame()
df_agg_RL_d  = pd.DataFrame()
df_agg_old_ENS_d = pd.DataFrame()

df_agg_gen_d[agg_zone] = data3_gen_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_d[agg_zone] = data3_dem_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_RL_d[agg_zone]  = data3_RL_d.loc[('HIST')][zones_list].sum(axis=1)

# --- end of aggregation ----

data3_ENS_fos = []
for FOS in range(1, 15+1):
    data3_ENS_fos.append(pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1))
df_ENS_sum_fos = sum((data3_ENS_fos[FOS] > 0) * 1 for FOS in range(15))

stat_df = dict()
for nFOS in nFOS_list:

    df_ENS_in_nFOS = pd.DataFrame()
    df_ENS_in_nFOS[agg_zone] = (df_ENS_sum_fos >= nFOS) * 1

    p_list  = []
    for p in range(len(LWS_percs)): # long computation, divide by 2 the number of threshold
        ## Otero
        # Find capacity thresholds
        Otero_rl3_thresh,  Otero_rl3_sigma  = get_thresholds(df_agg_RL_d,  RL_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)

        # Mask the data / Detect Drought days
        Otero_rl3_mask  = mask_data(df_agg_RL_d,  Otero_rl3_thresh,  False, 1, 0)

        ENS_fos_mask    = mask_data(df_ENS_in_nFOS, 0, False, 2, 0)

        # Calculate F
        Otero_rl3_stat  = get_f_score(ENS_fos_mask, Otero_rl3_mask,  beta=1)

        # Create Dataframe
        p_list.append( pd.concat([Otero_rl3_stat], keys=['RL'], names=['Drought type']))
        print('Done '+str(p+1)+'/'+str(len(LWS_percs)))

    stat_df[nFOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
print(f'    Done FOS {nFOS}')
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
pickle.dump(stat_df, open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "wb"))


#%%
# ---------------------------------------------
#  Plot only F
# ---------------------------------------------

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#gg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

figname = f"Validation_Otero22_NFOS_ENS_Scenario{scenario_EVA}_{agg_zone}"


# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[3].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

fig, axs = plt.subplots(1, 1, figsize=(5, 4))

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')

axs.set_title(zone_name, loc='left')
axs.set_title(f"Method Otero'22", loc='right')


metric = 'F'

# Create a color gradient
colors = plt.cm.viridis(np.linspace(0, 1, 15))
# Normalize the colors for the colormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
norm = Normalize(vmin=5, vmax=15)
scalar_map = ScalarMappable(norm=norm, cmap='plasma')

for nFOS in nFOS_list:
    ener_var = f'RL'
    label = f'Residual load (N={nFOS})'
    axs.plot(x, stat_df[nFOS].loc[(x, ener_var,  metric),(agg_zone)], label=label,  color=scalar_map.to_rgba(nFOS-1), alpha=0.8)

axs.set_ylabel('F-Score')
axs.set_xlabel("Percentile of daily energy values")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs.legend(facecolor="white", loc='upper right', framealpha=1)
axs.grid(axis='y')
axs.set_ylim((0, 0.6))

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.pdf", dpi=300)
#plt.close()
































# %%
# =================================================================
# Plot F-score | Otero 22 | F-beta
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

scenario_EVA = 'B'

figname = f"Validation_Otero22_ENS_scenario_{scenario_EVA}_{agg_zone}_Fbeta"



# ---------------------------------------------
# Compute data for figure
# ---------------------------------------------
start_time = time.time()


# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_gen_d = pd.DataFrame()
df_agg_dem_d = pd.DataFrame()
df_agg_RL_d  = pd.DataFrame()
df_agg_old_ENS_d = pd.DataFrame()

df_agg_gen_d[agg_zone] = data3_gen_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_d[agg_zone] = data3_dem_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_RL_d[agg_zone]  = data3_RL_d.loc[('HIST')][zones_list].sum(axis=1)

# --- end of aggregation ----

stat_df = dict()
for FOS in range(1, 15+1):

    df_agg_ENS_fos_d = pd.DataFrame()
    df_agg_ENS_fos_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)
    

    p_list  = []
    for p in range(len(LWS_percs)):
        ## Otero
        # Find capacity thresholds
        Otero_rl3_thresh,  Otero_rl3_sigma  = get_thresholds(df_agg_RL_d,  RL_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)

        # Mask the data / Detect Drought days
        Otero_rl3_mask  = mask_data(df_agg_RL_d,  Otero_rl3_thresh,  False, 1, 0)

        ENS_fos_mask    = mask_data(df_agg_ENS_fos_d, 0, False, 2, 0)

        # Calculate F
        Otero_rl3_stat_025  = get_f_score(ENS_fos_mask, Otero_rl3_mask,  beta=0.25)
        Otero_rl3_stat_05  = get_f_score(ENS_fos_mask, Otero_rl3_mask,  beta=0.5)
        Otero_rl3_stat_10  = get_f_score(ENS_fos_mask, Otero_rl3_mask,  beta=1)
        Otero_rl3_stat_20  = get_f_score(ENS_fos_mask, Otero_rl3_mask,  beta=2)
        Otero_rl3_stat_40  = get_f_score(ENS_fos_mask, Otero_rl3_mask,  beta=4)

        # Create Dataframe
        p_list.append( pd.concat([Otero_rl3_stat_025, Otero_rl3_stat_05, Otero_rl3_stat_10, Otero_rl3_stat_20, Otero_rl3_stat_40], 
                                 keys=['RL beta=0.25', 'RL beta=0.5', 'RL beta=1', 'RL beta=2', 'RL beta=4'],   names=['Drought type']))
        print('Done '+str(p+1)+'/'+str(len(LWS_percs)))

    stat_df[FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
    print(f'    Done FOS {FOS}')
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
pickle.dump(stat_df, open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "wb"))


#%%
# ---------------------------------------------
#  Plot only F
# ---------------------------------------------

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#gg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

scenario_EVA = 'B'

figname = f"Validation_Otero22_ENS_scenario_{scenario_EVA}_{agg_zone}_Fbeta"


# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL beta=0.25', 'RL beta=0.5', 'RL beta=1', 'RL beta=2', 'RL beta=4']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)

# Percentile for peak F-score
ener_var = 'RL beta=0.5'
metric = 'F'
F_max = quantiles_dict[ener_var][metric][1].max()
p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
print(f"RL3 (T=1): F_max = {F_max}, p_max = {p_max}")

fig, axs = plt.subplots(1, 1, figsize=(5, 4))

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')

axs.set_title(zone_name, loc='left')
axs.set_title(f"Method Otero'22", loc='right')


metric = 'F'
#for ncolor, ener_var, label in zip(range(5), 
#                                   ['RL beta=0.25', 'RL beta=0.5', 'RL beta=1', 'RL beta=2', 'RL beta=4'],  
#                                   [r'Residual load ($\beta=0.25$)', r'Residual load ($\beta=0.5$)', r'Residual load ($\beta=1$)', r'Residual load ($\beta=2$)', r'Residual load ($\beta=4$)']):
  
for ncolor, ener_var, label in zip(range(3), 
                                   ['RL beta=0.5', 'RL beta=1', 'RL beta=2'],  
                                   [r'Residual load ($\beta=0.5$)', r'Residual load ($\beta=1$)', r'Residual load ($\beta=2$)']):
    axs.plot(x, quantiles_dict[ener_var][metric][1], label=label,  color=dt_colors[ncolor], alpha=0.8)
    axs.fill_between(x, min_dict[ener_var][metric], max_dict[ener_var][metric], color=dt_colors[ncolor], alpha=0.5)
    #axs.plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    #axs.plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs.set_ylabel('F-Score')
axs.set_xlabel("Percentile of daily energy values")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs.legend(facecolor="white", loc='upper right', framealpha=1)
axs.grid(axis='y')
axs.set_ylim((0, 0.6))

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.pdf", dpi=300)
#plt.close()



#%%
# ---------------------------------------------
#  Plot Precision and recall
# ---------------------------------------------

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#gg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

scenario_EVA = 'B'

dataload = f"Validation_Otero22_ENS_scenario_{scenario_EVA}_{agg_zone}_Fbeta"
figname  = f"Validation_Otero22_ENS_scenario_{scenario_EVA}_{agg_zone}_Precision_Recall"


# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{dataload}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL beta=1']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()

    metric = 'precision'
    quantiles_dict[ener_var][metric] = np.quantile([
        stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values / (stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values + stat_df[FOS].loc[(x, ener_var,  'FP'),(agg_zone)].values) 
        for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
    min_dict[ener_var][metric] = np.min([
            stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values / (stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values + stat_df[FOS].loc[(x, ener_var,  'FP'),(agg_zone)].values) 
            for FOS in range(1, 15+1)], axis=0)
    max_dict[ener_var][metric] = np.max([
            stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values / (stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values + stat_df[FOS].loc[(x, ener_var,  'FP'),(agg_zone)].values) 
            for FOS in range(1, 15+1)], axis=0)
    
    metric = 'recall'
    quantiles_dict[ener_var][metric] = np.quantile([
        stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values / (stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values + stat_df[FOS].loc[(x, ener_var,  'FN'),(agg_zone)].values) 
        for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
    min_dict[ener_var][metric] = np.min([
            stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values / (stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values + stat_df[FOS].loc[(x, ener_var,  'FN'),(agg_zone)].values) 
            for FOS in range(1, 15+1)], axis=0)
    max_dict[ener_var][metric] = np.max([
            stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values / (stat_df[FOS].loc[(x, ener_var,  'TP'),(agg_zone)].values + stat_df[FOS].loc[(x, ener_var,  'FN'),(agg_zone)].values) 
            for FOS in range(1, 15+1)], axis=0)


fig, axs = plt.subplots(1, 1, figsize=(5, 4))

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')

axs.set_title(zone_name, loc='left')
axs.set_title(f"Method Otero'22", loc='right')


metric = 'precision'
#for ncolor, ener_var, label in zip(range(5), 
#                                   ['RL beta=0.25', 'RL beta=0.5', 'RL beta=1', 'RL beta=2', 'RL beta=4'],  
#                                   [r'Residual load ($\beta=0.25$)', r'Residual load ($\beta=0.5$)', r'Residual load ($\beta=1$)', r'Residual load ($\beta=2$)', r'Residual load ($\beta=4$)']):

#colors: ['limegreen', 'violet', 'dodgerblue', 'darkgreen', 'darkmagenta', 'darkblue']

ener_var = 'RL beta=1'
for ncolor, metric, label in zip(['darkmagenta', 'darkblue'],
                                 ['precision', 'recall'],  
                                 ['Precision', 'Recall']):
    axs.plot(x, quantiles_dict[ener_var][metric][1], label=label,  color=ncolor, alpha=0.8)
    axs.fill_between(x, min_dict[ener_var][metric], max_dict[ener_var][metric], color=ncolor, alpha=0.5)
    #axs.plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    #axs.plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs.set_ylabel('F-Score')
axs.set_xlabel("Percentile of daily energy values")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs.legend(facecolor="white", loc='upper right', framealpha=1)
axs.grid(axis='y')
axs.set_ylim((0, 1))

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.pdf", dpi=300)
#plt.close()




































# %%
# =================================================================
# Compute data for all TY | Otero 22 | Compare different TY 2033 and 2025 for scenario B
# =================================================================
# Dirty but quick implementation
# =================================================================

"""
# ----------------------
# Compute ENS for all TY
# ----------------------

eraa23_ens_csv    = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/ERAA_2023_ENS_hourly_data.csv'

# target year
ty = 2025

df_raw = pd.read_csv(eraa23_ens_csv, header=0)
# Use datatime and change "01/01/2025 00:00" to "2025-01-01 00:00:00"
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d/%m/%Y %H:%M')
# Generate the date range from 1982 to 2016 with hourly frequency
date_range = pd.date_range(start='1982-01-01', end='2016-12-31 23:00:00', freq='h')
filtered_date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]

for sce in ['B']:
    for fos in range(2, 15+1):
        
        df = pd.DataFrame({'Date': filtered_date_range})
        df.set_index('Date', inplace=True)

        # create one column for each available SZON zone
        for szon in df_raw['Bidding Zone'].unique():
            df[szon] = 0.
        
        # We look at a single target year (TY)
        df_sce_fos = df_raw[(df_raw['Scenario'] == f'Scenario {sce}') & (df_raw['FOS'] == fos) & (df_raw['Date'].dt.year == ty)]
        for idx, row in df_sce_fos.iterrows():
            # For target year 2028, we would to shift the date because Feb 29th is present but not dec 31st.
            # There is no ENS on dec 31st in any target year... 
            # ENS use the calendar of 2028. Consequence: ENS can occur on 02-29, but not on 12-31 --> shift the date
            is_leap_year = row['Date'].is_leap_year
            feb29_or_later = ((row['Date'].month == 2) & (row['Date'].day == 29)) | (row['Date'].month >= 3)
            correct_date = row['Date'] + dtime.timedelta(is_leap_year & feb29_or_later)
            # CY=2016 and Date=2033-01-09 11:00:00 --> Date=2016-01-09 11:00:00 
            change_year = correct_date.replace(year=row['CY'])
            df.loc[change_year, row['Bidding Zone']] = row['ENS (MWh)']

        # Uncomment to save hourly data (not needed for now)
        df.to_pickle(path_to_plot+f'Data/ERAA23_ENS_TY{ty}_Scenario{sce}_FOS{fos}_hourly.pkl')
        
        df_daily = get_daily_values(df,'sum')
        df_daily.to_pickle(path_to_plot+f'Data/ERAA23_ENS_TY{ty}_Scenario{sce}_FOS{fos}_daily.pkl')
        print(f'Saved EVA {sce} FOS {fos} TY {ty}')

"""

# -------------------------
# Compute Demand for all TY
# -------------------------
Demand_ERAA2023 = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Demand Dataset/'

# Define the time range with hourly frequency
time_index = pd.date_range(start='1982-01-01', end='2016-12-31 23:00', freq='h')
time_index = time_index[~((time_index.day == 29) & (time_index.month == 2))]
# Data fram with zone as column
df_Demand = dict()

#df_Demand[2033] = pd.read_pickle(path_to_plot+'Data/PEMMDB_demand_TY2033_hourly.pkl').loc['HIST']

for ty in [2025]:
    file = f'Demand_Timeseries_TY{ty}.xlsx'

    df_Demand[ty] = pd.DataFrame(index=time_index)
    df_Demand[ty].index.rename('Date', inplace=True)

    for zone in zones_szon:

        df_Demand[ty][zone] = 0.
        data_raw = pd.read_excel(Demand_ERAA2023+file, sheet_name=zone,header=1)

        for y in range(1982, 2016+1):
            df_Demand[ty].iloc[(y-1982)*365*24:(y+1-1982)*365*24, df_Demand[ty].columns.get_loc(zone)] = data_raw[y]




# --------------------------------------------------
# Compute Installed Capacity for all zone in TY 2025
# --------------------------------------------------


path_to_pemmdb_c  = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/PEMMDB_Installed_Capacities/' #'D:/PEMMDB/data_TY2033/01_PEMMDB/'
techs = ['SPV', 'WOF', 'WON']

SZON_intersection_Demand_IC = ['NL00', 'ITS1', 'ES00', 'DE00', 'ITCN', 'FI00', 'SI00', 'ME00', 'FR00', 'NOM1', 
                               'UKNI', 'SE01', 'RO00', 'CZ00', 'PL00', 'HU00', 'MK00', 'BA00', 'CY00', 'IE00', 
                               'NOS0', 'GR00', 'LT00', 'ITCA', 'NON1', 'ITCS', 'ITSI', 'RS00', 'UK00', 'SE03', 
                               'EE00', 'HR00', 'SE02', 'AL00', 'BE00', 'MT00', 'ITSA', 'LV00', 'GR03', 'CH00', 
                               'AT00', 'DKW1', 'PT00', 'DKE1', 'BG00', 'SK00', 'ITN1', 'SE04']

zones_szon = SZON_intersection_Demand_IC

#zones_szon = ['DE00']


data_IC = dict()

for ty_pecd3 in [2025]:

    cap_list = []
    for c in range(len(zones_szon)):
        file = 'PEMMDB_'+zones_szon[c]+'_NationalTrends_Renewables.xlsx'
        
        # Find out how many zones per country there are
        country = zones_szon[c][:-2]
        peon = get_zones([country],'PEON')
        peof_s = get_zones([country],'PEOF_slim')
        peof = get_zones([country],'PEOF')
        
        # Append 'Total', so that th Total is loaded as well
        peon.append('Total')
        peof_s.append('Total')
        peof.append('Total')
        
        # Load data
        data_raw = pd.read_excel(path_to_pemmdb_c+file,sheet_name='Zonal Evolution', header=1)#, skiprows = 33)

        ty_excel = ty_pecd3

        try:
        
            # Extract only necessary rows (zones) and sepearete into technologies
            data_won = data_raw.loc[(data_raw['Technology']=='Wind Onshore')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_excel]].set_index('PECD Zone')
            data_wof = data_raw.loc[(data_raw['Technology']=='Wind Offshore')&(data_raw['PECD Zone'].isin(peof_s))][['PECD Zone',ty_excel]].set_index('PECD Zone')
            data_spv_rt = data_raw.loc[(data_raw['Technology']=='Solar PV Rooftop')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_excel]].set_index('PECD Zone')
            data_spv_f  = data_raw.loc[(data_raw['Technology']=='Solar PV Farm')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_excel]].set_index('PECD Zone')
        
        except KeyError:

            # The PEMMDB Excel file can use TY 2027, not 2028 
            if ty_pecd3 == 2028:
                ty_excel = 2027

            # Extract only necessary rows (zones) and sepearete into technologies
            data_won = data_raw.loc[(data_raw['Technology']=='Wind Onshore')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_excel]].set_index('PECD Zone')
            data_wof = data_raw.loc[(data_raw['Technology']=='Wind Offshore')&(data_raw['PECD Zone'].isin(peof_s))][['PECD Zone',ty_excel]].set_index('PECD Zone')
            data_spv_rt = data_raw.loc[(data_raw['Technology']=='Solar PV Rooftop')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_excel]].set_index('PECD Zone')
            data_spv_f  = data_raw.loc[(data_raw['Technology']=='Solar PV Farm')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_excel]].set_index('PECD Zone')
        

        # Replace "slim" PEOF zones with extended names to avoid confusion with PEON zones
        index_mapping = dict(zip(peof_s, peof))
        data_wof.rename(index=index_mapping, inplace=True)

        # Combine SPVs
        data_spv = data_spv_rt + data_spv_f
        
        # Rename the 'Total' columns into the corresponding study zone
        data_spv.rename(index={"Total": zones_szon[c]}, inplace=True)
        data_wof.rename(index={"Total": zones_szon[c]}, inplace=True)
        data_won.rename(index={"Total": zones_szon[c]}, inplace=True)

        # For small territories, only one PEON per SZON.
        # SZON and PZON have the same name => Issue because 2 identical index.
        # Drop one occurrence of the duplicate index
        if len(data_spv.loc[zones_szon[c]]) > 1:
            data_spv = data_spv.loc[[zones_szon[c]]].head(1)
        if len(data_wof.loc[zones_szon[c]]) > 1:
            data_wof = data_wof.loc[[zones_szon[c]]].head(1)
        if len(data_won.loc[zones_szon[c]]) > 1:
            data_won = data_won.loc[[zones_szon[c]]].head(1)
        
        # Reshape, so that zones = columns and technologies = index
        tec_list = pd.concat([data_spv.T, data_wof.T, data_won.T], keys=techs, names=['Technology'])
        tec_list = tec_list.droplevel(level=1)
        
        cap_list.append(tec_list)
   
    data_IC[ty_pecd3] = pd.concat(cap_list, axis=1)



# -------------------------------------------
# Compute generation for all zone for TY 2025
# -------------------------------------------

# PEMMDB installed capacities with the right target year must be already loaded!
# If not, run the PEMMDB capacity loading cell before running this cell.
# Keep track that the same targetyear is chosen!

# Technologies / Variables of Interest and corresponding header sizes
techs = ['SPV', 'WOF', 'WON']#,'TAW'] 
aggs = ['PEON','PEOF','PEON']#,'SZON'] 
tech_agg = ['SPV/PEON', 'WOF/PEOF', 'WON/PEON']#, 'TAW/SZON']
tech_headers = [52, 52, 52]#, 52]
tech_ens = ['SPV_','_20','_30']#,'_TAW_'] 
path_to_pecd3     = 'F:/PECD3_1/'            #'D:/PECD3_1/'

"""
countries = []
for c in range(len(zones_szon)):
    countries.append(zones_szon[c][:-2])
countries = list(np.unique(countries))
"""
# Countries in `SZON_intersection_Demand_IC`
countries = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK', 
             'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 
             'LV', 'ME', 'MK', 'MT', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 
             'SE', 'SI', 'SK', 'UK']

# Some zone are absent in PECD v3.1 or the PEMMDB Demand 
zone_to_remove = ['ITCA', 'ITCA_OFF', 'FR15', 'DKBI_OFF', 'DKKF_OFF']

# We use TY 2033 for validation
ty_pecd3 = 2025


data3_list_tec_cf = [] # cf = capacity factor
data3_list_tec_ac = [] # ac = absolute capacity = generation
scenarios_3 = ['HIST'] # only HIST scenario in 3.1

for t in range(len(tech_agg)):
    # PECD 3.1 only has different HIST ENER data
    # CLIM data is the same as in 4.1
    # PROJ is not available in 3.1
    
    # only consider ENER and scenario HIST
    if techs[t]=='WOF' or techs[t]=='WON' or techs[t]=='SPV':
        domain = 'ENER'
        
        data3_list_sce_cf = []
        data3_list_sce_ac = []
        for s in range(len(scenarios_3)):
            datapath = path_to_pecd3 +techs[t]+'/'
                
            if os.path.exists(datapath):
                zones = get_zones(countries,aggs[t])  

                for zone_rm in zone_to_remove:
                    if zone_rm in zones:
                        print(f'Remove zone {zone_rm}')
                        zones.remove(zone_rm)
                
                datafile = glob.glob(datapath+'*'+tech_ens[t]+'*csv')[0]    # find specified file(s)
                    
                data3_all     = pd.read_csv(datafile, header=0)       # load the data
                data3_all['Date'] = pd.to_datetime(data3_all['Date'])
                data3_all.set_index('Date', inplace=True)
                data3_zones = data3_all[zones]                                     # only safe zones of interest
                data3_zones = data3_zones.replace(9.96921e+36,np.nan)        # replace NaN fillvalue with NaN  

               # if techs[t]=='WON':
               #     data3_caps = cf2cap(data3_zones, cap_WON, ref_year, zones)
               # elif techs[t]=='WOF':
               #     data3_caps = cf2cap(data3_zones, cap_WOF, ref_year, zones)
               # elif techs[t]=='SPV':
               #     data3_caps = cf2cap(data3_zones, cap_SPV, ref_year, zones)
                
                data3_list_sce_cf.append(data3_zones)
                data3_list_sce_ac.append(data3_zones*data_IC[ty_pecd3].loc[(techs[t]),zones])
                print('Loaded data for:  '+datafile)
                
            else:
                raise KeyError('No data in '+datapath)
        data3_list_tec_cf.append(pd.concat(data3_list_sce_cf,keys=scenarios_3,names=['Scenario']))
        data3_list_tec_ac.append(pd.concat(data3_list_sce_ac,keys=scenarios_3,names=['Scenario']))
data3_cf = pd.concat(data3_list_tec_cf,keys=techs,names=['Technology'])
data3_ac = pd.concat(data3_list_tec_ac,keys=techs,names=['Technology'])


# Sum up annual and daily values
# CF annual means
#data3_cf_mean_y = data3_cf.groupby(['Technology','Scenario',data3_cf.index.get_level_values('Date').year]).mean()
# AC annual sums
#data3_ac_sum_y = data3_ac.groupby(['Technology','Scenario',data3_ac.index.get_level_values('Date').year]).sum()
# CF daily means
#data3_cf_mean_d = get_daily_values_pecd(data3_cf,'mean')
# AC daily sums
data3_ac_sum_d = get_daily_values_pecd(data3_ac,'sum')

# Save it as pickels
#data3_cf.to_pickle(path_to_plot+f'Data/PECD3_CF_TY{ty_pecd3}_zonal_hourly.pkl')
#data3_ac.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_zonal_hourly.pkl')
#data3_cf_mean_y.to_pickle(path_to_plot+f'Data/PECD3_CF_TY{ty_pecd3}_zonal_annual.pkl')
#data3_cf_mean_d.to_pickle(path_to_plot+f'Data/PECD3_CF_TY{ty_pecd3}_zonal_daily.pkl')
#data3_ac_sum_y.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_zonal_annual.pkl')
data3_ac_sum_d.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_zonal_daily.pkl')


# Calculate zonal aggregations of Generation (for Otero et al 22 method)

data3_ac_national_h = pd.DataFrame()
data3_ac_national_d = pd.DataFrame()
data3_CF_national_h = pd.DataFrame()

zones_szon = SZON_intersection_Demand_IC

for zone_szon in zones_szon:
    # make a list of all Off- and Onshore zones of a country
    country = zone_szon[:2]
    peon_country = np.asarray(get_zones([country],'PEON'))
    peof_country = np.asarray(get_zones([country],'PEOF'))
    szon_of_peon = np.asarray(assign_SZON(peon_country, 'PEON'))
    szon_of_peof = np.asarray(assign_SZON(peof_country, 'PEOF'))
    # Get lists of PEON/PEOF for the SZON zone_szon
    peon_of_szon = peon_country[szon_of_peon==zone_szon]
    peof_of_szon = peof_country[szon_of_peof==zone_szon]
    print(f'peon_of_szon = {peon_of_szon}')
    print(f'peof_of_szon = {peof_of_szon}')
    pecd_of_szon = list(peon_of_szon) + list(peof_of_szon) 

    for zone_rm in zone_to_remove:
        if zone_rm in pecd_of_szon:
            print(f'Remove zone {zone_rm}')
            pecd_of_szon.remove(zone_rm)

    # sum up all the zones per country
    data3_ac_national_h[zone_szon] = data3_ac[pecd_of_szon].sum(axis=1)
    data3_ac_national_d[zone_szon] = data3_ac_sum_d[pecd_of_szon].sum(axis=1)
    data3_CF_national_h[zone_szon] = data3_ac[pecd_of_szon].sum(axis=1) / data_IC[ty_pecd3][pecd_of_szon].sum(axis=1) # CF = generation / IC

# Save hourly Capacity Factor at national (SZON) scale
data3_CF_national_h.to_pickle(path_to_plot+'Data/PECD3_CF_TY'+str(ty_pecd3)+'_national_hourly.pkl')

# Sum up WOF, WON and SPV
data3_ac_tsum_h = data3_ac_national_h.groupby(['Scenario','Date']).sum()
data3_ac_tsum_d = data3_ac_national_d.groupby(['Scenario','Date']).sum()

# Save RES generation at national (SZON) scale
data3_ac_tsum_h.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_national_hourly.pkl')
data3_ac_tsum_d.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_national_daily.pkl')



# %%
# =================================================================
# Plot F-score | Otero 22 | Compare different TY 2033 and 2025 for scenario B
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].



scenario_EVA = 'B'

figname = f"Validation_Otero22_ENS_scenario{scenario_EVA}_{agg_zone}_TY2025-2033"



# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()


# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
""" # OLD
ty = 2025
generation_d = get_daily_values(df_ERAA[ty][zone],'sum')
demand_d = get_daily_values(df_Demand[ty][zone],'sum')
df_agg_dem_d_2025 = pd.DataFrame()
df_agg_dem_d_2025[agg_zone] = demand_d
df_agg_gen_d_2025 = pd.DataFrame()
df_agg_gen_d_2025[agg_zone] = generation_d
df_agg_RL_d_2025 = pd.DataFrame()
df_agg_RL_d_2025[agg_zone] = demand_d - generation_d
"""



data3_dem_d_2025 = df_Demand[2025] # pd.read_pickle(path_to_data+'PEMMDB_demand_TY2025_daily.pkl')
data3_REP_d_2025 = pd.read_pickle(path_to_data+'PECD3_Generation_TY2025_national_daily.pkl')
start_date = '1982-01-01'
end_date   = '2016-12-31'
data3_cropped1_2025 = data3_REP_d_2025.query('Date>=@start_date and Date <= @end_date')
data3_gen_d_2025 = data3_cropped1_2025[~((data3_cropped1_2025.index.get_level_values(1).day == 29) & (data3_cropped1_2025.index.get_level_values(1).month == 2))]
data3_RL_d_2025 = data3_dem_d_2025 - data3_gen_d_2025

ty = 2025
df_agg_gen_d_2025 = pd.DataFrame()
df_agg_dem_d_2025 = pd.DataFrame()
df_agg_RL_d_2025  = pd.DataFrame()
df_agg_gen_d_2025[agg_zone] = data3_gen_d_2025.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_d_2025[agg_zone] = data3_dem_d_2025[zones_list].sum(axis=1)
df_agg_RL_d_2025[agg_zone]  = data3_RL_d_2025.loc[('HIST')][zones_list].sum(axis=1)




ty = 2033
df_agg_gen_d_2033 = pd.DataFrame()
df_agg_dem_d_2033 = pd.DataFrame()
df_agg_RL_d_2033  = pd.DataFrame()
df_agg_gen_d_2033[agg_zone] = data3_gen_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_d_2033[agg_zone] = data3_dem_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_RL_d_2033[agg_zone]  = data3_RL_d.loc[('HIST')][zones_list].sum(axis=1)



stat_df = dict()
for FOS in range(1, 15+1):

    ty = 2025
    df_ENS_fos_d_2025 = pd.DataFrame()
    df_ENS_fos_d_2025[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY{ty}_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)

    ty = 2033
    df_ENS_fos_d_2033 = pd.DataFrame()
    df_ENS_fos_d_2033[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY{ty}_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)

    
    p_list  = []
    for p in range(len(LWS_percs)):
        ## Otero
        # Find capacity thresholds
        Otero_rl2025_thresh,  Otero_rl2025_sigma  = get_thresholds(df_agg_RL_d_2025,  RL_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)
        Otero_rl2033_thresh,  Otero_rl2033_sigma  = get_thresholds(df_agg_RL_d_2033,  RL_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)
        
        Otero_dd2025_thresh,  Otero_dd2025_sigma  = get_thresholds(df_agg_dem_d_2025,  DD_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)
        Otero_dd2033_thresh,  Otero_dd2033_sigma  = get_thresholds(df_agg_dem_d_2033,  DD_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)

        Otero_lws2025_thresh,  Otero_lws2025_sigma  = get_thresholds(df_agg_gen_d_2025,  LWS_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)
        Otero_lws2033_thresh,  Otero_lws2033_sigma  = get_thresholds(df_agg_gen_d_2033,  LWS_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)

        # Mask the data / Detect Drought days
        Otero_rl2025_mask  = mask_data(df_agg_RL_d_2025,  Otero_rl2025_thresh,  False, 1, 0)
        Otero_rl2033_mask  = mask_data(df_agg_RL_d_2033,  Otero_rl2033_thresh,  False, 1, 0)

        Otero_dd2025_mask  = mask_data(df_agg_dem_d_2025,  Otero_dd2025_thresh,  False, 1, 0)
        Otero_dd2033_mask  = mask_data(df_agg_dem_d_2033,  Otero_dd2033_thresh,  False, 1, 0)

        Otero_lws2025_mask  = mask_data(df_agg_gen_d_2025,  Otero_lws2025_thresh,  True, 1, 0)
        Otero_lws2033_mask  = mask_data(df_agg_gen_d_2033,  Otero_lws2033_thresh,  True, 1, 0)
        
        ENS_fos_mask_2025 = mask_data(df_ENS_fos_d_2025, 0, False, 2, 0)
        ENS_fos_mask_2033 = mask_data(df_ENS_fos_d_2033, 0, False, 2, 0)

        
        # Calculate F (compared to ENS)
        Otero_rl2025_stat  = get_f_score(ENS_fos_mask_2025, Otero_rl2025_mask,  beta=1)
        Otero_rl2033_stat  = get_f_score(ENS_fos_mask_2033, Otero_rl2033_mask,  beta=1)

        Otero_dd2025_stat  = get_f_score(ENS_fos_mask_2025, Otero_dd2025_mask,  beta=1)
        Otero_dd2033_stat  = get_f_score(ENS_fos_mask_2033, Otero_dd2033_mask,  beta=1)

        Otero_lws2025_stat  = get_f_score(ENS_fos_mask_2025, Otero_lws2025_mask,  beta=1)
        Otero_lws2033_stat  = get_f_score(ENS_fos_mask_2033, Otero_lws2033_mask,  beta=1)


        # Create Dataframe
        p_list.append( pd.concat([Otero_rl2025_stat, Otero_rl2033_stat,
                                  Otero_dd2025_stat, Otero_dd2033_stat,
                                  Otero_lws2025_stat, Otero_lws2033_stat], 
                                 keys=['RL (TY 2025)', 'RL (TY 2033)',
                                       'DD (TY 2025)', 'DD (TY 2033)',
                                       'LWS (TY 2025)', 'LWS (TY 2033)'],   names=['Drought type']))

        print('Done '+str(p+1)+'/'+str(len(LWS_percs)))

    stat_df[FOS] = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
    print(f'    Done FOS {FOS}')
 
end_time = time.time()
print(f"Duration: {timedelta(seconds=end_time - start_time)}")

# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
pickle.dump(stat_df, open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "wb"))






# %%

# ---------------------------------------------
#  Plot only F
# ---------------------------------------------

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#zone = 'PL00'; agg_zone = zone; zones_list = [zone]

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].

# RL, LWS, DD
ener_variable = 'RL'; ener_variable_long = 'Residual load' 
#ener_variable = 'DD'; ener_variable_long = 'Demand' 
#ener_variable = 'LWS'; ener_variable_long = 'RES' 

scenario_EVA = 'B'

figname = f"Validation_Otero22_ENS_scenario{scenario_EVA}_{agg_zone}_TY2025-2033"



# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in [f'{ener_variable} (TY 2025)', f'{ener_variable} (TY 2033)']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)

# Percentile for peak F-score
"""
ener_var = 'RL'
metric = 'F'
F_max = quantiles_dict[ener_var][metric][1].max()
p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
print(f"RL3 (T=1): F_max = {F_max}, p_max = {p_max}")
"""

fig, axs = plt.subplots(1, 1, figsize=(5, 4))

if agg_zone == 'DE00':
    zone_name = "Germany"
elif agg_zone == 'FR00':
    zone_name = "France"
elif agg_zone == 'NL00':
    zone_name = "the Netherlands"
elif agg_zone == 'BE00':
    zone_name = "Belgium"
elif agg_zone == 'PL00':
    zone_name = "Poland"
elif agg_zone == 'NO':
    zone_name = "Norway"
elif agg_zone == 'UK':
    zone_name = "the United Kingdom"
elif agg_zone == 'DK':
    zone_name = "Denmark"
elif agg_zone == 'SE':
    zone_name = "Sweden"
elif agg_zone == 'IT':
    zone_name = "Italy"
elif agg_zone == 'CWE':
    zone_name = "Central Western\nEurope"
elif agg_zone == 'NWE':
    zone_name = "North Western\nEurope"
elif agg_zone == 'CoreRegion':
    zone_name = "Core Region"
elif agg_zone == 'CSA':
    zone_name = "Continental\nSynchronous Area"
else:
    zone_name = ""
    print('Warning: Set the title!')

axs.set_title(zone_name, loc='left')
axs.set_title(f"Method Otero'22", loc='right')

# Event time series

metric = 'F'
for ncolor, ener_var, label in zip(range(2),
                                   [f'{ener_variable} (TY 2033)', f'{ener_variable} (TY 2025)'], 
                                   [f'{ener_variable_long} (2033)', f'{ener_variable_long} (2025)']):
    axs.plot(x, quantiles_dict[ener_var][metric][1], label=label, alpha=0.8, color=dt_colors[ncolor])
    axs.fill_between(x, min_dict[ener_var][metric], max_dict[ener_var][metric], alpha=0.5, color=dt_colors[ncolor])
    #axs.plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    #axs.plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs.set_ylabel('F-Score')
axs.set_xlabel("Percentile of daily energy values")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs.legend(facecolor="white", loc='upper right', framealpha=1)
axs.grid(axis='y')
axs.set_ylim(0, 0.5)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}_{ener_variable}.pdf", dpi=300)
#plt.close()























#%%
# ===================================================================
# HWRW 1st week | Compare energy variables and methods
# ===================================================================

# --------------------------------------------------------------
# User defined parameters
# --------------------------------------------------------------
var_short_name = 'RL'
zone = 'DE00'
PECD_version = 'PECDv3'
Climatology_type = 'HWRW'; RollingWindow = 9 # weeks
#Climatology_type = 'HRW'; RollingWindow = 40 # days

# --------------------------------------------------------------
# Figure
# --------------------------------------------------------------

params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# Define color palette
# Demand
colour_dem = 'burlywood' # 0.03
colour_dem_clim = 'orange' # 1
colour_dem_hrw = 'tab:red' # 0.7

# Wind + solar generation
colour_gen = '#8cc6ad' # https://www.colorhexa.com/
colour_gen_clim = '#3ac98b'
colour_gen_hrw = '#1e6f4c'

# Residual load
colour_RL = 'skyblue' # 0.03
colour_RL_clim = 'lightsteelblue' # 1
colour_RL_hrw = 'dodgerblue' # 0.7

if var_short_name == 'RL':
    var_long_name = 'Residual load'
    ds_choice = data3_RL_h.loc[('HIST')]
    colour = colour_RL
    colour_clim = colour_RL_clim 
    colour_hrw = colour_RL_hrw
elif var_short_name == 'Demand':
    var_long_name = 'Demand'
    ds_choice = data3_dem_h.loc[('HIST')]
    colour = colour_dem
    colour_clim = colour_dem_clim 
    colour_hrw = colour_dem_hrw
elif var_short_name == 'REP':
    var_long_name = 'Wind & Solar'
    ds_choice = data3_gen_h.loc[('HIST')]
    colour = colour_gen
    colour_clim = colour_gen_clim 
    colour_hrw = colour_gen_hrw

# Set the data + hourly climatology
# ----------- new code 

# Set the data + hourly climatology
ds_plot = ds_choice.to_xarray()[[zone]] / 1000 # MW -> GW

# ---------------------
# Define climatology

ds_clim_hourly = Climatology_Hourly(ds_plot)

if Climatology_type == "HWRW":
    ds_clim_HWRW, MOH = Climatology_Hourly_Weekly_Rolling(InputDataSet=ds_plot, RollingWindow=RollingWindow, calendar2018=True)
elif Climatology_type == "HRW":
    ds_clim_HRW, MOH = Climatology_Hourly_Rolling(ds_plot, RollingWindow=RollingWindow)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

fig.autofmt_xdate()

# show years
year_dates = pd.date_range('1982-01-01', periods=8760, freq='1h')

### First set is the data + hourly clim

# we want to see all years
for year in np.arange(start=1982,stop=2016):
    
    # Show the data for all the years
    axes.plot(year_dates, ds_plot[zone].sel(Date=slice(str(year)+'-01-01', str(year)+'-12-31')), color=colour, alpha=0.3)

# show the hourly clim with all the years, only 13:00 for solar 
#axes.plot(year_dates, ds_clim_hourly[zone], color=colour_clim, alpha=1)
#axesS.plot(year_dates[13:8760:24], ds_ClimSPV.Hourly[13:8760:24], color=colour_solar_clim, alpha=1)


### show the hourly clim with other climatologies  


axes.plot(year_dates, ds_clim_hourly[zone], label='Initial climatology', color='grey', alpha=0.7, linewidth=2)
if Climatology_type == "HWRW":
    axes.plot(year_dates, ds_clim_HWRW[zone], label=f'HWRW = {RollingWindow} weeks', color=colour_hrw, alpha=0.7, linewidth=2)
elif Climatology_type == "HRW":
    axes.plot(year_dates, ds_clim_HRW[zone], label=f'HRW = {RollingWindow} days', color=colour_hrw, alpha=0.7, linewidth=2)

# formate the date-axis 
#axes.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3,6,9,12)))
#axes.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# set the legend and labels
axes.legend(fontsize='medium')


if zone == 'DE00':
    zone_name = "Germany"
else:
    print('Warning: Set the title!')

if Climatology_type == 'HWRW' and var_short_name == 'Demand': panel_label = 'a)'
if Climatology_type == 'HWRW' and var_short_name == 'REP': panel_label = 'b)'
if Climatology_type == 'HWRW' and var_short_name == 'RL': panel_label = 'c)'
if Climatology_type == 'HRW' and var_short_name == 'Demand': panel_label = 'd)'
if Climatology_type == 'HRW' and var_short_name == 'REP': panel_label = 'e)'
if Climatology_type == 'HRW' and var_short_name == 'RL': panel_label = 'f)'

axes.text(0.02, 0.97, panel_label, fontsize=14, transform=axes.transAxes, va='top')

# format labels
axes.set_title(zone_name, loc='left')
axes.set_title(f'{var_long_name} (all hours)', loc='right')
axes.set_ylabel('GW')

axes.set(xlim=(year_dates[0], year_dates[7*24]))
plt.tight_layout()

plt.savefig(path_to_plot+f'Sensitivity/Sensitivity_{Climatology_type}_{RollingWindow}_{PECD_version}_{var_short_name}_{zone}.pdf')

plt.show()





















#%%
# ===================================================================
# HWRW sensitivity | Compare window size (3 to 9 weeks)
# ===================================================================


# ----------------------------------------------------------------
# User defined parameters
# ----------------------------------------------------------------

var_short_name = 'RL'
zone = 'DE00'
PECD_version = 'PECDv3'

hour = 13

# ----------------------------------------------------------------
# Figure (> 5 min)
# ----------------------------------------------------------------


params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# Define color palette
# Demand
colour_dem = 'burlywood' # 0.03
colour_dem_clim = 'orange' # 1
colour_dem_hrw = 'tab:red' # 0.7

# Wind + solar generation
colour_gen = '#8cc6ad' # https://www.colorhexa.com/
colour_gen_clim = '#3ac98b'
colour_gen_hrw = '#1e6f4c'

# Residual load
colour_RL = 'skyblue' # 0.03
colour_RL_clim = 'lightsteelblue' # 1
colour_RL_hrw = 'dodgerblue' # 0.7

if var_short_name == 'RL':
    var_long_name = 'Residual load'
    ds_choice = data3_RL_h.loc[('HIST')]
    colour = colour_RL
    colour_clim = colour_RL_clim 
    colour_hrw = colour_RL_hrw
elif var_short_name == 'Demand':
    var_long_name = 'Demand'
    ds_choice = data3_dem_h.loc[('HIST')]
    colour = colour_dem
    colour_clim = colour_dem_clim 
    colour_hrw = colour_dem_hrw
elif var_short_name == 'REP':
    var_long_name = 'Wind & Solar'
    ds_choice = data3_gen_h.loc[('HIST')]
    colour = colour_gen
    colour_clim = colour_gen_clim 
    colour_hrw = colour_gen_hrw

# Set the data + hourly climatology
ds_plot = ds_choice.to_xarray() / 1000 # MW -> GW

# Define climatology
ds_clim_hourly = Climatology_Hourly(ds_plot)
ds_clim_HWRW3, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=3)
ds_clim_HWRW5, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=5)
ds_clim_HWRW7, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=7)
ds_clim_HWRW9, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=9)

"""
ds_clim_HWRW11, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=11)
ds_clim_HWRW15, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=15)
ds_clim_HWRW19, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=19)
ds_clim_HWRW23, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=23)
"""

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

fig.autofmt_xdate()

# show years
year_dates = pd.date_range('1982-01-01', periods=8760, freq='1h')

### First set is the data + hourly clim

# we want to see all years
for year in np.arange(start=1982,stop=2016):
    
    # Show the data for all the years
    ax.plot(year_dates[hour:8760:24], ds_plot[zone].sel(Date=slice(str(year)+'-01-01', str(year)+'-12-31'))[hour:8760:24], color=colour, alpha=0.1)
    ax.plot(year_dates[hour:8760:24], ds_plot[zone].sel(Date=slice(str(year)+'-01-01', str(year)+'-12-31'))[hour:8760:24], color=colour, alpha=0.1)

# show the hourly clim with all the years, only 13:00 for solar 
#axes.plot(year_dates, ds_clim_hourly[zone], color=colour_clim, alpha=1)
#axesS.plot(year_dates[13:8760:24], ds_ClimSPV.Hourly[13:8760:24], color=colour_solar_clim, alpha=1)

### show the hourly clim with other climatologies  

ax.plot(year_dates[hour:8760:24], ds_clim_hourly[zone][hour:8760:24], label='Initial climatology', color='grey', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW3[zone][hour:8760:24], label='HWRW = 3 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW5[zone][hour:8760:24], label='HWRW = 5 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW7[zone][hour:8760:24], label='HWRW = 7 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW9[zone][hour:8760:24], label='HWRW = 9 weeks', alpha=0.7)

"""
ax.plot(year_dates[hour:8760:24], ds_clim_hourly[zone][hour:8760:24], label='Initial climatology', color='grey', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW11[zone][hour:8760:24], label='HWRW = 11 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW15[zone][hour:8760:24], label='HWRW = 15 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW19[zone][hour:8760:24], label='HWRW = 19 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW23[zone][hour:8760:24], label='HWRW = 23 weeks', alpha=0.7)
"""

#axes[1].plot(year_dates[hour:8760:24], ds_clim_HWRW13[zone][hour:8760:24], label='13 weeks', color='black', alpha=0.7)

# formate the date-axis 
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(12+1)))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# set the legend and labels
ax.legend(fontsize='medium')
ax.legend(fontsize='medium')


if zone == 'DE00':
    zone_name = "Germany"
else:
    print('Warning: Set the title!')
ax.set_title(zone_name, loc='left')
ax.set_title(f'{var_long_name} (at {hour}:00)', loc='right')
ax.set_ylabel('GW')

ax.set(xlim=(year_dates[0:8760:24][0], year_dates[0:8760:24][-1]))

plt.tight_layout()

plt.savefig(path_to_plot+f'Sensitivity/Sensitivity_HWRW_3-9W_{PECD_version}_{var_short_name}_{zone}.pdf')

# make it look better

plt.show()


#%%
# ------------------------------------------
# HWRW 1st week
# ------------------------------------------

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

fig.autofmt_xdate()

# show years
year_dates = pd.date_range('1982-01-01', periods=8760, freq='1h')

### First set is the data + hourly clim

# we want to see all years
for year in np.arange(start=1982,stop=2016):
    
    # Show the data for all the years
    axes.plot(year_dates, ds_plot[zone].sel(Date=slice(str(year)+'-01-01', str(year)+'-12-31')), color=colour, alpha=0.3)

# show the hourly clim with all the years, only 13:00 for solar 
#axes.plot(year_dates, ds_clim_hourly[zone], color=colour_clim, alpha=1)
#axesS.plot(year_dates[13:8760:24], ds_ClimSPV.Hourly[13:8760:24], color=colour_solar_clim, alpha=1)


### show the hourly clim with other climatologies  


axes.plot(year_dates, ds_clim_hourly[zone], label='Initial climatology', color='grey', alpha=0.7, linewidth=2)
axes.plot(year_dates, ds_clim_HWRW3[zone], label='HWRW = 3 weeks', alpha=0.7, linewidth=2)
axes.plot(year_dates, ds_clim_HWRW5[zone], label='HWRW = 5 weeks', alpha=0.7, linewidth=2)
axes.plot(year_dates, ds_clim_HWRW7[zone], label='HWRW = 7 weeks', alpha=0.7, linewidth=2)
axes.plot(year_dates, ds_clim_HWRW9[zone], label='HWRW = 9 weeks', alpha=0.7, linewidth=2)

# formate the date-axis 
#axes.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3,6,9,12)))
#axes.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# set the legend and labels
axes.legend(fontsize='medium')


if zone == 'DE00':
    zone_name = "Germany"
else:
    print('Warning: Set the title!')

# format labels
axes.set_title(zone_name, loc='left')
axes.set_title(f'{var_long_name} (all hours)', loc='right')
axes.set_ylabel('GW')

axes.set(xlim=(year_dates[0], year_dates[7*24]))
plt.tight_layout()

plt.savefig(path_to_plot+f'Sensitivity/Sensitivity_HWRW_3-9W_Week1_{PECD_version}_{var_short_name}_{zone}.pdf')


plt.show()


































#%%
# ===================================================================
# HWRW sensitivity | Compare window size (11 to 23 weeks)
# ===================================================================


# ----------------------------------------------------------------
# User defined parameters
# ----------------------------------------------------------------

var_short_name = 'RL'
zone = 'DE00'
PECD_version = 'PECDv3'

hour = 13

# ----------------------------------------------------------------
# Figure (> 5 min)
# ----------------------------------------------------------------


params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# Define color palette
# Demand
colour_dem = 'burlywood' # 0.03
colour_dem_clim = 'orange' # 1
colour_dem_hrw = 'tab:red' # 0.7

# Wind + solar generation
colour_gen = '#8cc6ad' # https://www.colorhexa.com/
colour_gen_clim = '#3ac98b'
colour_gen_hrw = '#1e6f4c'

# Residual load
colour_RL = 'skyblue' # 0.03
colour_RL_clim = 'lightsteelblue' # 1
colour_RL_hrw = 'dodgerblue' # 0.7

if var_short_name == 'RL':
    var_long_name = 'Residual load'
    ds_choice = data3_RL_h.loc[('HIST')]
    colour = colour_RL
    colour_clim = colour_RL_clim 
    colour_hrw = colour_RL_hrw
elif var_short_name == 'Demand':
    var_long_name = 'Demand'
    ds_choice = data3_dem_h.loc[('HIST')]
    colour = colour_dem
    colour_clim = colour_dem_clim 
    colour_hrw = colour_dem_hrw
elif var_short_name == 'REP':
    var_long_name = 'Wind & Solar'
    ds_choice = data3_gen_h.loc[('HIST')]
    colour = colour_gen
    colour_clim = colour_gen_clim 
    colour_hrw = colour_gen_hrw

# Set the data + hourly climatology
ds_plot = ds_choice.to_xarray() / 1000 # MW -> GW

# Define climatology
ds_clim_hourly = Climatology_Hourly(ds_plot)

"""
ds_clim_HWRW3, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=3)
ds_clim_HWRW5, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=5)
ds_clim_HWRW7, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=7)
ds_clim_HWRW9, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=9)
"""

ds_clim_HWRW11, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=11)
ds_clim_HWRW15, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=15)
ds_clim_HWRW19, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=19)
ds_clim_HWRW23, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot, RollingWindow=23)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

fig.autofmt_xdate()

# show years
year_dates = pd.date_range('1982-01-01', periods=8760, freq='1h')

### First set is the data + hourly clim

# we want to see all years
for year in np.arange(start=1982,stop=2016):
    
    # Show the data for all the years
    ax.plot(year_dates[hour:8760:24], ds_plot[zone].sel(Date=slice(str(year)+'-01-01', str(year)+'-12-31'))[hour:8760:24], color=colour, alpha=0.1)
    ax.plot(year_dates[hour:8760:24], ds_plot[zone].sel(Date=slice(str(year)+'-01-01', str(year)+'-12-31'))[hour:8760:24], color=colour, alpha=0.1)

# show the hourly clim with all the years, only 13:00 for solar 
#axes.plot(year_dates, ds_clim_hourly[zone], color=colour_clim, alpha=1)
#axesS.plot(year_dates[13:8760:24], ds_ClimSPV.Hourly[13:8760:24], color=colour_solar_clim, alpha=1)

### show the hourly clim with other climatologies  
"""
ax.plot(year_dates[hour:8760:24], ds_clim_hourly[zone][hour:8760:24], label='Initial climatology', color='grey', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW3[zone][hour:8760:24], label='HWRW = 3 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW5[zone][hour:8760:24], label='HWRW = 5 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW7[zone][hour:8760:24], label='HWRW = 7 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW9[zone][hour:8760:24], label='HWRW = 9 weeks', alpha=0.7)
"""

ax.plot(year_dates[hour:8760:24], ds_clim_hourly[zone][hour:8760:24], label='Initial climatology', color='grey', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW11[zone][hour:8760:24], label='HWRW = 11 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW15[zone][hour:8760:24], label='HWRW = 15 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW19[zone][hour:8760:24], label='HWRW = 19 weeks', alpha=0.7)
ax.plot(year_dates[hour:8760:24], ds_clim_HWRW23[zone][hour:8760:24], label='HWRW = 23 weeks', alpha=0.7)


#axes[1].plot(year_dates[hour:8760:24], ds_clim_HWRW13[zone][hour:8760:24], label='13 weeks', color='black', alpha=0.7)

# formate the date-axis 
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(12+1)))
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# set the legend and labels
ax.legend(fontsize='medium')
ax.legend(fontsize='medium')


if zone == 'DE00':
    zone_name = "Germany"
else:
    print('Warning: Set the title!')
ax.set_title(zone_name, loc='left')
ax.set_title(f'{var_long_name} (at {hour}:00)', loc='right')
ax.set_ylabel('GW')

ax.set(xlim=(year_dates[0:8760:24][0], year_dates[0:8760:24][-1]))

plt.tight_layout()

plt.savefig(path_to_plot+f'Sensitivity/Sensitivity_HWRW_11-23W_{PECD_version}_{var_short_name}_{zone}.pdf')

# make it look better

plt.show()



#%%
# ---------------------------------------------------------------
# HWRW 1st week
# ---------------------------------------------------------------

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

fig.autofmt_xdate()

# show years
year_dates = pd.date_range('1982-01-01', periods=8760, freq='1h')

### First set is the data + hourly clim

# we want to see all years
for year in np.arange(start=1982,stop=2016):
    
    # Show the data for all the years
    axes.plot(year_dates, ds_plot[zone].sel(Date=slice(str(year)+'-01-01', str(year)+'-12-31')), color=colour, alpha=0.3)

# show the hourly clim with all the years, only 13:00 for solar 
#axes.plot(year_dates, ds_clim_hourly[zone], color=colour_clim, alpha=1)
#axesS.plot(year_dates[13:8760:24], ds_ClimSPV.Hourly[13:8760:24], color=colour_solar_clim, alpha=1)


### show the hourly clim with other climatologies  

axes.plot(year_dates, ds_clim_hourly[zone], label='Initial climatology', color='grey', alpha=0.7, linewidth=2)
axes.plot(year_dates, ds_clim_HWRW11[zone], label='HWRW = 11 weeks', alpha=0.7, linewidth=2)
axes.plot(year_dates, ds_clim_HWRW15[zone], label='HWRW = 15 weeks', alpha=0.7, linewidth=2)
axes.plot(year_dates, ds_clim_HWRW19[zone], label='HWRW = 19 weeks', alpha=0.7, linewidth=2)
axes.plot(year_dates, ds_clim_HWRW23[zone], label='HWRW = 23 weeks', alpha=0.7, linewidth=2)

# formate the date-axis 
#axes.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3,6,9,12)))
#axes.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# set the legend and labels
axes.legend(fontsize='medium')


if zone == 'DE00':
    zone_name = "Germany"
else:
    print('Warning: Set the title!')

# format labels
axes.set_title(zone_name, loc='left')
axes.set_title(f'{var_long_name} (all hours)', loc='right')
axes.set_ylabel('GW')

axes.set(xlim=(year_dates[0], year_dates[7*24]))
plt.tight_layout()

plt.savefig(path_to_plot+f'Sensitivity/Sensitivity_HWRW_11-23W_Week1_{PECD_version}_{var_short_name}_{zone}.pdf')


plt.show()






