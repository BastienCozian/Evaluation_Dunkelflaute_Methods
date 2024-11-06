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

from Dunkelflaute_function_library import get_zones, get_thresholds, detect_drought_Otero22, mask_data, get_f_score, \
    get_df_timerange, lin_reg, detect_drought_Li21, mask_df_by_entries, detect_DF_Li21, mask_Li21
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

# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']

scenario_EVA = 'B'

figname = f"Validation_Stoop24_ENS_scenario{scenario_EVA}_{agg_zone}_T1-7d"


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

PERIOD_length_days = 7
PERIOD_cluster_days = 6
# Get CREDI events
T7_CREDI_event, T7_event_dates, \
    T7_event_values = get_CREDI_events(df_agg_RL_h, agg_zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
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
        T7_thresh = np.quantile(T7_event_values, q=RL_percs[p], interpolation="nearest")

        # Mask the data / Detect Drought days
        T1_mask = mask_CREDI(T1_event_dates, T1_event_values, T1_thresh, PERIOD_length_days=1, zone=agg_zone, extreme_is_high=True)
        T3_mask = mask_CREDI(T3_event_dates, T3_event_values, T3_thresh, PERIOD_length_days=3, zone=agg_zone, extreme_is_high=True)
        T5_mask = mask_CREDI(T5_event_dates, T5_event_values, T5_thresh, PERIOD_length_days=5, zone=agg_zone, extreme_is_high=True)
        T7_mask = mask_CREDI(T7_event_dates, T7_event_values, T7_thresh, PERIOD_length_days=7, zone=agg_zone, extreme_is_high=True)
        ENS_fos_mask = mask_data(df_agg_ENS_fos_d, 0, False, 2, 0)

        # Calculate F
        T1_stat  = get_f_score(ENS_fos_mask, T1_mask,  beta=1)
        T3_stat  = get_f_score(ENS_fos_mask, T3_mask,  beta=1)
        T5_stat  = get_f_score(ENS_fos_mask, T5_mask,  beta=1)
        T7_stat  = get_f_score(ENS_fos_mask, T7_mask,  beta=1)

        # Create Dataframe
        p_list.append( pd.concat([T1_stat, T3_stat, T5_stat, T7_stat], keys=['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)', 'RL3 (T=7)'],   names=['Drought type']))
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

# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']

scenario_EVA = 'B'

figname = f"Validation_Stoop24_ENS_scenario{scenario_EVA}_{agg_zone}_T1-7d"


# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)', 'RL3 (T=7)']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(agg_zone)] for FOS in range(1, 15+1)], axis=0)


# Percentile for peak F-score
for ener_var in ['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)', 'RL3 (T=7)']:
    metric = 'F'
    F_max = quantiles_dict[ener_var][metric][1].max()
    p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
    print(f"{ener_var}: F_max = {F_max}, p_max = {p_max}")



fig, axs = plt.subplots(1, 1, figsize=(5, 4))

if agg_zone == 'DE00':
    axs.set_title(f"Method Stoop'23 for Germany")
else:
    print('Warning: Set the title!')

# Event time series

metric = 'F'
for ncolor, ener_var, label in zip(range(4), ['RL3 (T=1)', 'RL3 (T=3)', 'RL3 (T=5)', 'RL3 (T=7)'], ['Residual load (T=1d)', 'Residual load (T=3d)', 'Residual load (T=5d)', 'Residual load (T=7d)']):
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




































#%% 
# =================================================================
# Correlation for CREDI | T = 3, 5, 7 days
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# Clustering period. If PERIOD_length_days < PERIOD_cluster_days, then there may be overlaps. We authorize up to 25% overlap.
## T = 1
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]; PERIOD_length_days = 1; PERIOD_cluster_days = 1; p_max = 0.014
## T = 3
zone = 'DE00'; agg_zone = zone; zones_list = [zone]; PERIOD_length_days = 3; PERIOD_cluster_days = 3; p_max = 0.0144
## T = 5
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]; PERIOD_length_days = 5; PERIOD_cluster_days = 4; p_max = 0.0204
## T = 7
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]; PERIOD_length_days = 7; PERIOD_cluster_days = 6; p_max = 0.0172



scenario_EVA = 'B'



# --- Old Percentile for peak F-score. Computed in `DF_Validation_Stoop.py` ---
# FOR DE00
#p_max = 0.014   # T=1
#p_max = 0.0144  # T=3
#p_max = 0.0204  # T=5
#p_max = 0.0172  # T=7



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



# %%
# ---------------------------------------------
# Plot only RL
# ---------------------------------------------

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
    # Spearman rank-order correlation coefficient
    rho_value[FOS], p_spearman = stats.spearmanr(x, y)
    
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
ax.set_title(r'$T=$'+f'{PERIOD_length_days} days\n'+r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]; '+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]')

ax.set_ylabel('Summed-up ENS [GWh]')
ax.set_xlabel('CREDI [GWh]')
ax.legend(facecolor="white", loc='upper left', framealpha=1)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Correlation/{figname}.pdf")
#plt.close()






































# %%
