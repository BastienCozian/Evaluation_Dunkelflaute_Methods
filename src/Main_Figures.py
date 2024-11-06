"""
Plots to specificaly investigate the size of Europe

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
# Plot F-score | Li 21 | Compare RL, DD, RES for scenario B
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# No "aggregated zone" because we cannot sum capacity factor. For this, we would need to devide by the installed capacity
zone = 'DE00'
#agg_zone = zone; zones_list = [zone]

scenario_EVA = 'B'

figname = f"Validation_Li21_ENS_scenario{scenario_EVA}_{zone}"



# ---------------------------------------------
# Compute data for figure (~ 15s per variable and percentile)
# ---------------------------------------------
start_time = time.time()

# Percentile thresholds to investigate
CF_threshs = np.linspace(0.05, 0.6, 81)


# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types



stat_df = dict()
for FOS in range(1, 15+1):

    df_agg_ENS_fos_d = pd.DataFrame()
    df_agg_ENS_fos_d[[zone]] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[[zone]]
    
    p_list  = []
    for p in range(len(CF_threshs)):

        cfd3_events = detect_DF_Li21(data3_CF_h[[zone]], zone, CF_threshs[p])
        # Take only event with capacity factor below the threshold for at least 24h (cf. Li et al (2021))
        cfd3_24h_events = cfd3_events[cfd3_events['Duration'] >= 24]

        # Mask the data / Detect Drought days
        cf3_mask = mask_Li21(cfd3_24h_events, zone)
        ENS_fos_mask = mask_data(df_agg_ENS_fos_d, 0, False, 2, 0)

        # Calculate F (compared to ENS)
        cf3_stat = get_f_score(ENS_fos_mask, cf3_mask,  beta=1)

        # Create Dataframe
        p_list.append( pd.concat([cf3_stat], keys=['CF'],   names=['Drought type']))
        print('Done '+str(p+1)+'/'+str(len(CF_threshs)))

    stat_df[FOS] = pd.concat(p_list, keys=CF_threshs, names=['Percentiles'])
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


# Load data
stat_df = pickle.load(open(f"{path_to_plot}Plot_data/{figname}_stats.pkl", "rb"))

x=stat_df[1].index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)

quantiles_dict = dict()
min_dict = dict()
max_dict = dict()
for ener_var in ['CF']:
    quantiles_dict[ener_var] = dict()
    min_dict[ener_var] = dict()
    max_dict[ener_var] = dict()
    for metric in ['F', 'TP', 'TN', 'FP', 'FN']:
        quantiles_dict[ener_var][metric] = np.quantile([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], [0.25, 0.5, 0.75], axis=0)
        min_dict[ener_var][metric] = np.min([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], axis=0)
        max_dict[ener_var][metric] = np.max([stat_df[FOS].loc[(x, ener_var,  metric),(zone)] for FOS in range(1, 15+1)], axis=0)

# Percentile for peak F-score
ener_var = 'CF'
metric = 'F'
F_max = quantiles_dict[ener_var][metric][1].max()
p_max = x[quantiles_dict[ener_var][metric][1] == F_max][0]
print(f"CF: F_max = {F_max}, p_max = {p_max}")

fig, axs = plt.subplots(1, 1, figsize=(5, 4))

if zone == 'DE00':
    axs.set_title(f"Method Li'21 for Germany")
else:
    print('Warning: Set the title!')

metric = 'F'
for ncolor, ener_var, label in zip(range(1), ['CF'], ['Capacity factor']):
    axs.plot(x, quantiles_dict[ener_var][metric][1], label=label,  color=dt_colors[ncolor], alpha=0.8)
    axs.fill_between(x, min_dict[ener_var][metric], max_dict[ener_var][metric], color=dt_colors[ncolor], alpha=0.5)
    #axs.plot(x, min_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
    #axs.plot(x, max_dict[ener_var][metric], linestyle='dashed', color=dt_colors[ncolor], alpha=0.8)
axs.set_ylabel('F-Score')
axs.set_xlabel("Capacity factor threshold [GW/GW]")
#axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
axs.legend(facecolor="white", loc='upper right', framealpha=1)
axs.grid(axis='y')
axs.set_ylim((0, 0.5))

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.pdf", dpi=300)
#plt.close()




















#%%
# =================================================================
# Plot F-score | Otero 22 | Compare RL, DD, RES for scenario B
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

# --- Agregated zones
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00'] # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00'] # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']

scenario_EVA = 'B'

figname = f"Validation_Otero22_ENS_scenario{scenario_EVA}_{agg_zone}"



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
        Otero_dd3_thresh,  Otero_dd3_sigma  = get_thresholds(df_agg_dem_d,  DD_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)
        Otero_lws3_thresh,  Otero_lws3_sigma  = get_thresholds(df_agg_gen_d,  LWS_percs[p] , start_date='1982-01-01', end_date='2016-12-31', empirical=True)

        # Mask the data / Detect Drought days
        Otero_rl3_mask  = mask_data(df_agg_RL_d,  Otero_rl3_thresh,  False, 1, 0)
        Otero_dd3_mask  = mask_data(df_agg_dem_d,  Otero_dd3_thresh,  False, 1, 0)
        Otero_lws3_mask = mask_data(df_agg_gen_d,  Otero_lws3_thresh,  True, 1, 0)
        ENS_fos_mask    = mask_data(df_agg_ENS_fos_d, 0, False, 2, 0)

        # Calculate F
        Otero_rl3_stat  = get_f_score(ENS_fos_mask, Otero_rl3_mask,  beta=1)
        Otero_dd3_stat  = get_f_score(ENS_fos_mask, Otero_dd3_mask,  beta=1)
        Otero_lws3_stat  = get_f_score(ENS_fos_mask, Otero_lws3_mask,  beta=1)

        # Create Dataframe
        p_list.append( pd.concat([Otero_rl3_stat, Otero_lws3_stat, Otero_dd3_stat], keys=['RL', 'LWS', 'DD'],   names=['Drought type']))
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
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]

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

figname = f"Validation_Otero22_ENS_scenario{scenario_EVA}_{agg_zone}"

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
    for metric in ['F', 'TP', 'TN', 'FP', 'FN']:
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

if agg_zone == 'DE00':
    axs.set_title(f"Method Otero'22 for Germany")
elif agg_zone == 'FR00':
    axs.set_title(f"Method Otero'22 for France")
elif agg_zone == 'NL00':
    axs.set_title(f"Method Otero'22 for the Netherlands")
elif agg_zone == 'BE00':
    axs.set_title(f"Method Otero'22 for Belgium")
elif agg_zone == 'NO':
    axs.set_title(f"Method Otero'22 for Norway")
elif agg_zone == 'UK':
    axs.set_title(f"Method Otero'22 for the United Kingdom")
elif agg_zone == 'DK':
    axs.set_title(f"Method Otero'22 for Denmark")
elif agg_zone == 'SE':
    axs.set_title(f"Method Otero'22 for Sweden")
elif agg_zone == 'IT':
    axs.set_title(f"Method Otero'22 for Italy")
elif agg_zone == 'CWE':
    axs.set_title(f"Method Otero'22 for Central Western Europe")
elif agg_zone == 'NWE':
    axs.set_title(f"Method Otero'22 for North Western Europe")
elif agg_zone == 'CoreRegion':
    axs.set_title(f"Method Otero'22 for the Core Region")
elif agg_zone == 'CSA':
    axs.set_title(f"Method Otero'22 for the Continental Synchronous Area")
else:
    print('Warning: Set the title!')

metric = 'F'
for ncolor, ener_var, label in zip(range(3), ['RL', 'DD', 'LWS'], ['Residual load', 'Demand', 'RES']):
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




















#%%
# ===================================================================
# Correlation Otero 22 | ENS/Severity and ENS/Duration
# ===================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

scenario_EVA = 'B'

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.014
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.002
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0064
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0068
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.002
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']; p_max = 0.0104
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']; p_max = 0.0064
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']; p_max = 0.0168

# --- Agregated zones
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']

# --- (OLD) Percentile thresholds & capacity reference year from Otero et al. 2022

# FOR DE00
#p_max = 0.014 # 0.0132
#p_max = 0.022  # TODO: check for Otero with new data
#p_max = 0.0356 # TODO: check for Otero with new data
#p_max = 0.044  # TODO: check for Otero with new data

# FOR FR00
#p_max = 0.0068
#p_max = 0.026  # TODO: check for Otero with new data
#p_max = 0.0276 # TODO: check for Otero with new data
#p_max = 0.0204 # TODO: check for Otero with new data

# FOR CWE
#p_max = 0.0176

# FOR NO
#p_max = 0.0036 # T=1  # TODO: check for Otero with new data

# FOR IT
#p_max = 0.0284 # T=1 # TODO: check for Otero with new data

# For PL00
#p_max = 0.0084 # T=1 # TODO: check for Otero with new data

LWS_percentile = p_max
RL_percentile  = 1 - p_max
DD_percentile  = 1 - p_max

figname = f"Correlation_ENSvsOtero_ENS_Scenario{scenario_EVA}_{agg_zone}_pmax_{int(p_max*1000)}e-3"


# ---------------------------------------------
# Compute data for figure
# ---------------------------------------------

# --- Aggregation at the agg_zone level ---
# We create new dataframe that are organized in te same way as original dataframe, 
# but the column are not SZON but the name of the aggregation region.
# This is convenient to reuse the same code structure
df_agg_gen_d = pd.DataFrame()
df_agg_dem_d = pd.DataFrame()
df_agg_RL_d  = pd.DataFrame()

df_agg_gen_d[agg_zone] = data3_gen_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_dem_d[agg_zone] = data3_dem_d.loc[('HIST')][zones_list].sum(axis=1)
df_agg_RL_d[agg_zone]  = data3_RL_d.loc[('HIST')][zones_list].sum(axis=1)

# Add the 'HIST' index to make a multi-index array. This is done in order to use the function detect_drought_Otero22 
# which is adapted to explore data based on projections
df_agg_gen_d = pd.concat([df_agg_gen_d], keys=['HIST'], names=['Scenario'])
df_agg_dem_d = pd.concat([df_agg_dem_d], keys=['HIST'], names=['Scenario'])
df_agg_RL_d = pd.concat([df_agg_RL_d], keys=['HIST'], names=['Scenario'])
# --- end aggregation ----------------------

# Units for plotting
unit = 'TWh'
scaling = 0.000001 # 'unit' in MWh (bc original data is in MWh)

# --- Compute events ---
# Detect Events  
lws3_thresh, lws3_sigma = get_thresholds(df_agg_gen_d.loc[('HIST')][[agg_zone]], LWS_percentile, start_date='1982-01-01', end_date='2016-12-31', empirical=True)
rl3_thresh,  rl3_sigma  = get_thresholds(df_agg_RL_d.loc[('HIST')][[agg_zone]],  RL_percentile,  start_date='1982-01-01', end_date='2016-12-31', empirical=True)
dd3_thresh,  dd3_sigma  = get_thresholds(df_agg_dem_d.loc[('HIST')][[agg_zone]], DD_percentile,  start_date='1982-01-01', end_date='2016-12-31', empirical=True)
    
# Detect Drought days and events
lws3_days, lws3_events = detect_drought_Otero22(['HIST'],  [agg_zone], df_agg_gen_d, lws3_thresh, lws3_sigma)
rl3_days,  rl3_events  = detect_drought_Otero22(['HIST'],  [agg_zone], df_agg_RL_d,  rl3_thresh,  rl3_sigma, below=False)
dd3_days,  dd3_events  = detect_drought_Otero22(['HIST'],  [agg_zone], df_agg_dem_d, dd3_thresh,  dd3_sigma, below=False)

# --- Plot ---

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


# ---------------------------------------------
# Plot only RL
# ---------------------------------------------


# ----------
#  Severity
# ----------

fig, axs = plt.subplots(1, 1, figsize=(5,5))

idx = 0
#print(agg_zone+' RL  (4.1) Sev R value: '+str(lin_reg(x, y)[3]))
r_value = dict()
rho_value = dict()
ENS_list = []

for nFOS in range(1, 16+1): # use 16 = 1 to correctly put the scatter plot on top with its label first

    FOS = nFOS

    if nFOS == 16:
        FOS = 1

    r_value[FOS] = dict()
    rho_value[FOS] = dict()

    # Get ENS for this FOS scenario
    df_agg_ENS_fos_d = pd.DataFrame()
    df_agg_ENS_fos_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)

    for idx, ener_var, color_var, df_events in zip(range(1), ['RL'], dt_colors[:1], [rl3_events]):

        if nFOS == 1 or nFOS == 16:
            color = color_var
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

        x = df_events.loc[('HIST', agg_zone),('Severity (adapted)')]
        y = get_ENS_sums(df_agg_ENS_fos_d, df_events.loc[('HIST')], agg_zone)
        # Only Energy drought with Severity >= 0
        y = y[x >= 0]
        x = x[x >= 0]
        # Linear regression
        df_reg, intercept, slope, r_value[FOS][ener_var], p_value, reg_trend = lin_reg(x, y)
        # Spearman rank-order correlation coefficient
        rho_value[FOS][ener_var], p_spearman = stats.spearmanr(x, y)

        ENS_list.append(y)

        axs.scatter(x, y, color=color, label=label, alpha=alpha, marker=marker)
        if nFOS == 1:
            axs.plot(df_reg, c=color)
            
        axs.set_ylabel('Summed-up ENS [GWh]')
        axs.set_xlabel('Severity')
        axs.legend()

for idx, ener_var in enumerate(['RL']):
    r_q50 = np.quantile([r_value[FOS][ener_var] for FOS in range(1, 15+1)], 0.5)
    r_min = np.min([r_value[FOS][ener_var] for FOS in range(1, 15+1)])
    r_max = np.max([r_value[FOS][ener_var] for FOS in range(1, 15+1)])
    rho_q50 = np.quantile([rho_value[FOS][ener_var] for FOS in range(1, 15+1)], 0.5)
    rho_min = np.min([rho_value[FOS][ener_var] for FOS in range(1, 15+1)])
    rho_max = np.max([rho_value[FOS][ener_var] for FOS in range(1, 15+1)])
    #axs.set_title(f'{ener_var}, {agg_zone} (Scenario {scenario_EVA}, p={p_max})\n r={np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}] (Q50 [min, max])')
    axs.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]')

plt.savefig(f"{path_to_plot}Correlation/{figname}_severity.pdf")

# ----------
#  Duration
# ----------

fig, axs = plt.subplots(1, 1, figsize=(5,5))

idx = 0
#print(agg_zone+' RL  (4.1) Sev R value: '+str(lin_reg(x, y)[3]))
r_value = dict()

for nFOS in range(1, 16+1): # use 16 = 1 to correctly put the scatter plot on top with its label first

    FOS = nFOS

    if nFOS == 16:
        FOS = 1

    r_value[FOS] = dict()
    rho_value[FOS] = dict()

    # Get ENS for this FOS scenario
    df_agg_ENS_fos_d = pd.DataFrame()
    df_agg_ENS_fos_d[agg_zone] = pd.read_pickle(path_to_data+f'ERAA23_ENS_TY2033_Scenario{scenario_EVA}_FOS{FOS}_daily.pkl')[zones_list].sum(axis=1)

    for idx, ener_var, color_var, df_events in zip(range(1), ['RL'], dt_colors[:1], [rl3_events]):

        if nFOS == 1 or nFOS == 16:
            color = color_var
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

        x = df_events.loc[('HIST', agg_zone),('Duration')]
        y = get_ENS_sums(df_agg_ENS_fos_d, df_events.loc[('HIST')], agg_zone)
        # Linear regression
        df_reg, intercept, slope, r_value[FOS][ener_var], p_value, reg_trend = lin_reg(x, y)
        # Spearman rank-order correlation coefficient
        rho_value[FOS][ener_var], p_spearman = stats.spearmanr(x, y)

        axs.scatter(x, y, color=color, label=label, alpha=alpha, marker=marker)
        if nFOS == 1:
            axs.plot(df_reg, c=color)

        axs.set_ylabel('Summed-up ENS [GWh]')
        axs.set_xlabel('Duration [d]')
        axs.legend()

        


for idx, ener_var in enumerate(['RL']):
    r_q50 = np.quantile([r_value[FOS][ener_var] for FOS in range(1, 15+1)], 0.5)
    r_min = np.min([r_value[FOS][ener_var] for FOS in range(1, 15+1)])
    r_max = np.max([r_value[FOS][ener_var] for FOS in range(1, 15+1)])
    rho_q50 = np.quantile([rho_value[FOS][ener_var] for FOS in range(1, 15+1)], 0.5)
    rho_min = np.min([rho_value[FOS][ener_var] for FOS in range(1, 15+1)])
    rho_max = np.max([rho_value[FOS][ener_var] for FOS in range(1, 15+1)])
    #axs.set_title(f'{ener_var}, {agg_zone} (Scenario {scenario_EVA}, p={p_max})\n r={np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}] (Q50 [min, max])')
    axs.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]')

plt.savefig(f"{path_to_plot}Correlation/{figname}_duration.pdf")




































#%%
# =================================================================
# Plot F-score | Stoop 23 | Compare RL, DD, RES for scenario B
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1

# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]

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
        # Find capacity thresholds
        # Percentile is computed on the clustered events ("independent" events)
        # interpolation="nearest" gives the same result as Benjamin B.'s "empirical=True" in the function "get_thresholds"
        rl3_thresh = np.quantile(rl3_event_values, q=RL_percs[p], interpolation="nearest")
        lws3_thresh = np.quantile(lws3_event_values, q=LWS_percs[p], interpolation="nearest")
        dd3_thresh = np.quantile(dd3_event_values, q=DD_percs[p], interpolation="nearest")
        
        # Mask the data / Detect Drought days
        rl3_mask = mask_CREDI(rl3_event_dates, rl3_event_values, rl3_thresh, PERIOD_length_days, zone=agg_zone, extreme_is_high=True)
        lws3_mask = mask_CREDI(lws3_event_dates, lws3_event_values, lws3_thresh, PERIOD_length_days, zone=agg_zone, extreme_is_high=False)
        dd3_mask = mask_CREDI(dd3_event_dates, dd3_event_values, dd3_thresh, PERIOD_length_days, zone=agg_zone, extreme_is_high=True)
        ENS_fos_mask = mask_data(df_agg_ENS_fos_d, 0, False, 2, 0)

        # Calculate F
        rl3_stat  = get_f_score(ENS_fos_mask, rl3_mask,  beta=1)
        dd3_stat  = get_f_score(ENS_fos_mask, dd3_mask,  beta=1)
        lws3_stat  = get_f_score(ENS_fos_mask, lws3_mask,  beta=1)

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


#%%
# ---------------------------------------------
#  Plot only F
# ---------------------------------------------

PERIOD_length_days = 1
PERIOD_cluster_days = 1


# --- Non-aggregated zones
# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]

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
    for metric in ['F', 'TP', 'TN', 'FP', 'FN']:
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

if agg_zone == 'DE00':
    axs.set_title(f"Method Stoop'23 for Germany (T={PERIOD_length_days}d)")
elif agg_zone == 'FR00':
    axs.set_title(f"Method Stoop'23 for France (T={PERIOD_length_days}d)")
elif agg_zone == 'NL00':
    axs.set_title(f"Method Stoop'23 for the Netherlands (T={PERIOD_length_days}d)")
elif agg_zone == 'BE00':
    axs.set_title(f"Method Stoop'23 for Belgium (T={PERIOD_length_days}d)")
elif agg_zone == 'NO':
    axs.set_title(f"Method Stoop'23 for Norway (T={PERIOD_length_days}d)")
elif agg_zone == 'UK':
    axs.set_title(f"Method Stoop'23 for the United Kingdom (T={PERIOD_length_days}d)")
elif agg_zone == 'DK':
    axs.set_title(f"Method Stoop'23 for Denmark (T={PERIOD_length_days}d)")
elif agg_zone == 'SE':
    axs.set_title(f"Method Stoop'23 for Sweden (T={PERIOD_length_days}d)")
elif agg_zone == 'IT':
    axs.set_title(f"Method Stoop'23 for Italy (T={PERIOD_length_days}d)")
elif agg_zone == 'CWE':
    axs.set_title(f"Method Stoop'23 for Central Western Europe (T={PERIOD_length_days}d)")
elif agg_zone == 'NWE':
    axs.set_title(f"Method Stoop'23 for North Western Europe (T={PERIOD_length_days}d)")
elif agg_zone == 'CoreRegion':
    axs.set_title(f"Method Stoop'23 for the Core Region (T={PERIOD_length_days}d)")
elif agg_zone == 'CSA':
    axs.set_title(f"Stoop'23 for Continental Synchronous Area (T={PERIOD_length_days}d)")
else:
    print('Warning: Set the title!')

# Event time series

metric = 'F'
for ncolor, ener_var, label in zip(range(3), ['RL', 'DD', 'LWS'], ['Residual load', 'Demand', 'RES']):
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
#plt.show()

plt.savefig(f"{path_to_plot}Validation/{figname}.pdf", dpi=300)
#plt.close()




























#%% 
# =================================================================
# Correlation for CREDI | New ENS + possible aggregated scale
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# Duration of CREDI events
PERIOD_length_days = 1
# Clustering period. If PERIOD_length_days < PERIOD_cluster_days, then there may be overlaps. We authorize up to 25% overlap.
PERIOD_cluster_days = 1

scenario_EVA = 'B'

# Same structure as for "aggregated zone" so that the code work for aggregated and non-agregated zones
zone = 'DE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.014
#zone = 'NL00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0176
#zone = 'BE00'; agg_zone = zone; zones_list = [zone]; p_max = 0.0064
#zone = 'FR00'; agg_zone = zone; zones_list = [zone]; p_max = 0.004
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.0036
#agg_zone = 'UK'; zones_list = ['UK00','UKNI']; p_max = 0.0108
#agg_zone = 'DK'; zones_list = ['DKE1','DKW1']; p_max = 0.0112
#agg_zone = 'NWE'; zones_list = ['DE00','NL00','BE00','FR00','NOS0','NOM1','NON1','UK00','UKNI','DKE1','DKW1']; p_max = 0.0256

# Agregated zones
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00'] # Luxembourg is not in demand dataset
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04'] # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']


# --- Old Percentile for peak F-score. Computed in `DF_Validation_Stoop.py` ---
# FOR DE00
#p_max = 0.014  #0.0136
#p_max = 0.022  # TODO: check with new data
#p_max = 0.0356 # TODO: check with new data
#p_max = 0.044  # TODO: check with new data

# FOR FR00
#p_max = 0.0044
#p_max = 0.026  # TODO: check with new data
#p_max = 0.0276 # TODO: check with new data
#p_max = 0.0204 # TODO: check with new data

# FOR CWE
#p_max = 0.014 # T=1
#p_max = 0.038 # T=3
#p_max = 0.07 # T=5
#p_max = 0.136 # T=7

# FOR NO
#p_max = 0.0036 # T=1

# FOR IT
#p_max = 0.0284 # T=1

# For PL00
#p_max = 0.0084 # T=1


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
ax.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]')

ax.set_ylabel('Summed-up ENS [GWh]')
ax.set_xlabel('CREDI [GWh]')
ax.legend(facecolor="white", loc='upper left', framealpha=1)

plt.tight_layout()
#plt.show()

plt.savefig(f"{path_to_plot}Correlation/{figname}.pdf")
#plt.close()


























