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
    get_df_timerange, lin_reg, detect_drought_Li21, mask_df_by_entries, detect_DF_Li21, mask_Li21, \
    pairwise_comparison, condorcet_ranking
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
# CREDI illustration
# =================================================================

# ------------------------------------------------------------------
# User defined parameters
# ------------------------------------------------------------------

# Duration of CREDI events
PERIOD_length_days = 3
# Clustering period. If PERIOD_length_days < PERIOD_cluster_days, then there may be overlaps. We authorize up to 25% overlap.
PERIOD_cluster_days = 3

var_short_name = 'RL'
zone = 'DE00'
PECD_version = 'PECDv3'

# percentile for peak F-score. Computed in `DF_Validation_Stoop.py`
#p_max = 0.01
#p_max = 0.022 # T=3d
#p_max = 0.0356
#p_max = 0.044

# New
#p_max = 0.014 # T=1
p_max = 0.0384 # T=3d
#p_max = 0.048 # T=5
#p_max = 0.088 # T=7

# ------------------------------------------------------------------
# Figure (~ 3 min)
# ------------------------------------------------------------------

## Length of the period to consider for CREDI assessment (in hours)
# add 1 to get indexes that make sense 
PERIOD_length = PERIOD_length_days * 24 + 1 #193 # 8 days

# Sampling of the period (in hours)
PERIOD_stride = 24

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
ds_plot = ds_choice.to_xarray()[[zone]]

# Change units: MW -> GW
ds_plot = ds_plot / 1000

ds_clim_hourly = Climatology_Hourly(ds_plot)
ds_clim_HWRW, MOH = Climatology_Hourly_Weekly_Rolling(ds_plot)
ds_anom_HWRW = ds_plot.groupby(MOH) - ds_clim_HWRW
# Compute the cummulative sum over all hours, then take only the lat event hour to get one T-day average per day
ds_CREDI = ds_anom_HWRW.rolling(Date=PERIOD_length).construct(Date="event_hour", stride=PERIOD_stride).cumsum(dim='event_hour')#.sel(event_hour=PERIOD_length-1)

# Get CREDI events
rl3_CREDI_event, rl3_event_dates, \
    rl3_event_values = get_CREDI_events(ds_choice, zone, extreme_is_high=True, PERIOD_length_days=PERIOD_length_days,
                                        PERIOD_cluster_days=PERIOD_cluster_days, start_date='1982-01-01', end_date='2016-12-31')
rl3_thresh = np.quantile(rl3_event_values, q=1-p_max, interpolation="nearest")
CREDI_events = np.asarray(rl3_event_values) / 1000 # MWh -> GWh
thresh = np.asarray(rl3_thresh) / 1000 # MWh -> GWh
print(f'Threshold = {thresh} (p = {p_max})')

# we start a new figure
#fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,8), sharex=True)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,7), sharey='row', sharex='col', gridspec_kw={'width_ratios': [9, 1]})

# fix date-format
fig.autofmt_xdate()

# show years
year_dates = pd.date_range('1982-01-01', periods=8760, freq='1h')

ax[0,0].plot(year_dates[24*0:PERIOD_length+24*25], ds_plot[zone][24*0:PERIOD_length+24*25], color=colour, alpha=1)
ax[0,0].plot(year_dates[24*0:PERIOD_length+24*25], ds_clim_HWRW[zone][24*0:PERIOD_length+24*25], color=colour_hrw, alpha=0.7, linewidth=3, label='climatology')
#ax[0].plot(year_dates[00:8760:24], ds_clim_hourly[zone][hour:8760:24], color=colour_clim, alpha=1)

# Print threshold value
# T=3, Tc=3, p_max=0.022
#thresh = 3784.475895739609
#thresh = 4150 # hand picked
#thresh = 3454.7869942094176 # (p = 0.0384)
ax[1,0].axhline(thresh, alpha=0.5, color='black', linestyle='dashed', label=f'threshold (percentile {np.round(p_max, 2)})')

for nday in range(0, 25+1):

    if nday in [9, 12, 16, 21]:
        color = colour_hrw
        alpha = 1
    elif nday in [0, 3, 6, 24]:
        color = colour_hrw
        alpha = 0.3
    else:
        color = 'grey'
        alpha = 0.3
    # label
    if nday == 1:
        label='overlapping CREDI event'
    elif nday == 0:
        label='CREDI event (below threshold)'
    elif nday == 9:
        label='CREDI event (above threshold)'
    else:
        label=''

    ax[1,0].plot(year_dates[24*nday:PERIOD_length+24*nday], ds_CREDI[zone][PERIOD_length_days+nday], color=color, linewidth=3, alpha=alpha)
    ax[1,0].scatter(year_dates[24*nday:PERIOD_length+24*nday][-1], ds_CREDI[zone][PERIOD_length_days+nday][-1], 
                marker='o', s=150,  color=color, alpha=alpha, label=label)

# format labels
ax[0,0].set_ylabel(f'{var_long_name} [GW]')
ax[1,0].set_ylabel(f'CREDI {var_long_name} [GWh]')

# format legend
ax[0,0].legend(fontsize='medium')
ax[1,0].legend(fontsize='medium')

ax[0,0].grid(axis='x')
ax[1,0].grid(axis='x')

ax[1,0].set(xlim=(year_dates[24*0], year_dates[PERIOD_length+24*25]))

# --- PDF ---

ax[0,1].remove()

binsize = 40
bins = np.linspace(np.min(CREDI_events), np.max(CREDI_events), binsize)
hist = np.histogram(CREDI_events, bins=bins)
x = hist[1][:-1] + np.diff(hist[1])[0] / 2
ax[1,1].plot(hist[0], x, drawstyle='steps', label='RL', color='dodgerblue')
ax[1,1].axhline(thresh, alpha=0.5, color='black', linestyle='dashed')
ax[1,1].set_xlabel('Number of\nCREDI events')

ax[0,0].text(0.02, 0.95, 'a)', fontsize=14, transform=ax[0,0].transAxes, va='top')
ax[1,0].text(0.02, 0.95, 'b)', fontsize=14, transform=ax[1,0].transAxes, va='top')
ax[1,1].text(0.15, 0.95, 'c)', fontsize=14, transform=ax[1,1].transAxes, va='top')

plt.tight_layout()
#plt.savefig(path_to_plot+f'Illustrations/Illustration_CREDIevents_{var_short_name}_{zone}.pdf')

# make it look better
plt.show()








































#%%
# =================================================================
# Plot F-score | Li 21 | Compare RL, DD, RES for scenario B
# =================================================================

# ---------------------------------------------
# User defined parameters
# ---------------------------------------------

# No "aggregated zone" because we cannot sum capacity factor. For this, we would need to devide by the installed capacity
zone = 'DE00'
#agg_zone = zone; zones_list = ['zone']

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
    zone_name = "Germany"
elif zone == 'FR00':
    zone_name = "France"
elif zone == 'NL00':
    zone_name = "the Netherlands"
elif zone == 'BE00':
    zone_name = "Belgium"
else:
    zone_name = ""
    print('Warning: Set the title!')

axs.set_title(zone_name, loc='left')
axs.set_title(f"Method Li'21", loc='right')

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
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00']; p_max = 0.0172 # Luxembourg is not in demand dataset
#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00']; p_max = 0.0316  # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00']; p_max = 0.056  # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].


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
            
        axs.set_ylabel('Total ENS [GWh]')
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
    #axs.text(0.05, 0.8, zone_name, fontsize=10, transform=axs.transAxes, ha='left', va='top')
    axs.set_title(zone_name, loc='left')
    axs.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')

plt.tight_layout()
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

        axs.set_ylabel('Total ENS [GWh]')
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
    #axs.text(0.05, 0.8, zone_name, fontsize=10, transform=axs.transAxes, ha='left', va='top')
    axs.set_title(zone_name, loc='left')
    axs.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')

plt.tight_layout()
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
#zone = 'DE00'; agg_zone = zone; zones_list = [zone]
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
axs.set_title(f"Method Stoop'23 (T={PERIOD_length_days}d)", loc='right')

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
#agg_zone = 'CWE'; zones_list = ['AT00', 'BE00', 'CH00', 'DE00', 'FR00', 'NL00']; p_max = 0.0132 # Luxembourg is not in demand dataset
#agg_zone = 'NO'; zones_list = ['NOS0', 'NOM1', 'NON1']; p_max = 0.0036
#agg_zone = 'SE'; zones_list = ['SE01', 'SE03', 'SE04']; p_max = 0.0024 # There is no SE02 data in the new and old ENS dataset
#agg_zone = 'IT'; zones_list = ['ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA']; p_max = 0.0316

#agg_zone = 'CoreRegion'; zones_list = ['BE00', 'FR00', 'NL00', 'DE00', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00']; p_max = 0.0208 # Luxembourg is not in demand dataset
#agg_zone = 'CSA'; zones_list = ['PT00', 'ES00', 'BE00', 'FR00', 'NL00', 'DE00', 'DKW1', 'CH00', 'ITN1', 'ITCN', 'ITCS', 'ITS1', 'ITCA', 'ITSI', 'ITSA', 'PL00', 'CZ00', 'AT00', 'SI00', 'SK00', 'HU00', 'HR00', 'RO00', 'BA00', 'RS00', 'ME00', 'MK00', 'GR00', 'BG00']; p_max = 0.05  # (Part of) Continental Synchronous Area (based on available data). 'DKE1' is part of the Nordic Zone. No data for ['FR15', 'MD00', 'UA01', 'UA02', 'CR00', 'TR00'].


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
# TODO: Dirty -> I used an old piece of code
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
ax.set_title(zone_name, loc='left')
ax.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')

ax.set_ylabel('Total ENS [GWh]')
ax.set_xlabel('CREDI [GWh]')
ax.legend(facecolor="white", loc='upper left', framealpha=1)

plt.tight_layout()
plt.savefig(f"{path_to_plot}Correlation/{figname}.pdf")
#plt.close()


























#%%
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

figname = f"YearRank_CumulativeRL_ENS_Scenario{scenario_EVA}_{agg_zone}"


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

"""
x_list  = [df_RL_annual,                        df_RL_annual,                       df_RL_annual,                       df_RL_annual]
x_labels= ['Cumulative residual load [TWh]',    'Cumulative residual load [TWh]',   'Cumulative residual load [TWh]',   'Cumulative residual load [TWh]']
y_list  = [df_ENS_annual_sum,                   df_ENS_annual_max,                  df_ENS_hours,                       df_ENS_days]
y_labels= ['Total ENS [MWh]',                   'Max daily ENS [MWh]',              'Hours of ENS [h]',         'Days with ENS>0 [d]']
axs_idx = [0, 1, 2, 3]

fig, axs = plt.subplots(1, 4, figsize=(20,5))
for x, x_label, y, y_label, idx in zip(x_list, x_labels, y_list, y_labels, axs_idx):

"""

x  = df_RL_annual
x_label= 'Cumulative residual load [TWh]'
y  = df_ENS_hours
y_label= 'Hours of ENS [h]'

fig, axs = plt.subplots(1, 1, figsize=(5,5))

r_value = dict()
rho_value = dict()

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

    axs.scatter(x, y[FOS], color=color, label=label, alpha=alpha, marker=marker)
    if FOS == 1:
        axs.plot(df_reg, c=color)
    axs.set_ylabel(y_label)
    axs.set_xlabel(x_label)
    axs.legend(loc='upper left')

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

# Show worst years for FOS=1 on the figure
FOS = 1
n_extreme_years_FOS = 5
list_extreme_years_FOS = y[FOS].sort_values().index.to_list()[::-1][:n_extreme_years_FOS]
for year in list_extreme_years_FOS:
    axs.text(x.loc[year] * 0.997, y[FOS].loc[year], year, ha='right', color=dt_colors[1])

r_q50 = np.quantile([r_value[FOS] for FOS in range(1, 15+1)], 0.5)
r_min = np.min([r_value[FOS] for FOS in range(1, 15+1)])
r_max = np.max([r_value[FOS] for FOS in range(1, 15+1)])
rho_q50 = np.quantile([rho_value[FOS] for FOS in range(1, 15+1)], 0.5)
rho_min = np.min([rho_value[FOS] for FOS in range(1, 15+1)])
rho_max = np.max([rho_value[FOS] for FOS in range(1, 15+1)])
#axs.set_title(f'{ener_var}, {agg_zone} (Scenario {scenario_EVA}, p={p_max})\n r={np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}] (Q50 [min, max])')

axs.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')
axs.set_title(zone_name, loc='left')

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

figname = f"YearRank_Otero_ENS_Scenario{scenario_EVA}_{agg_zone}"


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


"""
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
for x, x_label, y, y_label, idx in zip(x_list, x_labels, y_list, y_labels, axs_idx):
"""

x       = Otero_rl3_year['Severity (adapted)']
x_label = 'Total Severity'
y       = df_ENS_hours
y_label = 'Hours of ENS [h]'


fig, axs = plt.subplots(1, 1, figsize=(5, 5))

r_value = dict()
rho_value = dict()

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

    axs.scatter(x, y[FOS], color=color, label=label, alpha=alpha, marker=marker)
    if FOS == 1:
        axs.plot(df_reg, c=color)
    axs.set_ylabel(y_label)
    axs.set_xlabel(x_label)
    axs.legend(loc='upper left')

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

# Show worst years for FOS=1 on the figure
FOS = 1
n_extreme_years_FOS = 5
list_extreme_years_FOS = y[FOS].sort_values().index.to_list()[::-1][:n_extreme_years_FOS]
for year in list_extreme_years_FOS:
    axs.text(x.loc[year] * 0.98, y[FOS].loc[year], year, ha='right', color=dt_colors[1])


r_q50 = np.quantile([r_value[FOS] for FOS in range(1, 15+1)], 0.5)
r_min = np.min([r_value[FOS] for FOS in range(1, 15+1)])
r_max = np.max([r_value[FOS] for FOS in range(1, 15+1)])
rho_q50 = np.quantile([rho_value[FOS] for FOS in range(1, 15+1)], 0.5)
rho_min = np.min([rho_value[FOS] for FOS in range(1, 15+1)])
rho_max = np.max([rho_value[FOS] for FOS in range(1, 15+1)])
#axs.set_title(f'{ener_var}, {agg_zone} (Scenario {scenario_EVA}, p={p_max})\n r={np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}] (Q50 [min, max])')

axs.set_title(r'$r=$'+f'{np.round(r_q50, 2)} [{np.round(r_min, 2)}, {np.round(r_max, 2)}]\n'+r'$\rho=$'+f'{np.round(rho_q50, 2)} [{np.round(rho_min, 2)}, {np.round(rho_max, 2)}]', loc='right')
axs.set_title(zone_name, loc='left')

fig.tight_layout()
plt.savefig(f"{path_to_plot}Correlation/{figname}.pdf")