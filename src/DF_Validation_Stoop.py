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
from datetime import timedelta

from Dunkelflaute_function_library import get_zones, get_thresholds, detect_drought_Otero22, mask_data, get_f_score, get_df_timerange
from CREDIfunctions import Modified_Ordinal_Hour, Climatology_Hourly, Climatology_Hourly_Rolling, \
    Climatology_Hourly_Weekly_Rolling, get_CREDI_threshold, get_f_score_CREDI

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
dt_colors = ['limegreen', 'violet', 'dodgerblue', 'darkgreen', 'darkmagenta', 'darkblue']

# Percentile thresholds to investigate
#LWS_percs = np.linspace(0.0001,0.15,200)
#LWS_percs = np.linspace(0.0001, 0.15, 40)
LWS_percs = np.array([0.0001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15])
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



#%%
# =================================================================
# Load hourly data
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
# Figure (~ 30 min ?)
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


# For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

p_list  = []
for p in range(len(LWS_percs)):
    # find capacity thresholds
    lws3_CREDI_event, lws3_thresh, \
    lws3_event_dates, lws3_event_values = get_CREDI_threshold(data3_gen_h.loc[('HIST')], zone, percentile=LWS_percs[p], 
                                                              extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                                              PERIOD_cluster_days=PERIOD_cluster_days, 
                                                              start_date='1982-01-01', end_date='2016-12-31')
    lws4_CREDI_event, lws4_thresh, \
    lws4_event_dates, lws4_event_values = get_CREDI_threshold(data4_gen_h.loc[('HIST')], zone, percentile=LWS_percs[p], 
                                                              extreme_is_high=False, PERIOD_length_days=PERIOD_length_days, 
                                                              PERIOD_cluster_days=PERIOD_cluster_days, 
                                                              start_date='1982-01-01', end_date='2016-12-31')
    rl3_CREDI_event, rl3_thresh, \
    rl3_event_dates, rl3_event_values = get_CREDI_threshold(data3_RL_h.loc[('HIST')], zone, percentile=RL_percs[p], 
                                                            extreme_is_high=True, PERIOD_length_days=PERIOD_length_days, 
                                                            PERIOD_cluster_days=PERIOD_cluster_days, 
                                                            start_date='1982-01-01', end_date='2016-12-31')
    rl4_CREDI_event, rl4_thresh, \
    rl4_event_dates, rl4_event_values = get_CREDI_threshold(data4_RL_h.loc[('HIST')], zone, percentile=RL_percs[p], 
                                                            extreme_is_high=True, PERIOD_length_days=PERIOD_length_days, 
                                                            PERIOD_cluster_days=PERIOD_cluster_days, 
                                                            start_date='1982-01-01', end_date='2016-12-31')
    dd3_CREDI_event, dd3_thresh, \
    dd3_event_dates, dd3_event_values = get_CREDI_threshold(data3_dem_h.loc[('HIST')], zone, percentile=DD_percs[p], 
                                                            extreme_is_high=True, PERIOD_length_days=PERIOD_length_days, 
                                                            PERIOD_cluster_days=PERIOD_cluster_days, 
                                                            start_date='1982-01-01', end_date='2016-12-31')
    dd4_CREDI_event, dd4_thresh, \
    dd4_event_dates, dd4_event_values = get_CREDI_threshold(data4_dem_h.loc[('HIST')], zone, percentile=DD_percs[p], 
                                                            extreme_is_high=True, PERIOD_length_days=PERIOD_length_days, 
                                                            PERIOD_cluster_days=PERIOD_cluster_days, 
                                                            start_date='1982-01-01', end_date='2016-12-31')

    # Mask the data / Detect Drought days
    lws3_mask  = mask_data(lws3_CREDI_event[[zone]], lws3_thresh, True, 1, 0)
    lws4_mask  = mask_data(lws4_CREDI_event[[zone]], lws4_thresh, True, 1, 0)
    rl3_mask  = mask_data(rl3_CREDI_event[[zone]], rl3_thresh, False, 1, 0)
    rl4_mask  = mask_data(rl4_CREDI_event[[zone]], rl4_thresh, False, 1, 0)
    dd3_mask  = mask_data(dd3_CREDI_event[[zone]], dd3_thresh, False, 1, 0)
    dd4_mask  = mask_data(dd4_CREDI_event[[zone]], dd4_thresh, False, 1, 0)

    # Calculate F (compared to ENS)
    lws3_stat  = get_f_score_CREDI(ens_mask, lws3_mask, zone, PERIOD_length_days, beta=1)
    lws4_stat  = get_f_score_CREDI(ens_mask, lws4_mask, zone, PERIOD_length_days, beta=1)
    rl3_stat  = get_f_score_CREDI(ens_mask, rl3_mask, zone, PERIOD_length_days, beta=1)
    rl4_stat  = get_f_score_CREDI(ens_mask, rl4_mask, zone, PERIOD_length_days, beta=1)
    dd3_stat  = get_f_score_CREDI(ens_mask, dd3_mask, zone, PERIOD_length_days, beta=1)
    dd4_stat  = get_f_score_CREDI(ens_mask, dd4_mask, zone, PERIOD_length_days, beta=1)

    # Create Dataframe
    p_list.append( pd.concat([lws3_stat, lws4_stat, rl3_stat, rl4_stat, dd3_stat, dd4_stat], 
                             keys=['LWS3', 'LWS4', 'RL3', 'RL4', 'DD3', 'DD4'],   names=['Drought type']))
    print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
stat_df  = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
 


# Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
#stat_df.to_pickle(path_to_data+'validation_Otero22_detection_stats.pkl')
#stat_df.to_csv(   path_to_data+'validation_Otero22_detection_stats.csv',sep=';')


#%% Plot F / threshold (comparing to ENS / validation)
x=stat_df.index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)


for c in range(len(zones_szon)):
    fig, axs = plt.subplots(3, 2, figsize=(10,16))
    fig.suptitle('Stoop Method for '+zones_szon[c]+' using '+ens_dataset+' ENS\n(Investigates only Yes/No occurance and no measure of severity)')
    
    # Event time series
    idx, idy = 0, [0,1]
    ax_big = plt.subplot(3, 1, 1)
    ax_big.plot(x, stat_df.loc[(x, 'LWS3',  'F'),(zones_szon[c])], label='LWS3',  color=dt_colors[0], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'LWS4',  'F'),(zones_szon[c])], label='LWS4',  color=dt_colors[3], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'RL3',  'F'),(zones_szon[c])], label='RL3',  color=dt_colors[1], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'RL4',  'F'),(zones_szon[c])], label='RL4',  color=dt_colors[4], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'DD3',  'F'),(zones_szon[c])], label='DD3',  color=dt_colors[2], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'DD4',  'F'),(zones_szon[c])], label='DD4',  color=dt_colors[5], alpha=0.8)
    ax_big.set_ylabel('F-Score')
    ax_big.set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
    axs[0,0].remove()
    axs[0,1].remove()
    
    idx, idy = 1, 0
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'TP'),(zones_szon[c])], label='LWS3',  color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'TP'),(zones_szon[c])], label='LWS4',  color=dt_colors[3], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TP'),(zones_szon[c])], label='RL3',  color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'TP'),(zones_szon[c])], label='RL4',  color=dt_colors[4], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TP'),(zones_szon[c])], label='DD3',  color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'TP'),(zones_szon[c])], label='DD4',  color=dt_colors[5], alpha=0.8)
    axs[idx, idy].set_ylabel('True Positives (of '+str(nr_of_pos[zones_szon[c]])+' in total)\n(DF detected, when ENS)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 1, 1
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'TN'),(zones_szon[c])], label='LWS3',  color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'TN'),(zones_szon[c])], label='LWS4',  color=dt_colors[3], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TN'),(zones_szon[c])], label='RL3',  color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'TN'),(zones_szon[c])], label='RL4',  color=dt_colors[4], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TN'),(zones_szon[c])], label='DD3',  color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'TN'),(zones_szon[c])], label='DD4',  color=dt_colors[5], alpha=0.8)
    axs[idx, idy].set_ylabel('True Negatives (of '+str(nr_of_neg[zones_szon[c]])+' in total)\n(no DF detected, when no ENS)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 2, 0
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'FP'),(zones_szon[c])], label='LWS3',  color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'FP'),(zones_szon[c])], label='LWS4',  color=dt_colors[3], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FP'),(zones_szon[c])], label='RL3',  color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'FP'),(zones_szon[c])], label='RL4',  color=dt_colors[4], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FP'),(zones_szon[c])], label='DD3',  color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'FP'),(zones_szon[c])], label='DD4',  color=dt_colors[5], alpha=0.8)
    axs[idx, idy].set_ylabel('False Positives (of '+str(nr_of_neg[zones_szon[c]])+' in total)\n(DF detected, when no ENS)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 2, 1
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3',  'FN'),(zones_szon[c])], label='LWS3',  color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4',  'FN'),(zones_szon[c])], label='LWS4',  color=dt_colors[3], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FN'),(zones_szon[c])], label='RL3',  color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'FN'),(zones_szon[c])], label='RL4',  color=dt_colors[4], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FN'),(zones_szon[c])], label='DD3',  color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'FN'),(zones_szon[c])], label='DD4',  color=dt_colors[5], alpha=0.8)
    axs[idx, idy].set_ylabel('False Negatives (of '+str(nr_of_pos[zones_szon[c]])+' in total)\n(No DF detected, when ENS)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    plt.tight_layout()
    plt.show()

    plt.savefig(f"{path_to_plot}Validation/{figname}.{plot_format}", dpi=300)
    plt.close()

    print(f"Saved {path_to_plot}Validation/{figname}.{plot_format}")


# TODO: change the x-axis name to more explicit of what is the percentile

# TODO: add figures for "recall" and "precision"
# %%
