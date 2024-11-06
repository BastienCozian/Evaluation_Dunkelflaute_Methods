# -*- coding: utf-8 -*-
"""
This scripts does the validating of Method 2 (inspired by Otero et al. 2022)
For several thresholds energy droughts are detected and evaluated using ENS data.
Results of the Evaluation are saved as pickles.
Plots are also created to visualize the "best-fittting" threshold.

For questions, refer to benjamin.biewald@tennet.eu
"""

#%% Import packages

import numpy as np
import pandas as pd
from Dunkelflaute_function_library import get_zones
from Dunkelflaute_function_library import get_thresholds
from Dunkelflaute_function_library import detect_drought_Otero22
from Dunkelflaute_function_library import mask_data
from Dunkelflaute_function_library import get_f_score
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

#%% Set parameters
path_to_data      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'  #'D:/Dunkelflaute/Data/'
path_to_plot      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'      #'D:/Dunkelflaute/'
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

dt_colors = ['green', 'purple', 'blue', 'maroon', 'cyan', 'orange']
dt_colors = ['green', 'green', 'blue', 'blue', 'orange', 'orange']

# Percentile thresholds to investigate
LWS_percs = np.linspace(0.0001,0.15,200)
RL_percs  = 1-LWS_percs
DD_percs  = 1-LWS_percs
        
zones_peon = zones = get_zones(countries,'PEON')  
zones_szon = zones = get_zones(countries,'SZON')  
zones_peof = zones = get_zones(countries,'PEOF')  

scen_timespan = [42,51,51,51] # TODO: Automate the nr of years per scenario


#%% Load data
#data4_REP_d = pd.read_pickle(path_to_data+'PECD4_Generation_national_daily.pkl')
#data3_REP_d = pd.read_pickle(path_to_data+'PECD3_Generation_national_daily.pkl')
#data3_RL_d  = pd.read_pickle(path_to_data+'PECD3_PEMMDB_RL_national_daily.pkl')
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
data3_cropped2 = data3_cropped1[~((data3_cropped1.index.get_level_values(1).day == 29) & (data3_cropped1.index.get_level_values(1).month == 2))]
data4_cropped2 = data4_cropped1[~((data4_cropped1.index.get_level_values(1).day == 29) & (data4_cropped1.index.get_level_values(1).month == 2))]

data3_RL_d = data3_dem_d - w*data3_cropped2
data4_RL_d = data4_dem_d - w*data4_cropped2




#%% For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

ens_mask = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos = (ens_mask==2).sum()
nr_of_neg = (ens_mask==0).sum()

p_list  = []
p_list2 = []
for p in range(len(LWS_percs)):
    
# find capacity thresholds
    lws4_thresh, lws4_sigma = get_thresholds(data4_REP_d.loc[('HIST')], LWS_percs[p], start_date='1980-01-01', end_date='2019-12-31', empirical=True)
    lws3_thresh, lws3_sigma = get_thresholds(data3_REP_d.loc[('HIST')], LWS_percs[p], start_date='1980-01-01', end_date='2019-12-31', empirical=True)
    rl3_thresh,  rl3_sigma  = get_thresholds(data3_RL_d.loc[('HIST')] ,  RL_percs[p] , start_date='1980-01-01', end_date='2019-12-31', empirical=True)
    rl4_thresh,  rl4_sigma  = get_thresholds(data4_RL_d.loc[('HIST')] ,  RL_percs[p] , start_date='1980-01-01', end_date='2019-12-31', empirical=True)
    dd3_thresh,  dd3_sigma  = get_thresholds(data3_dem_d.loc[('HIST')] ,   DD_percs[p] , start_date='1980-01-01', end_date='2019-12-31', empirical=True)
    dd4_thresh,  dd4_sigma  = get_thresholds(data4_dem_d.loc[('HIST')] , DD_percs[p] , start_date='1980-01-01', end_date='2019-12-31', empirical=True)

    
# Mask the data / Detect Drought days
    lws4_mask = mask_data(data4_REP_d.loc[('HIST')], lws4_thresh, True,  1, 0)
    lws3_mask = mask_data(data3_REP_d.loc[('HIST')], lws3_thresh, True,  1, 0)
    rl3_mask  = mask_data(data3_RL_d.loc[('HIST')],  rl3_thresh,  False, 1, 0)
    rl4_mask  = mask_data(data4_RL_d.loc[('HIST')],  rl4_thresh,  False, 1, 0)
    dd3_mask  = mask_data(data3_dem_d.loc[('HIST')] ,   dd3_thresh,  False, 1, 0)
    dd4_mask  = mask_data(data4_dem_d.loc[('HIST')], dd4_thresh,  False, 1, 0)

# Calculate F (compared to ENS)
    lws4_stat = get_f_score(ens_mask, lws4_mask, beta=1)
    lws3_stat = get_f_score(ens_mask, lws3_mask, beta=1)
    rl3_stat  = get_f_score(ens_mask, rl3_mask,  beta=1)
    rl4_stat  = get_f_score(ens_mask, rl4_mask,  beta=1)
    dd3_stat  = get_f_score(ens_mask, dd3_mask,  beta=1)
    dd4_stat  = get_f_score(ens_mask, dd4_mask,  beta=1)
    
# Calculate F (compared to each other)
    lws4_mask_val = mask_data(data4_REP_d.loc[('HIST')], lws4_thresh, True,  2, 0)
    lws3_mask_val = mask_data(data3_REP_d.loc[('HIST')], lws3_thresh, True,  2, 0)
    rl4_mask_val  = mask_data(data4_RL_d.loc[('HIST')],  rl4_thresh, False,  2, 0)

    lws4_lws3_stat = get_f_score(lws4_mask_val, lws3_mask, beta=1)
    rl4_rl3_stat = get_f_score(rl4_mask_val, rl3_mask, beta=1)
    lws4_rl4_stat = get_f_score(lws4_mask_val, rl4_mask, beta=1)
    lws3_rl3_stat = get_f_score(lws3_mask_val, rl3_mask, beta=1)
    
# Create Dataframe
    p_list.append( pd.concat([lws4_stat, lws3_stat, rl3_stat, rl4_stat, dd3_stat, dd4_stat], keys=['LWS4', 'LWS3', 'RL3', 'RL4', 'DD3', 'DD4'],   names=['Drought type']))
    p_list2.append(pd.concat([lws4_lws3_stat, rl4_rl3_stat, lws4_rl4_stat, lws3_rl3_stat],  keys=['LWS4-LWS3', 'RL4-RL3', 'LWS4-RL4', 'LWS3-RL3'], names=['Drought type']))
    print('Done '+str(p+1)+'/'+str(len(LWS_percs)))
stat_df  = pd.concat(p_list,  keys=LWS_percs, names=['Percentiles'])
stat_df2 = pd.concat(p_list2, keys=LWS_percs, names=['Percentiles'])


#%% Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
#stat_df.to_pickle(path_to_data+'validation_Otero22_detection_stats.pkl')
#stat_df.to_csv(   path_to_data+'validation_Otero22_detection_stats.csv',sep=';')


#%% Plot F / threshold (comparing to ENS / validation)
x=stat_df.index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)


for c in range(len(zones_szon)):
    fig, axs = plt.subplots(3, 2, figsize=(10,16))
    fig.suptitle('Validation of Otero Method for '+zones_szon[c]+' using '+ens_dataset+' ENS\n(Investigates only Yes/No occurance and no measure of severity)')
    
    # Event time series
    idx, idy = 0, [0,1]
    ax_big = plt.subplot(3, 1, 1)
    #ax_big.plot(x, stat_df.loc[(x, 'LWS4', 'F'),(zones_szon[c])], label='LWS4', color=dt_colors[0], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'LWS3', 'F'),(zones_szon[c])], label='LWS (PECD 3.1)', color=dt_colors[1], alpha=0.8)
    #ax_big.plot(x, stat_df.loc[(x, 'RL4',  'F'),(zones_szon[c])], label='RL4',  color=dt_colors[2], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'RL3',  'F'),(zones_szon[c])], label='RL (PECD 3.1)',  color=dt_colors[3], alpha=0.8)
    #ax_big.plot(x, stat_df.loc[(x, 'DD4',  'F'),(zones_szon[c])], label='DD4',  color=dt_colors[4], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'DD3',  'F'),(zones_szon[c])], label='DD (PECD 3.1)',  color=dt_colors[5], alpha=0.8)
    ax_big.set_ylabel('F-Score')
    ax_big.set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 1, 0
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4', 'TP'),(zones_szon[c])], label='LWS4', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3', 'TP'),(zones_szon[c])], label='LWS3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'TP'),(zones_szon[c])], label='RL4',  color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TP'),(zones_szon[c])], label='RL3',  color=dt_colors[3], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'TP'),(zones_szon[c])], label='DD4',  color=dt_colors[4], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TP'),(zones_szon[c])], label='DD3',  color=dt_colors[5], alpha=0.8)
    axs[idx, idy].set_ylabel('True Positives (of '+str(nr_of_pos[zones_szon[c]])+' in total)\n(DF detected, when ENS)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 1, 1
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4', 'TN'),(zones_szon[c])], label='LWS4', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3', 'TN'),(zones_szon[c])], label='LWS3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'TN'),(zones_szon[c])], label='RL4',  color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'TN'),(zones_szon[c])], label='RL3',  color=dt_colors[3], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'TN'),(zones_szon[c])], label='DD4',  color=dt_colors[4], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'TN'),(zones_szon[c])], label='DD3',  color=dt_colors[5], alpha=0.8)
    axs[idx, idy].set_ylabel('True Negatives (of '+str(nr_of_neg[zones_szon[c]])+' in total)\n(no DF detected, when no ENS)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 2, 0
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4', 'FP'),(zones_szon[c])], label='LWS4', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3', 'FP'),(zones_szon[c])], label='LWS3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'FP'),(zones_szon[c])], label='RL4',  color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FP'),(zones_szon[c])], label='RL3',  color=dt_colors[3], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'FP'),(zones_szon[c])], label='DD4',  color=dt_colors[4], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FP'),(zones_szon[c])], label='DD3',  color=dt_colors[5], alpha=0.8)
    axs[idx, idy].set_ylabel('False Positives (of '+str(nr_of_neg[zones_szon[c]])+' in total)\n(DF detected, when no ENS)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 2, 1
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS4', 'FN'),(zones_szon[c])], label='LWS4', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'LWS3', 'FN'),(zones_szon[c])], label='LWS3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL4',  'FN'),(zones_szon[c])], label='RL4',  color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'RL3',  'FN'),(zones_szon[c])], label='RL3',  color=dt_colors[3], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD4',  'FN'),(zones_szon[c])], label='DD4',  color=dt_colors[4], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'DD3',  'FN'),(zones_szon[c])], label='DD3',  color=dt_colors[5], alpha=0.8)
    axs[idx, idy].set_ylabel('False Negatives (of '+str(nr_of_pos[zones_szon[c]])+' in total)\n(No DF detected, when ENS)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    plt.tight_layout()
    plt.savefig(path_to_plot+'Validation/Validation_Otero22_'+ens_dataset+'_ENS_'+zones_szon[c]+'_REPweighted_'+str(w)+'.'+plot_format,dpi=300)
    plt.close()
    
    print('Saved '+path_to_plot+'Validation_Otero22_'+ens_dataset+'_ENS_'+zones_szon[c]+'_REPweighted_'+str(w)+'.'+plot_format)
    
#%% Plot F / threshold (comparing to each other)
x=stat_df.index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)


for c in range(len(zones_szon)):
    fig, axs = plt.subplots(3, 2, figsize=(10,16))
    fig.suptitle('Comparison of Otero Methods with each other for '+zones_szon[c]+' (ENS: '+ens_dataset+')\n(Investigates only Yes/No occurance and no measure of severity)')
    
    # Event time series
    idx, idy = 0, [0,1]
    ax_big = plt.subplot(3, 1, 1)
    ax_big.plot(x, stat_df2.loc[(x, 'LWS4-LWS3', 'F'),(zones_szon[c])], label='LWS4-LWS3', color=dt_colors[0], alpha=0.8)
    ax_big.plot(x, stat_df2.loc[(x, 'LWS3-RL3', 'F'),(zones_szon[c])], label='LWS3-RL3',   color=dt_colors[1], alpha=0.8)
    ax_big.plot(x, stat_df2.loc[(x, 'RL4-RL3', 'F'),(zones_szon[c])], label='RL4-RL3', color=dt_colors[2], alpha=0.8)
    ax_big.plot(x, stat_df2.loc[(x, 'LWS4-RL4', 'F'),(zones_szon[c])], label='LWS4-RL4',   color=dt_colors[3], alpha=0.8)
    ax_big.set_ylabel('F-Score')
    ax_big.set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 1, 0
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS4-LWS3', 'TP'),(zones_szon[c])], label='LWS4-LWS3', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS3-RL3', 'TP'),(zones_szon[c])], label='LWS3-RL3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'RL4-RL3', 'TP'),(zones_szon[c])], label='RL4-RL3', color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS4-RL4', 'TP'),(zones_szon[c])], label='LWS4-RL4',   color=dt_colors[3], alpha=0.8)
    axs[idx, idy].set_ylabel('True Positives (of '+str(nr_of_pos[zones_szon[c]])+' in total)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 1, 1
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS4-LWS3', 'TN'),(zones_szon[c])], label='LWS4-LWS3', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS3-RL3', 'TN'),(zones_szon[c])], label='LWS3-RL3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'RL4-RL3', 'TN'),(zones_szon[c])], label='RL4-RL3', color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS4-RL4', 'TN'),(zones_szon[c])], label='LWS4-RL4',   color=dt_colors[3], alpha=0.8)
    axs[idx, idy].set_ylabel('True Negatives (of '+str(nr_of_neg[zones_szon[c]])+' in total)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 2, 0
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS4-LWS3', 'FP'),(zones_szon[c])], label='LWS4-LWS3', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS3-RL3', 'FP'),(zones_szon[c])], label='LWS3-RL3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'RL4-RL3', 'FP'),(zones_szon[c])], label='RL4-RL3', color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS4-RL4', 'FP'),(zones_szon[c])], label='LWS4-RL4',   color=dt_colors[3], alpha=0.8)
    axs[idx, idy].set_ylabel('False Positives (of '+str(nr_of_neg[zones_szon[c]])+' in total)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 2, 1
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS4-LWS3', 'FN'),(zones_szon[c])], label='LWS4-LWS3', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS3-RL3', 'FN'),(zones_szon[c])], label='LWS3-RL3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'RL4-RL3', 'FN'),(zones_szon[c])], label='RL4-RL3', color=dt_colors[2], alpha=0.8)
    axs[idx, idy].plot(x, stat_df2.loc[(x, 'LWS4-RL4', 'FN'),(zones_szon[c])], label='LWS4-RL4',   color=dt_colors[3], alpha=0.8)
    axs[idx, idy].set_ylabel('False Negatives (of '+str(nr_of_pos[zones_szon[c]])+' in total)')
    axs[idx, idy].set_xlabel('LWS Percentile threshold (RL = 1-LWS)')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    plt.tight_layout()
    plt.savefig(path_to_plot+'Validation/Validation_Otero22_'+ens_dataset+'_DroughtTypes_'+zones_szon[c]+'_REPweighted_'+str(w)+'.'+plot_format,dpi=300)
    plt.close()
    
    print('Saved '+path_to_plot+'Validation_Otero22_'+ens_dataset+'_DroughtTypes_'+zones_szon[c]+'_REPweighted_'+str(w)+'.'+plot_format)
