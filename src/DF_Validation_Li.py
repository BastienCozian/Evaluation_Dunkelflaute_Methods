# -*- coding: utf-8 -*-
"""
This scripts does the validating of Method 1 (inspired by Li et al. 2021)
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
from Dunkelflaute_function_library import detect_drought_Li21
from Dunkelflaute_function_library import mask_df_by_entries
from Dunkelflaute_function_library import mask_data
from Dunkelflaute_function_library import get_f_score
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

#%% Set parameters

path_to_data      = 'D:/Dunkelflaute/Data/'
path_to_plot      = 'D:/Dunkelflaute/'
plot_format       = 'png'

# Countries (NUT0) of interest (must be a list of two letter abbreviations)
countries = ['DE','NL']

# ENS Data (which data to use?)
ens_dataset = 'AO' # 'AO' or 'ERAA23'
# Adequacy Outlook Scenario
ao_scen = 'W.v8' # 'W.v8' or 'S.v6' for either wind or solar dominated scenario (only if ens_dataset='AO')

dt_colors = ['green', 'purple']

# Percentile thresholds to investigate
CF_threshs = np.linspace(0.001,0.25,200)

# Target Years
ty_pecd3 = 2033
ty_pecd4 = 2030

# Models (and colors for plotting)
models = ['CMR5','ECE3','MEHR']
model_colors = ['blue', 'orange', 'red']
hist4_color = 'forestgreen'
hist3_color = 'purple'

# Pathways
pathways = ['SP245']
        
#%% Initializing variables for later use
scenarios = []
scen_colors = []
scenarios.append('HIST')
scen_colors.append(hist4_color)                                               
for p in range(len(pathways)):
    for m in range(len(models)):
        scenarios.append(pathways[p]+'/'+models[m])
        scen_colors.append(model_colors[m]) 
    
scen_names=[]
for s in scenarios:
    scen_names.append(s.replace('/','_'))
        
zones_peon = zones = get_zones(countries,'PEON')  
zones_szon = zones = get_zones(countries,'SZON')  
zones_peof = zones = get_zones(countries,'PEOF')  

scen_timespan = [42,51,51,51] # TODO: Automate the nr of years per scenario
scen_timespan_3 = 38 # How many years in PECD 3.1


#%% Load data
data4_CF_h = pd.read_pickle(path_to_data+'PECD4_CF_TY'+str(ty_pecd4)+'_national_hourly.pkl')
data3_CF_h = pd.read_pickle(path_to_data+'PECD3_CF_TY'+str(ty_pecd3)+'_national_hourly.pkl')
if ens_dataset=='AO':
    data3_ENS_h = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_hourly.pkl')
    data3_ENS_d = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_daily.pkl')
elif ens_dataset=='ERAA23':
    data3_ENS_h = pd.read_pickle(path_to_data+'ERAA23_ENS_TY2033_hourly.pkl')
    data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_ENS_TY2033_daily.pkl')
else:
    raise KeyError('ENS Dataset not existent!')

#data3_CF_sum_h = data3_CF_h.groupby(['Scenario','Date']).sum()
#data4_CF_sum_h = data4_CF_h.groupby(['Scenario','Date']).sum()



#%% For every percentile threshold
# TODO: the results F, TP, FN, ... are Series objects, so that pd.concat doesn't work
# Change the get_f_score method so that the results is a proper dataframe containing all the statistics
# Then delete the concatting of the STatistics and directly continue with appending the drought types

ens_mask_h = mask_data(data3_ENS_h, 0, False, 2, 0)
ens_mask_d = mask_data(data3_ENS_d, 0, False, 2, 0)
nr_of_pos_h = (ens_mask_h==2).sum()
nr_of_neg_h = (ens_mask_h==0).sum()
nr_of_pos_d = (ens_mask_d==2).sum()
nr_of_neg_d = (ens_mask_d==0).sum()

p_list  = []
p_list2 = []
for p in range(len(CF_threshs)):
    

# Mask the data / Detect Drought days WRONG CHANGE
# TODO: 1) Make the masking right
# Create a second, daily mask (for only >24h events)

# Detect drought hours and events
    cfd3_hours, cfd3_events = detect_drought_Li21(['HIST'],  zones_szon, data3_CF_h, CF_threshs[p])
    cfd4_hours, cfd4_events = detect_drought_Li21(scenarios, zones_szon, data4_CF_h, CF_threshs[p])
    cfd3_events_gret24 = cfd3_events[cfd3_events['Duration']>24]
    cfd4_events_gret24 = cfd4_events[cfd4_events['Duration']>24]
    
# Mask the data / Detect Drought days
    cf3_mask_h = mask_df_by_entries(data3_CF_h, cfd3_hours, ['HIST'],  1, 0).loc['HIST']
    cf4_mask_h = mask_df_by_entries(data4_CF_h, cfd4_hours, scenarios, 1, 0).loc['HIST']
    
    #cf3_mask_d = np.empty((len(cf3_mask_h.shape[0]/24), len(zones_szon)))
    #for d in range(cf3_mask_d.shape[0])
    #    cf3_mask_d[d,:] =
        

# Calculate F (compared to ENS)
    cf4_stat = get_f_score(ens_mask_h, cf4_mask_h, beta=1)
    cf3_stat = get_f_score(ens_mask_h, cf3_mask_h, beta=1)
    
# Calculate F (compared to each other)
    cf4_mask_val_h = mask_df_by_entries(data4_CF_h, cfd4_hours, scenarios, 2, 0).loc['HIST']
    cf43_stat = get_f_score(cf4_mask_val_h, cf3_mask_h, beta=1)

    
# Create Dataframe
    p_list.append( pd.concat([cf4_stat, cf3_stat, cf43_stat], keys=['CF4', 'CF3', 'CF4-CF3'],   names=['Drought type']))
    print('Done '+str(p+1)+'/'+str(len(CF_threshs)))
stat_df  = pd.concat(p_list,  keys=CF_threshs, names=['Thresholds'])


#%% Save the Data
# DataFrame: multiindex = (LWS3, LWS4, RL3, RL4),(F, TP, FN, FP, TN),(threshholds);  columns=countries
stat_df.to_pickle(path_to_data+'validation_Li21_detection_stats.pkl')
stat_df.to_csv(   path_to_data+'validation_Li21_detection_stats.csv',sep=';')


#%% Plot F / threshold (comparing to ENS / validation)
x=stat_df.index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)


for c in range(len(zones_szon)):
    fig, axs = plt.subplots(3, 2, figsize=(10,16))
    fig.suptitle('Validation of Li Method for '+zones_szon[c]+' using '+ens_dataset+' ENS\n(Investigates only Yes/No occurance and no measure of severity)')
    
    # Event time series
    idx, idy = 0, [0,1]
    ax_big = plt.subplot(3, 1, 1)
    ax_big.plot(x, stat_df.loc[(x, 'CF4', 'F'),(zones_szon[c])], label='CF4', color=dt_colors[0], alpha=0.8)
    ax_big.plot(x, stat_df.loc[(x, 'CF3', 'F'),(zones_szon[c])], label='CF3', color=dt_colors[1], alpha=0.8)
    ax_big.set_ylabel('F-Score')
    ax_big.set_xlabel('CF threshold')
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 1, 0
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF4', 'TP'),(zones_szon[c])], label='CF4', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF3', 'TP'),(zones_szon[c])], label='CF3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].set_ylabel('True Positives (of '+str(nr_of_pos_h[zones_szon[c]])+' in total)\n(DF detected, when ENS)')
    axs[idx, idy].set_xlabel('CF threshold')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 1, 1
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF4', 'TN'),(zones_szon[c])], label='CF4', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF3', 'TN'),(zones_szon[c])], label='CF3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].set_ylabel('True Negatives (of '+str(nr_of_neg_h[zones_szon[c]])+' in total)\n(no DF detected, when no ENS)')
    axs[idx, idy].set_xlabel('CF threshold')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 2, 0
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF4', 'FP'),(zones_szon[c])], label='CF4', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF3', 'FP'),(zones_szon[c])], label='CF3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].set_ylabel('False Positives (of '+str(nr_of_neg_h[zones_szon[c]])+' in total)\n(DF detected, when no ENS)')
    axs[idx, idy].set_xlabel('CF threshold')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 2, 1
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF4', 'FN'),(zones_szon[c])], label='CF4', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF3', 'FN'),(zones_szon[c])], label='CF3', color=dt_colors[1], alpha=0.8)
    axs[idx, idy].set_ylabel('False Negatives (of '+str(nr_of_pos_h[zones_szon[c]])+' in total)\n(No DF detected, when ENS)')
    axs[idx, idy].set_xlabel('CF threshold')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    plt.tight_layout()
    plt.savefig(path_to_plot+'Validation/Validation_Li21_'+ens_dataset+'_ENS_'+zones_szon[c]+'.'+plot_format,dpi=300)
    plt.close()
    
    print('Saved '+path_to_plot+'Validation_Li21_'+ens_dataset+'_ENS_'+zones_szon[c]+'.'+plot_format)
    
#%% Plot F / threshold (comparing to each other)
x=stat_df.index.levels[0] # LWS Percentile thresholds (1-x = RL percentile thresholds)


for c in range(len(zones_szon)):
    fig, axs = plt.subplots(3, 2, figsize=(10,16))
    fig.suptitle('Comparison of Li Methods with each other for '+zones_szon[c]+' (ENS: '+ens_dataset+')\n(Investigates only Yes/No occurance and no measure of severity)')
    
    # Event time series
    idx, idy = 0, [0,1]
    ax_big = plt.subplot(3, 1, 1)
    ax_big.plot(x, stat_df.loc[(x, 'CF4-CF3', 'F'),(zones_szon[c])], label='CF4-CF3', color=dt_colors[0], alpha=0.8)
    ax_big.set_ylabel('F-Score')
    ax_big.set_xlabel('CF threshold')
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    ax_big.legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 1, 0
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF4-CF3', 'TP'),(zones_szon[c])], label='CF4-CF3', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].set_ylabel('True Positives (of '+str(nr_of_pos_h[zones_szon[c]])+' in total)')
    axs[idx, idy].set_xlabel('CF threshold')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 1, 1
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF4-CF3', 'TN'),(zones_szon[c])], label='CF4-CF3', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].set_ylabel('True Negatives (of '+str(nr_of_neg_h[zones_szon[c]])+' in total)')
    axs[idx, idy].set_xlabel('CF threshold')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 2, 0
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF4-CF3', 'FP'),(zones_szon[c])], label='CF4-CF3', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].set_ylabel('False Positives (of '+str(nr_of_neg_h[zones_szon[c]])+' in total)')
    axs[idx, idy].set_xlabel('CF threshold')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
    
    idx, idy = 2, 1
    axs[idx, idy].plot(x, stat_df.loc[(x, 'CF4-CF3', 'FN'),(zones_szon[c])], label='CF4-CF3', color=dt_colors[0], alpha=0.8)
    axs[idx, idy].set_ylabel('False Negatives (of '+str(nr_of_pos_h[zones_szon[c]])+' in total)')
    axs[idx, idy].set_xlabel('CF threshold')
    axs[idx, idy].yaxis.set_major_locator(MaxNLocator(integer=True))
    #axs[idx, idy].set_ylim(ymin-0.1*yabs, ymax+0.1*yabs)
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    plt.tight_layout()
    plt.savefig(path_to_plot+'Validation/Validation_Li21_'+ens_dataset+'_DroughtTypes_'+zones_szon[c]+'.'+plot_format,dpi=300)
    plt.close()
    
    print('Saved '+path_to_plot+'Validation_Li21_'+ens_dataset+'_DroughtTypes_'+zones_szon[c]+'.'+plot_format)
