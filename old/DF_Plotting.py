# -*- coding: utf-8 -*-
"""
This plot loads in the results of energy drought detection by method 1 and method 2 and creates various plots.
It also loads in the underlying data used for detection and ENS data for validation plots.

For questions, refer to benjamin.biewald@tennet.eu
"""

#%% Import packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from matplotlib.ticker import MaxNLocator
from Dunkelflaute_function_library import get_zones
from Dunkelflaute_function_library import get_thresholds
from Dunkelflaute_function_library import get_df_timerange
from Dunkelflaute_function_library import lin_reg
from Dunkelflaute_function_library import mask_data
from Dunkelflaute_function_library import mask_df_by_entries
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

#%% Specify parameters
path_to_data      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'
path_to_plot      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'
#path_to_data      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/Data/'  #'D:/Dunkelflaute/Data/'
#path_to_plot      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'      #'D:/Dunkelflaute/'
plot_format  = 'png'

droughttypes = ['LWS', 'RL', 'DD', 'CF']
# ENS Data (which data to use?)
ens_dataset = 'ERAA23' # 'AO' or 'ERAA23'
# Adequacy Outlook Scenario
ao_scen = 'W.v8' # 'W.v8' or 'S.v6' for either wind or solar dominated scenario (only if ens_dataset='AO')

# Units for plotting
unit = 'TWh'
scaling = 0.000001 # 'unit' in MWh (bc original data is in MWh)

# Countries (NUT0) of interest (must be a list of two letter abbreviations)
countries = ['DE','NL']

# Target Years
ty_pecd3 = 2033
ty_pecd4 = 2050

# Percentile thresholds & capacity reference year from Otero et al. 2022
# Paper used 0.1 and 0.9, validation yielded 0.01 and 0.99 / 0.98 as best fits
LWS_percentile = 0.01 # 0.01
RL_percentile  = 0.99 # 0.98
DD_percentile  = 0.99
# Capacity Factor Threshold for all technologies (see Li et al. 2021)
CF_thresh      = 0.2  # original: 0.2

# Droughttypea
droughttypes = ['LWS', 'RL', 'DD', 'CFD']
dt_colors = ['maroon', 'magenta', 'navy', 'gray']

# Models
models = ['CMR5','ECE3','MEHR']
model_colors = ['blue', 'orange', 'red']
hist3_color = 'purple'

# Pathways
pathways = ['SP245']

# Technologies / Variables of Interest and corresponding header sizes
techs = ['SPV', 'WOF', 'WON']#,'TAW'] 
techs_linestyle = ['solid', 'dashed', 'dotted']
aggs = ['PEON','PEOF','PEON']#,'SZON'] 
tech_agg = ['SPV/PEON', 'WOF/PEOF', 'WON/PEON']#, 'TAW/SZON']

scenarios = []
scen_colors = []
scenarios.append('HIST')
scen_colors.append('forestgreen')                                                       
for p in range(len(pathways)):
    for m in range(len(models)):
        scenarios.append(models[m]+'/'+pathways[p])
        scen_colors.append(model_colors[m]) 
    
scen_names=[]
for s in scenarios:
    scen_names.append(s.replace('/','_'))
        
zones_peon = zones = get_zones(countries,'PEON')  
zones_szon = zones = get_zones(countries,'SZON')  
zones_peof = zones = get_zones(countries,'PEOF')  

scen_timespan = [42,51,51,51] # TODO: Automate the nr of years per scenario
scen_timespan_3 = 38 # How many years in PECD 3.1

#%% Load in Data
data4_REP_d = pd.read_pickle(path_to_data+'PECD4_Generation_TY'+str(ty_pecd4)+'_national_daily.pkl') * scaling
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
cfd4_events = pd.read_pickle(path_to_data+'CFD4_TY'+str(ty_pecd4)+'_Thresh_'+str(CF_thresh)+'_events.pkl')

lws3_events = pd.read_pickle(path_to_data+'LWS3_TY'+str(ty_pecd3)+'_Thresh_'+str(LWS_percentile)+'_events.pkl')
rl3_events  = pd.read_pickle(path_to_data+'RL3_TY'+str(ty_pecd3)+'_Thresh_'+str(RL_percentile)+'_events.pkl')
dd3_events  = pd.read_pickle(path_to_data+'DD3_TY'+str(ty_pecd3)+'_Thresh_'+str(DD_percentile)+'_events.pkl')
cfd3_events = pd.read_pickle(path_to_data+'CFD3_TY'+str(ty_pecd3)+'_Thresh_'+str(CF_thresh)+'_events.pkl')

cfd4_hours = pd.read_pickle(path_to_data+'CFD4_TY'+str(ty_pecd4)+'_Thresh_'+str(CF_thresh)+'_hours.pkl')
cfd3_hours = pd.read_pickle(path_to_data+'CFD3_TY'+str(ty_pecd3)+'_Thresh_'+str(CF_thresh)+'_hours.pkl')

if ens_dataset=='AO':
    data3_ENS_d = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_daily.pkl')
    data3_ENS_h = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_hourly.pkl')
elif ens_dataset=='ERAA23':
    data3_ENS_d = pd.read_pickle(path_to_data+'ERAA23_ENS_TY2033_daily.pkl')
    data3_ENS_h = pd.read_pickle(path_to_data+'AO_'+ao_scen+'_ENS_hourly.pkl')
else:
    raise KeyError('ENS Dataset not existent!')


dur_max = int(np.round(np.amax([np.amax(lws4_events['Duration']),np.amax(rl4_events['Duration']),np.amax(dd4_events['Duration']),np.amax(cfd4_events['Duration'])/24,np.amax(lws3_events['Duration']),np.amax(rl3_events['Duration']),np.amax(dd3_events['Duration']),np.amax(cfd3_events['Duration'])/24]),0))
sev_max = np.amax([np.amax(lws4_events['Severity (adapted)']),np.amax(rl4_events['Severity (adapted)']),np.amax(dd4_events['Severity (adapted)']),np.amax(lws3_events['Severity (adapted)']),np.amax(rl3_events['Severity (adapted)']),np.amax(dd3_events['Severity (adapted)'])])
sev_min = np.amin([np.amin(lws4_events['Severity (adapted)']),np.amin(rl4_events['Severity (adapted)']),np.amin(dd4_events['Severity (adapted)']),np.amin(lws3_events['Severity (adapted)']),np.amin(rl3_events['Severity (adapted)']),np.amin(dd3_events['Severity (adapted)'])])

#%% Calculate Thresholds
# Histogramtop lines with thresholds
# For each Zone individual plot, and each Droughttype in panel (so 4 panels)

lws4_thresh, lws4_sigma = get_thresholds(data4_REP_d.loc[('HIST')], LWS_percentile, start_date='1980-01-01', end_date='2019-12-31', empirical=True)
lws3_thresh, lws3_sigma = get_thresholds(data3_REP_d.loc[('HIST')], LWS_percentile, start_date='1980-01-01', end_date='2019-12-31', empirical=True)
rl4_thresh,  rl4_sigma  = get_thresholds(data4_RL_d.loc[('HIST')],  RL_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)
rl3_thresh,  rl3_sigma  = get_thresholds(data3_RL_d.loc[('HIST')],  RL_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)
dd4_thresh,  dd4_sigma  = get_thresholds(data4_dem_d.loc[('HIST')], DD_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)
dd3_thresh,  dd3_sigma  = get_thresholds(data3_dem_d.loc[('HIST')], DD_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)

#%% Plot Distribution of underlying data (Generation, RL, Demand, CF)
binsize=40

for c in range(len(zones_szon)):
    # PECD 4
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    fig.suptitle('Distribution of underlying data for '+zones_szon[c]+'\n(PECDv4.1, TY'+str(ty_pecd4)+')')
    
    idx, idy = 0, 0
    bins = np.linspace(0,np.amax(data4_REP_d[zones_szon[c]]),binsize)
    for s in range(len(scenarios)):
        hist = np.histogram(data4_REP_d.loc[(scenarios[s]),(zones_szon[c])], bins=bins)
        x = hist[1][:-1] + np.diff(hist[1])[0]/2
        axs[idx, idy].plot(x, hist[0], label=scenarios[s]+' (PECD 4.1)', color=scen_colors[s])
    axs[idx, idy].set_title('Daily REP')
    axs[idx, idy].axvline(lws4_thresh[c], alpha=0.5, color='black')
    axs[idx, idy].set_xlim(0,np.amax(data4_REP_d[zones_szon[c]]))
    axs[idx, idy].set_ylabel('Number of Days')
    axs[idx, idy].set_xlabel('Daily Generation ['+unit+']')
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 0, 1
    bins = np.linspace(np.amin(data4_RL_d[zones_szon[c]]),np.amax(data4_RL_d[zones_szon[c]]),binsize)
    for s in range(len(scenarios)):
        hist = np.histogram(data4_RL_d.loc[(scenarios[s]),(zones_szon[c])], bins=bins)
        x = hist[1][:-1] + np.diff(hist[1])[0]/2
        axs[idx, idy].plot(x, hist[0], label=scenarios[s]+' (PECD 4.1)', color=scen_colors[s])
    axs[idx, idy].set_title('Daily RL')
    axs[idx, idy].axvline(rl4_thresh[c], alpha=0.5, color='black')
    axs[idx, idy].set_xlim(np.amin(data4_RL_d[zones_szon[c]]),np.amax(data4_RL_d[zones_szon[c]]))
    axs[idx, idy].set_ylabel('Number of Days')
    axs[idx, idy].set_xlabel('Daily Residual Load ['+unit+']')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 1, 0
    bins = np.linspace(np.amin(data4_dem_d[zones_szon[c]]),np.amax(data4_dem_d[zones_szon[c]]),binsize)
    for s in range(len(scenarios)):
        hist = np.histogram(data4_dem_d.loc[(scenarios[s]),(zones_szon[c])], bins=bins)
        x = hist[1][:-1] + np.diff(hist[1])[0]/2
        axs[idx, idy].plot(x, hist[0], label=scenarios[s]+' (PECD 4.1)', color=scen_colors[s])
    axs[idx, idy].set_title('Daily electric energy demand')
    axs[idx, idy].axvline(dd4_thresh[c], alpha=0.5, color='black')
    axs[idx, idy].set_xlim(np.amin(data4_dem_d[zones_szon[c]]),np.amax(data4_dem_d[zones_szon[c]]))
    axs[idx, idy].set_ylabel('Number of Days')
    axs[idx, idy].set_xlabel('Daily demand ['+unit+']')
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 1, 1
    bins = np.linspace(np.amin(data4_CF_h[zones_szon[c]]),np.amax(data4_CF_h[zones_szon[c]]),binsize)
    for t in range(len(techs)):
        for s in range(len(scenarios)):
            hist = np.histogram(data4_CF_h.loc[(techs[t],scenarios[s]),(zones_szon[c])], bins=bins)
            x = hist[1][:-1] + np.diff(hist[1])[0]/2
            axs[idx, idy].plot(x, hist[0], label='HIST (PECD 3.1, '+techs[t]+')', linestyle=techs_linestyle[t], color=scen_colors[s])
    axs[idx, idy].set_title('Hourly Capacity Factors')
    axs[idx, idy].axvline(CF_thresh, alpha=0.5, color='black')
    axs[idx, idy].set_xlim(np.amin(data4_CF_h[zones_szon[c]]),np.amax(data4_CF_h[zones_szon[c]]))
    axs[idx, idy].set_ylim(0,np.amax(hist[0])*1.5) # just so that the many zeros of SPV don't put the y axis out of scale
    axs[idx, idy].set_ylabel('Number of Hours')
    axs[idx, idy].set_xlabel('Hourly capacity factor')
    axs[idx, idy].legend(facecolor="white", loc='upper center', framealpha=1)
    
    plt.tight_layout()
    plt.savefig(path_to_plot+'Other_figures/DistributionThreshold_PECD4_'+zones_szon[c]+'_TY'+str(ty_pecd4)+'.'+plot_format,dpi=300)
    plt.close()
    
    
    # PECD 3
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    fig.suptitle('Distribution of underlying data for '+zones_szon[c]+'\n(PECDv3.1, TY'+str(ty_pecd3)+')')
    
    # Event time series
    idx, idy = 0, 0
    bins = np.linspace(0,np.amax(data3_REP_d[zones_szon[c]]),binsize)
    hist = np.histogram(data3_REP_d.loc[('HIST'),(zones_szon[c])], bins=bins)
    x = hist[1][:-1] + np.diff(hist[1])[0]/2
    axs[idx, idy].plot(x, hist[0], label='HIST (PECD 3.1)', color=hist3_color)
    axs[idx, idy].set_title('Daily REP')
    axs[idx, idy].axvline(lws3_thresh[c], alpha=0.5, color='black')
    axs[idx, idy].set_xlim(0,np.amax(data3_REP_d[zones_szon[c]]))
    axs[idx, idy].set_ylabel('Number of Days')
    axs[idx, idy].set_xlabel('Generation ['+unit+']')
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 0, 1
    bins = np.linspace(np.amin(data3_RL_d[zones_szon[c]]),np.amax(data3_RL_d[zones_szon[c]]),binsize)
    hist = np.histogram(data3_RL_d.loc[('HIST'),(zones_szon[c])], bins=bins)
    x = hist[1][:-1] + np.diff(hist[1])[0]/2
    axs[idx, idy].plot(x, hist[0], label='HIST (PECD 3.1)', color=hist3_color)
    axs[idx, idy].set_title('Daily Residual Load')
    axs[idx, idy].axvline(rl3_thresh[c], alpha=0.5, color='black')
    axs[idx, idy].set_xlim(np.amin(data3_RL_d[zones_szon[c]]),np.amax(data3_RL_d[zones_szon[c]]))
    axs[idx, idy].set_ylabel('Number of Days')
    axs[idx, idy].set_xlabel('Residual Load ['+unit+']')
    axs[idx, idy].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 1, 0
    bins = np.linspace(np.amin(data3_dem_d[zones_szon[c]]),np.amax(data3_dem_d[zones_szon[c]]),binsize)
    hist = np.histogram(data3_dem_d.loc[('HIST'),(zones_szon[c])], bins=bins)
    x = hist[1][:-1] + np.diff(hist[1])[0]/2
    axs[idx, idy].plot(x, hist[0], label='HIST (PECD 3.1)', color=hist3_color)
    axs[idx, idy].set_title('Daily Electric Energy Demand')
    axs[idx, idy].axvline(dd3_thresh[c], alpha=0.5, color='black')
    axs[idx, idy].set_xlim(np.amin(data3_dem_d[zones_szon[c]]),np.amax(data3_dem_d[zones_szon[c]]))
    axs[idx, idy].set_ylabel('Number of Days')
    axs[idx, idy].set_xlabel('Demand ['+unit+']')
    axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
    
    idx, idy = 1, 1
    bins = np.linspace(np.amin(data3_CF_h[zones_szon[c]]),np.amax(data3_CF_h[zones_szon[c]]),binsize)
    for t in range(len(techs)):
        hist = np.histogram(data3_CF_h.loc[(techs[t],'HIST'),(zones_szon[c])], bins=bins)
        x = hist[1][:-1] + np.diff(hist[1])[0]/2
        axs[idx, idy].plot(x, hist[0], label='HIST (PECD 3.1, '+techs[t]+')', color=hist3_color, linestyle=techs_linestyle[t])
    axs[idx, idy].set_title('Hourly Capacity Factors')
    axs[idx, idy].axvline(CF_thresh, alpha=0.5, color='black')
    axs[idx, idy].set_xlim(np.amin(data3_CF_h[zones_szon[c]]),np.amax(data3_CF_h[zones_szon[c]]))
    axs[idx, idy].set_ylim(0,np.amax(hist[0])*1.5) # just so that the many zeros of SPV don't put the y axis out of scale
    axs[idx, idy].set_ylabel('Number of Hours')
    axs[idx, idy].set_xlabel('Hourly capacity factor')
    axs[idx, idy].legend(facecolor="white", loc='upper center', framealpha=1)
    
    plt.tight_layout()
    plt.savefig(path_to_plot+'Other_figures/DistributionThreshold_PECD3_'+zones_szon[c]+'_TY'+str(ty_pecd3)+'.'+plot_format,dpi=300)
    plt.close()

#%% Calculate Annual Statistics
# Perhaps as annual statistics? SO quasi Hist:
# x=years, y=nr of events / summed up duration / summed up severity

def sum_up_to_years_wSev(df_events):
    annual_stat = df_events[['Duration','Severity (adapted)']].groupby(['Scenario','Countries',df_events['Startdate'].dt.year]).sum()
    annual_stat['Nr of events'] = df_events[['Duration','Severity (adapted)']].groupby(['Scenario','Countries',df_events['Startdate'].dt.year]).size()
    annual_stat.index.rename(['Year'],level=[2], inplace=True)
    return annual_stat

def sum_up_to_years_woSev(df_events):
    annual_stat = pd.DataFrame()
    annual_stat['Duration'] = df_events['Duration'].groupby(['Scenario','Countries',df_events['Startdate'].dt.year]).sum()/24
    annual_stat['Nr of events'] = df_events['Duration'].groupby(['Scenario','Countries',df_events['Startdate'].dt.year]).size()
    annual_stat.index.rename(['Year'],level=[2], inplace=True)
    return annual_stat

lws4_annual_stats = sum_up_to_years_wSev(lws4_events)
lws3_annual_stats = sum_up_to_years_wSev(lws3_events)
rl4_annual_stats = sum_up_to_years_wSev(rl4_events)
rl3_annual_stats = sum_up_to_years_wSev(rl3_events)
dd4_annual_stats = sum_up_to_years_wSev(dd4_events)
dd3_annual_stats = sum_up_to_years_wSev(dd3_events)
cfd4_annual_stats = sum_up_to_years_woSev(cfd4_events)
cfd3_annual_stats = sum_up_to_years_woSev(cfd3_events)

drougth_stats = [lws4_annual_stats, rl4_annual_stats, dd4_annual_stats, cfd4_annual_stats]

#%% Plot Time/Duration & Time/Severity + Linear regression (Annual Statistics)

for c in range(len(zones_szon)):
    # For each Zone and each Droughttype (3 panels for Nr of events, sum of duration and sum of severity (adapted))
    for d in range(len(droughttypes)):
        fig, axs = plt.subplots(3, 1, figsize=(10,8))
        
        idx, idy = 0, 0
        for s in range(len(scenarios)):
            x=drougth_stats[d].loc[(scenarios[s],zones_szon[c])].index
            y=drougth_stats[d].loc[(scenarios[s],zones_szon[c]),('Nr of events')]
            axs[idx].plot(x, y, label=scenarios[s]+' (PECDv4.1)', color=scen_colors[s], marker='.', linestyle='None', alpha=0.5)
            lr = lin_reg(x,y)
            if lr[4]<0.05:
                axs[idx].plot(lr[0], color=scen_colors[s])
            else:
                axs[idx].plot(lr[0], color=scen_colors[s], linestyle='dotted')
        axs[idx].set_title('Number of Drought events per year ('+zones_szon[c]+')')
        axs[idx].set_ylabel('Number of Events')
        axs[idx].set_xlabel('Year')
        axs[idx].legend(facecolor="white", loc='upper center', framealpha=1)
        
        idx, idy = 1, 0
        for s in range(len(scenarios)):
            x=drougth_stats[d].loc[(scenarios[s],zones_szon[c])].index
            y=drougth_stats[d].loc[(scenarios[s],zones_szon[c]),('Duration')]
            axs[idx].plot(x, y, label=scenarios[s]+' (PECDv4.1)', color=scen_colors[s], marker='.', linestyle='None', alpha=0.5)
            lr = lin_reg(x,y)
            if lr[4]<0.05:
                axs[idx].plot(lr[0], color=scen_colors[s])
            else:
                axs[idx].plot(lr[0], color=scen_colors[s], linestyle='dotted')
        axs[idx].set_title('Total number of drought days per year ('+zones_szon[c]+')')
        axs[idx].set_ylabel('Summed up Durations [d]')
        axs[idx].set_xlabel('Year')
        axs[idx].legend(facecolor="white", loc='upper center', framealpha=1)
        
        if not droughttypes[d]=='CFD':
            idx, idy = 2, 0
            for s in range(len(scenarios)):
                x=drougth_stats[d].loc[(scenarios[s],zones_szon[c])].index
                y=drougth_stats[d].loc[(scenarios[s],zones_szon[c]),('Severity (adapted)')]
                axs[idx].plot(x, y, label=scenarios[s]+' (PECDv4.1)', color=scen_colors[s], marker='.', linestyle='None', alpha=0.5)
                lr = lin_reg(x,y)
                if lr[4]<0.05:
                    axs[idx].plot(lr[0], color=scen_colors[s])
                else:
                    axs[idx].plot(lr[0], color=scen_colors[s], linestyle='dotted')
            axs[idx].set_title('Sum of severities of all events per year ('+zones_szon[c]+')')
            axs[idx].set_ylabel('Summed up Severity')
            axs[idx].set_xlabel('Year')
            axs[idx].legend(facecolor="white", loc='upper center', framealpha=1)
        
        
        plt.tight_layout()
        plt.savefig(path_to_plot+'Other_figures/DF_AnnualStatistics_'+droughttypes[d]+'_ScenarioComparison_'+zones_szon[c]+'.'+plot_format,dpi=300)
        plt.close()
    
    # For each Zone and each scenario (3 panels for Nr of events, sum of duration and sum of severity (adapted))
    for s in range(len(scenarios)):
        fig, axs = plt.subplots(3, 1, figsize=(10,8))
        
        idx, idy = 0, 0
        for d in range(len(droughttypes)):
            x=drougth_stats[d].loc[(scenarios[s],zones_szon[c])].index
            y=drougth_stats[d].loc[(scenarios[s],zones_szon[c]),('Nr of events')]
            axs[idx].plot(x, y, label=droughttypes[d]+' (PECDv4.1)', color=dt_colors[d], marker='.', linestyle='None', alpha=0.5)
            lr = lin_reg(x,y)
            if lr[4]<0.05:
                axs[idx].plot(lr[0], color=dt_colors[d])
            else:
                axs[idx].plot(lr[0], color=dt_colors[d], linestyle='dotted')
        axs[idx].set_title('Number of Drought events per year ('+zones_szon[c]+')')
        axs[idx].set_ylabel('Number of Events')
        axs[idx].set_xlabel('Year')
        axs[idx].legend(facecolor="white", loc='upper center', framealpha=1)
        
        idx, idy = 1, 0
        for d in range(len(droughttypes)):
            x=drougth_stats[d].loc[(scenarios[s],zones_szon[c])].index
            y=drougth_stats[d].loc[(scenarios[s],zones_szon[c]),('Duration')]
            axs[idx].plot(x, y, label=droughttypes[d]+' (PECDv4.1)', color=dt_colors[d], marker='.', linestyle='None', alpha=0.5)
            lr = lin_reg(x,y)
            if lr[4]<0.05:
                axs[idx].plot(lr[0], color=dt_colors[d])
            else:
                axs[idx].plot(lr[0], color=dt_colors[d], linestyle='dotted')
        axs[idx].set_title('Total number of drought days per year ('+zones_szon[c]+')')
        axs[idx].set_ylabel('Summed up Durations [d]')
        axs[idx].set_xlabel('Year')
        axs[idx].legend(facecolor="white", loc='upper center', framealpha=1)
        
        idx, idy = 2, 0
        for d in range(len(droughttypes)-1):
            x=drougth_stats[d].loc[(scenarios[s],zones_szon[c])].index
            y=drougth_stats[d].loc[(scenarios[s],zones_szon[c]),('Severity (adapted)')]
            axs[idx].plot(x, y, label=droughttypes[d]+' (PECDv4.1)', color=dt_colors[d], marker='.', linestyle='None', alpha=0.5)
            lr = lin_reg(x,y)
            if lr[4]<0.05:
                axs[idx].plot(lr[0], color=dt_colors[d])
            else:
                axs[idx].plot(lr[0], color=dt_colors[d], linestyle='dotted')
        axs[idx].set_title('Sum of severities of all events per year ('+zones_szon[c]+')')
        axs[idx].set_ylabel('Summed up Severity')
        axs[idx].set_xlabel('Year')
        axs[idx].legend(facecolor="white", loc='upper center', framealpha=1)
        
        
        plt.tight_layout()
        plt.savefig(path_to_plot+'Other_figures/DF_AnnualStatistics_'+scenarios[s].replace('/','_')+'_DroughttypesComparison_'+zones_szon[c]+'.'+plot_format,dpi=300)
        plt.close()
        

        



#%% Plot Hist-top-lines of Duration & Severity for 30yr periods
# compare: 1980-2010 & 2035-2065
# For each Zone and each Droughttype

def startdate_between(df_events, start_date, end_date):
    selection = df_events[(df_events['Startdate']>start_date)&(df_events['Startdate']<end_date)]
    return selection

totyears = 30 # total years of timeframe
droughttypes_df = [lws4_events, rl4_events, dd4_events, cfd4_events]
hist_df = []
futu_df = []
dur_max_l = []
sev_min_l = []
sev_max_l = []
for d in range(len(droughttypes_df)):
    hist_df.append(startdate_between(droughttypes_df[d], '1980-01-01', '2009-12-31'))
    futu_df.append(startdate_between(droughttypes_df[d], '2035-01-01', '2064-12-31'))
    dur_max_l.append(np.amax(startdate_between(droughttypes_df[d], '2035-01-01', '2064-12-31')['Duration']))
    
    if droughttypes[d]=='CFD':
        hist_df[d]['Duration'] = hist_df[d]['Duration']/24
        futu_df[d]['Duration'] = futu_df[d]['Duration']/24
        dur_max_l[d] = int(np.round(dur_max_l[d]/24))
    else:
        sev_min_l.append(np.amin(startdate_between(droughttypes_df[d], '2035-01-01', '2064-12-31')['Severity (adapted)']))
        sev_max_l.append(np.amax(startdate_between(droughttypes_df[d], '2035-01-01', '2064-12-31')['Severity (adapted)']))
    

for c in range(len(zones_szon)):
    for d in range(len(droughttypes)):
        fig, axs = plt.subplots(2, 2, figsize=(14,10))
        
        # Duration
        idx, idy = 0, 0
        bins = np.linspace(0.5,dur_max_l[d]-0.5,dur_max_l[d])
        hist = np.histogram(hist_df[d].loc[(scenarios[0],zones_szon[c]),('Duration')], bins=bins)
        x = hist[1][:-1] + np.diff(hist[1])[0]/2
        axs[idx, idy].plot(x, hist[0], label=scenarios[0]+' (PECD 4.1)', color=scen_colors[0], marker='.', alpha=0.8)
        for s in range(1,len(scenarios)):
            hist = np.histogram(futu_df[d].loc[(scenarios[s],zones_szon[c]),('Duration')], bins=bins)
            x = hist[1][:-1] + np.diff(hist[1])[0]/2
            axs[idx, idy].plot(x, hist[0], label=scenarios[s]+' (PECD 4.1)', color=scen_colors[s], marker='.', alpha=0.8)
        axs[idx, idy].set_title('Duration of Drought Events ('+zones_szon[c]+')')
        axs[idx, idy].set_ylabel('Number of Events')
        axs[idx, idy].set_xlabel('Duration [d]')
        #axs[idx, idy].set_xlim(bins[0]-np.diff(bins)[0]/2,bins[-1]+np.diff(bins)[0]/2)
        axs[idx, idy].set_xlim(bins[0],bins[-1])
        axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
        
        idx, idy = 1, 0
        bins = np.linspace(0.5,dur_max_l[d]-0.5,dur_max_l[d])
        hist = np.histogram(hist_df[d].loc[(scenarios[0],zones_szon[c]),('Duration')], bins=bins)
        x = hist[1][:-1] + np.diff(hist[1])[0]/2
        axs[idx, idy].plot(x, totyears/hist[0], label=scenarios[0]+' (PECD 4.1)', color=scen_colors[0], linestyle='None', marker='o', alpha=0.8)
        for s in range(1,len(scenarios)):
            hist = np.histogram(futu_df[d].loc[(scenarios[s],zones_szon[c]),('Duration')], bins=bins)
            x = hist[1][:-1] + np.diff(hist[1])[0]/2
            axs[idx, idy].plot(x, totyears/hist[0], label=scenarios[s]+' (PECD 4.1)', color=scen_colors[s], linestyle='None', marker='o', alpha=0.8)
        axs[idx, idy].set_title('Duration of Drought Events ('+zones_szon[c]+')')
        axs[idx, idy].set_ylabel('Return time [yr]')
        axs[idx, idy].set_xlabel('Duration [d]')
        axs[idx, idy].set_xlim(bins[0],bins[-1])
        axs[idx, idy].set_yscale('log')
        axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
        
        # Severity
        if not droughttypes[d]=='CFD':
            idx, idy = 0, 1
            bins = np.linspace(sev_min_l[d],sev_max_l[d],20)
            hist = np.histogram(hist_df[d].loc[(scenarios[0],zones_szon[c]),('Severity (adapted)')], bins=bins)
            x = hist[1][:-1] + np.diff(hist[1])[0]/2
            axs[idx, idy].plot(x, hist[0], label=scenarios[0]+' (PECD 4.1)', color=scen_colors[0], marker='.', alpha=0.8)
            for s in range(1,len(scenarios)):
                hist = np.histogram(futu_df[d].loc[(scenarios[s],zones_szon[c]),('Severity (adapted)')], bins=bins)
                x = hist[1][:-1] + np.diff(hist[1])[0]/2
                axs[idx, idy].plot(x, hist[0], label=scenarios[s]+' (PECD 4.1)', color=scen_colors[s], marker='.', alpha=0.8)
            axs[idx, idy].set_title('Severity of Drought Events ('+zones_szon[c]+')')
            axs[idx, idy].set_ylabel('Number of Events')
            axs[idx, idy].set_xlabel('Severity')
            axs[idx, idy].set_xlim(bins[0],bins[-1])
            axs[idx, idy].legend(facecolor="white", loc='upper right', framealpha=1)
            
            idx, idy = 1, 1
            bins = np.linspace(sev_min_l[d],sev_max_l[d],20)
            hist = np.histogram(hist_df[d].loc[(scenarios[0],zones_szon[c]),('Severity (adapted)')], bins=bins)
            x = hist[1][:-1] + np.diff(hist[1])[0]/2
            axs[idx, idy].plot(x, totyears/hist[0], label=scenarios[0]+' (PECD 4.1)', color=scen_colors[0], linestyle='None', marker='o', alpha=0.8)
            for s in range(1,len(scenarios)):
                hist = np.histogram(futu_df[d].loc[(scenarios[s],zones_szon[c]),('Severity (adapted)')], bins=bins)
                x = hist[1][:-1] + np.diff(hist[1])[0]/2
                axs[idx, idy].plot(x, totyears/hist[0], label=scenarios[s]+' (PECD 4.1)', color=scen_colors[s], linestyle='None', marker='o', alpha=0.8)
            axs[idx, idy].set_title('Severity of Drought Events ('+zones_szon[c]+')')
            axs[idx, idy].set_ylabel('Return Time [yr]')
            axs[idx, idy].set_xlabel('Severity')
            axs[idx, idy].set_xlim(bins[0],bins[-1])
            axs[idx, idy].set_yscale('log')
            axs[idx, idy].legend(facecolor="white", loc='lower right', framealpha=1)
        
        plt.tight_layout()
        plt.savefig(path_to_plot+'Other_figures/DF_EventDistribution_'+droughttypes[d]+'_ScenarioComparison_'+zones_szon[c]+'.'+plot_format,dpi=300)
        plt.close()

#%% Plot Correlation of Duration/Severity + Linear regression
# all zones and the whole timeframe combined
# For each droughttype a panel
for c in range(len(zones_szon)):
    
    fig, axs = plt.subplots(1, 3, figsize=(14,4))
    
    idx, idy = 0, 0
    x=lws3_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=lws3_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    axs[idx].scatter(x=x, y=y, c=hist3_color, label='HIST (3.1)', alpha=0.2)
    axs[idx].plot(lin_reg(x,y)[0], c=hist3_color)
    for s in range(len(scenarios)):
        x=lws4_events.loc[(scenarios[s],zones_szon[c]),('Duration')]
        y=lws4_events.loc[(scenarios[s],zones_szon[c]),('Severity (adapted)')]
        axs[idx].scatter(x=x, y=y, c=scen_colors[s], label=scenarios[s]+' (PECD 4.1)', alpha=0.2)
        axs[idx].plot(lin_reg(x,y)[0], c=scen_colors[s])
    axs[idx].set_title('Low Wind & Solar ('+zones_szon[c]+')')
    axs[idx].set_ylabel('Severity')
    axs[idx].set_xlabel('Duration [d]')
    axs[idx].set_xlim(0,dur_max)
    axs[idx].set_ylim(sev_min,sev_max)
    axs[idx].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 1, 0
    x=rl3_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=rl3_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    axs[idx].scatter(x=x, y=y, c=hist3_color, label='HIST (3.1)', alpha=0.2)
    axs[idx].plot(lin_reg(x,y)[0], c=hist3_color)
    for s in range(len(scenarios)):
        x=rl4_events.loc[(scenarios[s],zones_szon[c]),('Duration')]
        y=rl4_events.loc[(scenarios[s],zones_szon[c]),('Severity (adapted)')]
        axs[idx].scatter(x=x, y=y, c=scen_colors[s], label=scenarios[s]+' (PECD 4.1)', alpha=0.2)
        axs[idx].plot(lin_reg(x,y)[0], c=scen_colors[s])
    axs[idx].set_title('Residual Load Drought ('+zones_szon[c]+')')
    axs[idx].set_ylabel('Severity')
    axs[idx].set_xlabel('Duration [d]')
    axs[idx].set_xlim(0,dur_max)
    axs[idx].set_ylim(sev_min,sev_max)
    axs[idx].legend(facecolor="white", loc='upper left', framealpha=1)
    
    idx, idy = 2, 0
    x=dd3_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=dd3_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    axs[idx].scatter(x=x, y=y, c=hist3_color, label='HIST (3.1)', alpha=0.2)
    axs[idx].plot(lin_reg(x,y)[0], c=hist3_color)
    for s in range(len(scenarios)):
        x=dd4_events.loc[(scenarios[s],zones_szon[c]),('Duration')]
        y=dd4_events.loc[(scenarios[s],zones_szon[c]),('Severity (adapted)')]
        axs[idx].scatter(x=x, y=y, c=scen_colors[s], label=scenarios[s]+' (PECD 4.1)', alpha=0.2)
        axs[idx].plot(lin_reg(x,y)[0], c=scen_colors[s])
    axs[idx].set_title('Demand Drought ('+zones_szon[c]+')')
    axs[idx].set_ylabel('Severity')
    axs[idx].set_xlabel('Duration [d]')
    axs[idx].set_xlim(0,dur_max)
    axs[idx].set_ylim(sev_min,sev_max)
    axs[idx].legend(facecolor="white", loc='upper left', framealpha=1)
    
    plt.tight_layout()
    plt.savefig(path_to_plot+'Correlation/DF_CorrelationDurationSeverity_'+zones_szon[c]+'.'+plot_format,dpi=300)
    plt.close()

    



#%% Plot Timelines of DF Detection and ENS (only for AO ENS data so far!)
# only HIST (obviously)
# whithout February 29th
# Grid plot: X=DoY, Y=Year, Colors: White=TN, Blue=FP, Red=FN, Green=TP
# For each droughttype individual panels

# Currently does not work because some dataset are missing days
# -> "ValueError: cannot reshape array of size 12740 into shape (35,365)"
# TODO: Correct issue

def dropLeapDays(df):
    return df[~((df.index.get_level_values('Date').day == 29) & (df.index.get_level_values('Date').month == 2))]

# Create colormap
colors = ['whitesmoke', 'royalblue', 'red', 'limegreen']
cmap=ListedColormap(colors)
legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0],label='No DF, No ENS'),
                   Patch(facecolor=colors[1], edgecolor=colors[1],label='DF detected, but no ENS'),
                   Patch(facecolor=colors[2], edgecolor=colors[2],label='No DF, but ENS'),
                   Patch(facecolor=colors[3], edgecolor=colors[3],label='DF detected and ENS')]
# Set X and Y
if ens_dataset=='AO':
    X1= np.arange(364)
    X2= np.arange(364*24)
else:
    X1= np.arange(365)
    X2= np.arange(365*24)
Y = np.arange(1982,2017) # common period of ENS & PECD
# Generate masked data
ens_mask_d = mask_data(data3_ENS_d, 0, False, 2, 0)
ens_mask_h = mask_data(data3_ENS_h, 0, False, 2, 0)

lws4_mask = mask_data(data4_REP_d.loc[('HIST')], lws4_thresh, True,  1, 0)
lws3_mask = mask_data(data3_REP_d.loc[('HIST')], lws3_thresh, True,  1, 0)
rl3_mask  = mask_data(data3_RL_d.loc[('HIST')],  rl3_thresh,  False, 1, 0)
rl4_mask  = mask_data(data4_RL_d.loc[('HIST')],  rl4_thresh,  False, 1, 0)
dd3_mask  = mask_data(data3_dem_d.loc[('HIST')], dd3_thresh,  False, 1, 0)
dd4_mask  = mask_data(data4_dem_d.loc[('HIST')], dd4_thresh,  False, 1, 0)
cf3_mask  = mask_df_by_entries(data3_CF_h, cfd3_hours, ['HIST'],  1, 0).loc['HIST']
cf4_mask  = mask_df_by_entries(data4_CF_h, cfd4_hours, scenarios, 1, 0).loc['HIST']

def get_detection_mask(ens,df):
    same_index = df.index.intersection(ens.index)
    detection_mask = ens.loc[same_index] + df.loc[same_index]
    return detection_mask

lws4_detmask = get_detection_mask(dropLeapDays(ens_mask_d),dropLeapDays(lws4_mask))
lws3_detmask = get_detection_mask(dropLeapDays(ens_mask_d),dropLeapDays(lws3_mask))
rl4_detmask  = get_detection_mask(dropLeapDays(ens_mask_d),dropLeapDays(rl4_mask))
rl3_detmask  = get_detection_mask(dropLeapDays(ens_mask_d),dropLeapDays(rl3_mask))
dd4_detmask  = get_detection_mask(dropLeapDays(ens_mask_d),dropLeapDays(dd4_mask))
dd3_detmask  = get_detection_mask(dropLeapDays(ens_mask_d),dropLeapDays(dd3_mask))
cfd4_detmask = get_detection_mask(dropLeapDays(ens_mask_h),dropLeapDays(cf4_mask))
cfd3_detmask = get_detection_mask(dropLeapDays(ens_mask_h),dropLeapDays(cf3_mask))

# Tranform detection masks into the right format
def detmask2matrix(detmask,X,Y):
    return np.reshape(detmask, (len(Y),len(X)))

for c in range(len(zones_szon)):
    fig, axs = plt.subplots(2, 2, figsize=(18,8))
        
    idx, idy = 0, 0
    axs[idx, idy].pcolormesh(X1, Y, detmask2matrix(lws3_detmask[zones_szon[c]],X1,Y), cmap=cmap)
    axs[idx, idy].set_title('LWS (PECD 3.1,'+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Year')
    axs[idx, idy].set_xlabel('Day of Year')
    axs[idx, idy].legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)
    
    idx, idy = 1, 0
    axs[idx, idy].pcolormesh(X1, Y, detmask2matrix(rl3_detmask[zones_szon[c]],X1,Y), cmap=cmap)
    axs[idx, idy].set_title('RL (PECD 3.1,'+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Year')
    axs[idx, idy].set_xlabel('Day of Year')
    axs[idx, idy].legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)
    
    idx, idy = 0, 1
    axs[idx, idy].pcolormesh(X1, Y, detmask2matrix(dd3_detmask[zones_szon[c]],X1,Y), cmap=cmap)
    axs[idx, idy].set_title('DD (PECD 3.1,'+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Year')
    axs[idx, idy].set_xlabel('Day of Year')
    axs[idx, idy].legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)
    
    idx, idy = 1, 1
    axs[idx, idy].pcolormesh(X2, Y, detmask2matrix(cfd3_detmask[zones_szon[c]],X2,Y), cmap=cmap)
    axs[idx, idy].set_title('CFD (PECD 3.1,'+ens_dataset+', '+zones_szon[c]+')')
    axs[idx, idy].set_ylabel('Year')
    axs[idx, idy].set_xlabel('Day of Year')
    axs[idx, idy].legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)
    
    
    
    plt.tight_layout()
    plt.savefig(path_to_plot+'Timeline/DF_ENSDetectionMatrix_'+ens_dataset+'_'+zones_szon[c]+'.'+plot_format,dpi=300)
    plt.close()
        
        # TODO: Some masks are shorter? Why? In year with leap day the 31.12. is missing (and the leap day...)
      
#%% Plot Timeline of DFs just for Projections

# Create colormap
colors = ['whitesmoke', 'black']
cmap=ListedColormap(colors)
legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0],label='No Drought'),
                   Patch(facecolor=colors[1], edgecolor=colors[1],label='Energy Drought')]

# Tranform detection masks into the right format
def detmask2matrix(detmask,X,Y):
    return np.reshape(detmask, (len(Y),len(X)))

X1 = np.arange(365)
X2 = np.arange(365*24)
Y = np.arange(2015,2066)

for s in range(1,len(scenarios)):

    lws4_mask = mask_data(dropLeapDays(data4_REP_d.loc[(scenarios[s])]), lws4_thresh, True,  1, 0)
    rl4_mask  = mask_data(dropLeapDays(data4_RL_d.loc[((scenarios[s]))]),  rl4_thresh,  False, 1, 0)
    dd4_mask  = mask_data(dropLeapDays(data4_dem_d.loc[((scenarios[s]))]), dd4_thresh,  False, 1, 0)
    cfd4_mask = mask_df_by_entries(dropLeapDays(data4_CF_h), cfd4_hours, scenarios, 1, 0).loc[(scenarios[s])]
    
    for c in range(len(zones_szon)):
        fig, axs = plt.subplots(2, 2, figsize=(18,8))
            
        idx, idy = 0, 0
        axs[idx, idy].pcolormesh(X1, Y, detmask2matrix(lws4_mask[zones_szon[c]],X1,Y), cmap=cmap)
        axs[idx, idy].set_title('LWS (PECD 4.1,'+zones_szon[c]+')')
        axs[idx, idy].set_ylabel('Year')
        axs[idx, idy].set_xlabel('Day of Year')
        axs[idx, idy].legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)
        
        idx, idy = 1, 0
        axs[idx, idy].pcolormesh(X1, Y, detmask2matrix(rl4_mask[zones_szon[c]],X1,Y), cmap=cmap)
        axs[idx, idy].set_title('RL (PECD 4.1,'+zones_szon[c]+')')
        axs[idx, idy].set_ylabel('Year')
        axs[idx, idy].set_xlabel('Day of Year')
        axs[idx, idy].legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)
        
        idx, idy = 0, 1
        axs[idx, idy].pcolormesh(X1, Y, detmask2matrix(dd4_mask[zones_szon[c]],X1,Y), cmap=cmap)
        axs[idx, idy].set_title('DD (PECD 4.1,'+zones_szon[c]+')')
        axs[idx, idy].set_ylabel('Year')
        axs[idx, idy].set_xlabel('Day of Year')
        axs[idx, idy].legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)
        
        idx, idy = 1, 1
        axs[idx, idy].pcolormesh(X2, Y, detmask2matrix(cfd4_mask[zones_szon[c]],X2,Y), cmap=cmap)
        axs[idx, idy].set_title('CFD (PECD 4.1,'+zones_szon[c]+')')
        axs[idx, idy].set_ylabel('Year')
        axs[idx, idy].set_xlabel('Day of Year')
        axs[idx, idy].legend(handles=legend_elements, facecolor="white", loc='upper center', framealpha=1)
     
        
        plt.tight_layout()
        plt.savefig(path_to_plot+'Timeline/DF_PROJDetectionTimelines_'+scenarios[s].replace('/','_')+'_'+zones_szon[c]+'.'+plot_format,dpi=300)
        plt.close()
      
#%% Plot Correlation of ENS/Severity and ENS/Duration

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
    x=lws4_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    y=get_ENS_sums(data3_ENS_d, lws4_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[0], label='LWS (PECD4.1)', alpha=0.5, marker='x')
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[0], label='Lin Reg (PECD4.1)',  alpha=0.3)
    test[c,0,0,0]=lin_reg(x, y)[3]
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
    x=rl4_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    y=get_ENS_sums(data3_ENS_d, rl4_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[1], label='RL (PECD4.1)', alpha=0.5, marker='x')
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[1], label='Lin Reg (PECD4.1)',  alpha=0.3)
    test[c,0,1,0]=lin_reg(x, y)[3]
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
    x=dd4_events.loc[('HIST',zones_szon[c]),('Severity (adapted)')]
    y=get_ENS_sums(data3_ENS_d, dd4_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[2], label='DD (PECD4.1)', alpha=0.5, marker='x')
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[2], label='Lin Reg (PECD4.1)',  alpha=0.3)
    #print(zones_szon[c]+' DD  (4.1) Sev R value: '+str(lin_reg(x, y)[3]))
    test[c,0,2,0]=lin_reg(x, y)[3]
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
    plt.savefig(path_to_plot+'Correlation/DF_CorrelationENSSeverity_'+ens_dataset+'_'+zones_szon[c]+'.'+plot_format,dpi=300)
    plt.close()
    
    
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    
    idx, idy = 0, 0
    x=lws4_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=get_ENS_sums(data3_ENS_d, lws4_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[0], label='LWS (PECD4.1)', alpha=0.5, marker='x')
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[0], label='Lin Reg (PECD4.1)',  alpha=0.3)
    #print(zones_szon[c]+' LWS (4.1) Dur R value: '+str(lin_reg(x, y)[3]))
    test[c,0,0,1]=lin_reg(x, y)[3]
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
    x=rl4_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=get_ENS_sums(data3_ENS_d, rl4_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[1], label='RL (PECD4.1)', alpha=0.5, marker='x')
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[1], label='Lin Reg (PECD4.1)',  alpha=0.3)
    #print(zones_szon[c]+' RL  (4.1) Dur R value: '+str(lin_reg(x, y)[3]))
    test[c,0,1,1]=lin_reg(x, y)[3]
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
    x=dd4_events.loc[('HIST',zones_szon[c]),('Duration')]
    y=get_ENS_sums(data3_ENS_d, dd4_events.loc[('HIST')],zones_szon[c])
    axs[idx, idy].scatter(x, y, color=dt_colors[2], label='DD (PECD4.1)', alpha=0.5, marker='x')
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[2], label='Lin Reg (PECD4.1)',  alpha=0.3)
    #print(zones_szon[c]+' DD  (4.1) Dur R value: '+str(lin_reg(x, y)[3]))
    test[c,0,2,1]=lin_reg(x, y)[3]
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
    x=cfd4_events.loc[('HIST',zones_szon[c]),('Duration')]/24
    y=get_ENS_sums(data3_ENS_h, cfd4_events.loc[('HIST')],zones_szon[c], 24)
    axs[idx, idy].scatter(x, y, color=dt_colors[3], label='CFD (PECD4.1)', alpha=0.5, marker='x')
    axs[idx, idy].plot(lin_reg(x, y)[0], c=dt_colors[3], label='Lin Reg (PECD4.1)',  alpha=0.3)
    #print(zones_szon[c]+' CFD (4.1) Dur R value: '+str(lin_reg(x, y)[3]))
    test[c,0,3,1]=lin_reg(x, y)[3]
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
    plt.savefig(path_to_plot+'Correlation/DF_CorrelationENSDuration_'+ens_dataset+'_'+zones_szon[c]+'.'+plot_format,dpi=300)
    plt.close()
    
#%% TEMPORÄR
df3_stats = lws3_annual_stats
df4_stats = lws4_annual_stats

test_mean = pd.DataFrame()
test_mean['HIST (3.1)'] = df3_stats.loc['HIST'].mean()
test_mean['HIST (4.1)'] = df4_stats.loc['HIST'].mean()
test_mean['SP245/CMR5'] = df4_stats.loc['SP245/CMR5'].mean()
test_mean['SP245/ECE3'] = df4_stats.loc['SP245/ECE3'].mean()
test_mean['SP245/MEHR'] = df4_stats.loc['SP245/MEHR'].mean()

test_max = pd.DataFrame()
test_max['HIST (3.1)'] = df3_stats.loc['HIST'].max()
test_max['HIST (4.1)'] = df4_stats.loc['HIST'].max()
test_max['SP245/CMR5'] = df4_stats.loc['SP245/CMR5'].max()
test_max['SP245/ECE3'] = df4_stats.loc['SP245/ECE3'].max()
test_max['SP245/MEHR'] = df4_stats.loc['SP245/MEHR'].max()
