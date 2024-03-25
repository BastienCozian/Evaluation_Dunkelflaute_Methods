# -*- coding: utf-8 -*-
"""
This script detects energy droughts using method 2 (inspired by Otero et al. 2022)

Detected drought days and drought events (with corresponding properties) are determined and saves as pickles.
Also a mask for all days (masking drought days) is saved.

For questions, refer to benjamin.biewald@tennet.eu
"""

#%% Import modules

import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from Dunkelflaute_function_library import get_zones
from Dunkelflaute_function_library import get_thresholds
from Dunkelflaute_function_library import detect_drought_Otero22
from Dunkelflaute_function_library import mask_data
from Dunkelflaute_function_library import get_f_score
from matplotlib.ticker import MaxNLocator


#%% Specify parameters

path_to_data      = 'D:/Dunkelflaute/Data/'
path_to_plot      = 'D:/Dunkelflaute/'
path_to_shapefile = 'D:/PECD4_1/ShapeFiles/General/'
plot_format       = 'png'

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

# Study zones (SZON) for evaluation (so far data for only 'DE00' & 'NL00' is available)
eval_szon = ['DE00', 'NL00']
# TODO: Leave this but make it so that it loads automatially the right AO data
ao_scen = 'W.v8' # 'W.v8' or 'S.v6' for either wind or solar dominated scenario

# Models
models = ['CMR5','ECE3','MEHR']
model_colors = ['blue', 'orange', 'red']
hist3_color = 'purple'

# Pathways
pathways = ['SP245']

# Technologies / Variables of Interest and corresponding header sizes
techs = ['SPV', 'WOF', 'WON']#,'TAW'] 
aggs = ['PEON','PEOF','PEON']#,'SZON'] 
tech_agg = ['SPV/PEON', 'WOF/PEOF', 'WON/PEON']#, 'TAW/SZON']
tech_headers = [52, 0, 0]#, 52]
tech_ens = ['SPV_','_20','_30']#,'_TAW_'] 


scenarios = []
scen_colors = []
scenarios.append('HIST')
scen_colors.append('forestgreen')                                                       
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
  

#%% Functions

def get_df_timerange(df, start_date_a, end_date_a):
    return df.query('Date>=@start_date_a and Date <= @end_date_a') 

def find_idx_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin(axis=0)
    return idx

#%% Load the preprocessed data (DataFrames as pickles)

data4_REP_d = pd.read_pickle(path_to_data+'PECD4_Generation_TY'+str(ty_pecd4)+'_national_daily.pkl')
data3_REP_d = pd.read_pickle(path_to_data+'PECD3_Generation_TY'+str(ty_pecd3)+'_national_daily.pkl')
data4_RL_d  = pd.read_pickle(path_to_data+'PECD4_ETM_RL_TY'+str(ty_pecd4)+'_national_daily.pkl')
data3_RL_d  = pd.read_pickle(path_to_data+'PECD3_PEMMDB_RL_TY'+str(ty_pecd3)+'_national_daily.pkl')

data4_dem_d = pd.read_pickle(path_to_data+'ETM_demand_TY'+str(ty_pecd4)+'_daily.pkl')
data3_dem_d = pd.read_pickle(path_to_data+'PEMMDB_demand_TY'+str(ty_pecd3)+'_daily.pkl')

#data3_ENS_d = pd.read_pickle(path_to_data+'AO_W.v8_ENS_daily.pkl')
#data_cap    = pd.read_pickle(path_to_data+'PEMMDB_capacities_TY2033.pkl')

#%% Detect Events  

lws4_thresh, lws4_sigma = get_thresholds(data4_REP_d.loc[('HIST')], LWS_percentile, start_date='1980-01-01', end_date='2019-12-31', empirical=True)
lws3_thresh, lws3_sigma = get_thresholds(data3_REP_d.loc[('HIST')], LWS_percentile, start_date='1980-01-01', end_date='2019-12-31', empirical=True)
rl4_thresh,  rl4_sigma  = get_thresholds(data4_RL_d.loc[('HIST')],  RL_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)
rl3_thresh,  rl3_sigma  = get_thresholds(data3_RL_d.loc[('HIST')],  RL_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)
dd4_thresh,  dd4_sigma  = get_thresholds(data4_dem_d.loc[('HIST')], DD_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)
dd3_thresh,  dd3_sigma  = get_thresholds(data3_dem_d.loc[('HIST')], DD_percentile,  start_date='1980-01-01', end_date='2019-12-31', empirical=True)
    
# Detect Drought days and events
lws4_days, lws4_events = detect_drought_Otero22(scenarios, zones_szon, data4_REP_d, lws4_thresh, lws4_sigma)
lws3_days, lws3_events = detect_drought_Otero22(['HIST'],  zones_szon, data3_REP_d, lws3_thresh, lws3_sigma)
rl4_days,  rl4_events  = detect_drought_Otero22(scenarios, zones_szon, data4_RL_d,  rl4_thresh,  rl4_sigma, below=False)
rl3_days,  rl3_events  = detect_drought_Otero22(['HIST'],  zones_szon, data3_RL_d,  rl3_thresh,  rl3_sigma, below=False)
dd4_days,  dd4_events  = detect_drought_Otero22(scenarios, zones_szon, data4_dem_d, dd4_thresh,  dd4_sigma, below=False)
dd3_days,  dd3_events  = detect_drought_Otero22(['HIST'],  zones_szon, data3_dem_d, dd3_thresh,  dd3_sigma, below=False)

# Mask the data / Detect Drought days
#ens_mask  = mask_data(data3_ENS_d, 0, False, 2, 0)
lws4_mask = mask_data(data4_REP_d, lws4_thresh, True,  1, 0)
lws3_mask = mask_data(data3_REP_d, lws3_thresh, True,  1, 0)
rl4_mask  = mask_data(data4_RL_d,  rl4_thresh,  False, 1, 0)
rl3_mask  = mask_data(data3_RL_d,  rl3_thresh,  False, 1, 0)
dd4_mask  = mask_data(data4_dem_d, dd4_thresh,  False, 1, 0)
dd3_mask  = mask_data(data3_dem_d, dd3_thresh,  False, 1, 0)

# Calculate F
#lws4_stat = get_f_score(ens_mask, lws4_mask, beta=1)
#lws3_stat = get_f_score(ens_mask, lws3_mask, beta=1)
#rl3_stat  = get_f_score(ens_mask, rl3_mask,  beta=1)

#%% Save detection outputs
lws4_events.to_pickle(path_to_plot+'Data/LWS4_TY'+str(ty_pecd4)+'_Thresh_'+str(LWS_percentile)+'_events.pkl')
lws3_events.to_pickle(path_to_plot+'Data/LWS3_TY'+str(ty_pecd3)+'_Thresh_'+str(LWS_percentile)+'_events.pkl')
rl4_events.to_pickle(path_to_plot+'Data/RL4_TY'+str(ty_pecd4)+'_Thresh_'+str(RL_percentile)+'_events.pkl')
rl3_events.to_pickle(path_to_plot+'Data/RL3_TY'+str(ty_pecd3)+'_Thresh_'+str(RL_percentile)+'_events.pkl')
dd4_events.to_pickle(path_to_plot+'Data/DD4_TY'+str(ty_pecd4)+'_Thresh_'+str(DD_percentile)+'_events.pkl')
dd3_events.to_pickle(path_to_plot+'Data/DD3_TY'+str(ty_pecd3)+'_Thresh_'+str(DD_percentile)+'_events.pkl')

lws4_mask.to_pickle(path_to_plot+'Data/LWS4_TY'+str(ty_pecd4)+'_Thresh_'+str(LWS_percentile)+'_mask.pkl')
lws3_mask.to_pickle(path_to_plot+'Data/LWS3_TY'+str(ty_pecd3)+'_Thresh_'+str(LWS_percentile)+'_mask.pkl')
rl4_mask.to_pickle(path_to_plot+'Data/RL4_TY'+str(ty_pecd4)+'_Thresh_'+str(RL_percentile)+'_mask.pkl')
rl3_mask.to_pickle(path_to_plot+'Data/RL3_TY'+str(ty_pecd3)+'_Thresh_'+str(RL_percentile)+'_mask.pkl')
dd4_mask.to_pickle(path_to_plot+'Data/DD4_TY'+str(ty_pecd4)+'_Thresh_'+str(DD_percentile)+'_mask.pkl')
dd3_mask.to_pickle(path_to_plot+'Data/DD3_TY'+str(ty_pecd3)+'_Thresh_'+str(DD_percentile)+'_mask.pkl')


