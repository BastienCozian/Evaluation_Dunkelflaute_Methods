# -*- coding: utf-8 -*-
"""
This script detects energy droughts using method 1 (inspired by Li et al. 2021)

Detected drought hours and drought events (with corresponding properties) are determined and saves as pickles.
Also a mask for all hours (masking drought hours) is saved.

For questions, refer to benjamin.biewald@tennet.eu
"""

#%% Import packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from Dunkelflaute_function_library import get_zones
from Dunkelflaute_function_library import get_thresholds
from Dunkelflaute_function_library import detect_drought_Li21
from Dunkelflaute_function_library import mask_df_by_entries
from Dunkelflaute_function_library import get_f_score
from matplotlib.ticker import MaxNLocator

#%% Specify Parameters

path_to_data      = 'D:/Dunkelflaute/Data/'
path_to_plot      = 'D:/Dunkelflaute/'
plot_format       = 'png'

# Capacity Factor Threshold for all technologies (see Li et al. 2021)
cf_threshold = 0.03  # original: 0.2

# Countries (NUT0) of interest (must be a list of two letter abbreviations)
countries = ['DE','NL']

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

#%% Functions

def get_df_timerange(df, start_date_a, end_date_a):
    return df.query('Date>=@start_date_a and Date <= @end_date_a') 

def find_idx_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin(axis=0)
    return idx


#%% Load the preprocessed data (DataFrames as pickles)

data4_CF_h = pd.read_pickle(path_to_data+'PECD4_CF_TY'+str(ty_pecd4)+'_national_hourly.pkl')
data3_CF_h = pd.read_pickle(path_to_data+'PECD3_CF_TY'+str(ty_pecd3)+'_national_hourly.pkl')


#%% Detect Dunkelflauten

df3_hours, df3_events = detect_drought_Li21(['HIST'],  zones_szon, data3_CF_h, cf_threshold)
df4_hours, df4_events = detect_drought_Li21(scenarios, zones_szon, data4_CF_h, cf_threshold)

df3_events_gret24 = df3_events[df3_events['Duration']>24]
df4_events_gret24 = df4_events[df4_events['Duration']>24]

df4_mask = mask_df_by_entries(data4_CF_h, df4_hours, scenarios, 1, 0)
df3_mask = mask_df_by_entries(data3_CF_h, df3_hours, ['HIST'],  1, 0)


#%% Save the detection outputs

df3_hours.to_pickle(path_to_plot+'Data/CFD3_TY'+str(ty_pecd3)+'_Thresh_'+str(cf_threshold)+'_hours.pkl')
df4_hours.to_pickle(path_to_plot+'Data/CFD4_TY'+str(ty_pecd4)+'_Thresh_'+str(cf_threshold)+'_hours.pkl')

df3_events_gret24.to_pickle(path_to_plot+'Data/CFD3_TY'+str(ty_pecd3)+'_Thresh_'+str(cf_threshold)+'_events.pkl')
df4_events_gret24.to_pickle(path_to_plot+'Data/CFD4_TY'+str(ty_pecd4)+'_Thresh_'+str(cf_threshold)+'_events.pkl')

df3_mask.to_pickle(path_to_plot+'Data/CFD3_TY'+str(ty_pecd3)+'_Thresh_'+str(cf_threshold)+'_mask.pkl')
df4_mask.to_pickle(path_to_plot+'Data/CFD4_TY'+str(ty_pecd4)+'_Thresh_'+str(cf_threshold)+'_mask.pkl')
