# -*- coding: utf-8 -*-
"""
This script loads several different data sources needed for the Dunkelflaute analysis 
and unifies them in dataframe formats for easier loading in further scripts.

Dataframes then are saved as pickles.

Note: Do not run all of the cells after another, but only what you are interested to.
Some data e.g. requires ty=2033, whereas some ty=2050.

For questions, refer to benjamin.biewald@tennet.eu
"""

#%% Import packages

import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Dunkelflaute_function_library import get_zones
from Dunkelflaute_function_library import get_daily_values
from Dunkelflaute_function_library import get_daily_values_pecd
from Dunkelflaute_function_library import get_daily_values_etm

#%% Specify parameters

path_to_pecd4     = 'F:/C3S_PECD_v4.1/'                         #'D:/PECD4_1/'
path_to_pecd3     = 'F:/PECD3_1/'            #'D:/PECD3_1/'
path_to_etm_d     = 'F:/Data_Dunkelflaute_analysis/PECD4_ETM_demand/'           #'D:/PECD4_1/ETM_Demand/exports/demand/'
path_to_eraa23_d  = '' # This data ended up not being used                  #'D:/ERAA23/Demand Dataset/' 
path_to_eraa23_ens= 'F:/Data_Dunkelflaute_analysis/ERAA23_ENS/'                 #'D:/ERAA23/ENS/'
path_to_ao_ens    = 'F:/Data_Dunkelflaute_analysis/Adequacy_Outlook_ENS/'      #'D:/AdequacyOutlook/'
path_to_pemmdb_c  = 'F:/Data_Dunkelflaute_analysis/PEMMDB_Installed_Capacities/' #'D:/PEMMDB/data_TY2033/01_PEMMDB/'
path_to_pemmdb_d  = 'F:/Data_Dunkelflaute_analysis/PEMMDB_TY2033_demand/'    #'D:/PEMMDB/data_TY2033/04_LOAD/TY2033/'
path_to_plot      = 'F:/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'      #'D:/Dunkelflaute/'
#path_to_shapefile = 'D:/PECD4_1/ShapeFiles/General/' # Not used

# Countries (NUT0) of interest (must be a list of two letter abbreviations)
countries = ['DE','NL'] # right now only data for DE and NL is certainly available

# Study zones (SZON) for evaluation (so far data for only 'DE00' & 'NL00' is available)
eval_szon = ['DE00', 'NL00']
ao_scen = 'W.v8' # 'W.v8' or 'S.v6' for either wind or solar dominated scenario (only for AO ENS data relevant)
etm_scenario = 'DE' # either 'DE' or 'GA' (Distributed Energy / Global Ambitions)

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
tech_headers = [52, 52, 52]#, 52]
tech_ens = ['SPV_','_20','_30']#,'_TAW_'] 

# Target years 
# PECD3.1: 2033 (for PEMMDB demand data only 2033 is available!)
# PECD4.1: 2030, 2040 or 2050 (demand data for only these years is available)
# PEMMDB installed capacity data is availale for 2022, 2025, 2027, 2030, 2033, 2040 & 2050
ty = 2050 # 2030

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
        
zones_peon = get_zones(countries,'PEON')  
zones_szon = get_zones(countries,'SZON')  
zones_peof = get_zones(countries,'PEOF')  

scen_timespan = [42,51,51,51] # TODO: Automate the nr of years per scenario
  

#%% Load the PEMMDB capacity data of target year 
# Run this before loading any of the PECD data

if np.isin(ty, [2022, 2025, 2027, 2030, 2033, 2040, 2050])==False:
    raise KeyError('Targetyear (ty) must be either 2022, 2025, 2027, 2030, 2033, 2040 or 2050 and not '+str(ty))

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
    
    # Extract only necessary rows (zones) and sepearete into technologies
    data_won = data_raw.loc[(data_raw['Technology']=='Wind Onshore')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty]].set_index('PECD Zone')
    data_wof = data_raw.loc[(data_raw['Technology']=='Wind Offshore')&(data_raw['PECD Zone'].isin(peof_s))][['PECD Zone',ty]].set_index('PECD Zone')
    data_spv_rt = data_raw.loc[(data_raw['Technology']=='Solar PV Rooftop')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty]].set_index('PECD Zone')
    data_spv_f  = data_raw.loc[(data_raw['Technology']=='Solar PV Farm')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty]].set_index('PECD Zone')
    
    # Replace "slim" PEOF zones with extended names to avoid confusion with PEON zones
    index_mapping = dict(zip(peof_s, peof))
    data_wof.rename(index=index_mapping, inplace=True)

    # Combine SPVs
    data_spv = data_spv_rt + data_spv_f
    
    # Rename the 'Total' columns into the corresponding study zone
    data_spv.rename(index={"Total": zones_szon[c]}, inplace=True)
    data_wof.rename(index={"Total": zones_szon[c]}, inplace=True)
    data_won.rename(index={"Total": zones_szon[c]}, inplace=True)
    
    # Reshape, so that zones = columns and technologies = index
    tec_list = pd.concat([data_spv.T, data_wof.T, data_won.T], keys=techs, names=['Technology'])
    tec_list = tec_list.droplevel(level=1)
    
    cap_list.append(tec_list)
   
data_cap = pd.concat(cap_list, axis=1)
data_cap.to_pickle(path_to_plot+'Data/PEMMDB_capacities_TY'+str(ty)+'.pkl')

print('Preprocessed PEMMDB capacity data')


#%% Load the PEMMDB demand data (1982-2016), Target Year: 2033

# Sheets sometimes have different formats
# Two formats:     short = Date, Hour, 1982, 1983, ..., 2016    AND     long = Date, Month, Day, Hour, 1981, 1983, ..., 2019

file = 'DEMAND_Timeseries_2033.xlsx'

# Prepare the format of the dataframe
time_index = pd.Series(np.zeros(35*365*24)) # 35 years * 365 days * 24 hours
demand     = np.zeros(35*365*24) # 35 years * 365 days * 24 hours
data3_demand = pd.DataFrame()
years = np.arange(1982,2017)

# Load each zone (=sheet)
for z in range(len(zones_szon)):
    data_raw = pd.read_excel(path_to_pemmdb_d+file,sheet_name=zones_szon[z],header=10)
    
    # For each year (=column) calculate datetime index and get the right demand data
    for y in range(len(years)):
        h = data_raw['Hour']-1
        time_index[y*365*24:(y+1)*365*24] =  pd.to_datetime(data_raw['Date'].astype(str)+str(years[y]) +' '+h.astype(str)+':00:00', format="%d.%m.%Y %H:%M:%S")
        demand[y*365*24:(y+1)*365*24] = data_raw[years[y]]
    
    data3_demand[zones_szon[z]] = demand
data3_demand.index = time_index
data3_demand.index = pd.to_datetime(data3_demand.index)
data3_demand.index.rename('Date', inplace=True)
data3_demand = pd.concat([data3_demand],keys=['HIST'],names=['Scenario'])

# Save it as pickels
data3_demand.to_pickle(path_to_plot+'Data/PEMMDB_demand_TY2033_hourly.pkl')

# Sum up annual / daily values
data3_demand_sum_d = get_daily_values_etm(data3_demand,'sum')
data3_demand_sum_y = data3_demand.groupby(['Scenario',data3_demand.index.get_level_values('Date').year]).sum()
data3_demand_sum_d.to_pickle(path_to_plot+'Data/PEMMDB_demand_TY2033_daily.pkl')
data3_demand_sum_y.to_pickle(path_to_plot+'Data/PEMMDB_demand_TY2033_annual.pkl')

print('Preprocessed PEMMDB demand data')

#%% Load the ETM demand data (based on PECD4.1)

if np.isin(ty, [2030,2040,2050])==False:
    raise KeyError('Targetyear (ty) must be either 2030, 2040 or 2050 and not '+str(ty))

list_sce = []
for s in range(len(scenarios)):
    if scenarios[s]=='HIST':
        sty = 1980
        eny = 2021
        timeframe=str(sty)+'-'+str(eny)
        mod = 'HIST'
    else:
        sty = 2015
        eny = 2065
        timeframe=str(sty)+'-'+str(eny)
        mod = scenarios[s][:-6] # cut off the 'SP245' as it is not part of the filename
    
    #index_array = pd.to_datetime(np.arange(np.datetime64(str(sty)+'-01-01'), np.datetime64(str(eny)+'-12-31'), np.timedelta64(1, 'h')))
    index_array = np.empty((eny-sty+1)*(24*365), dtype='datetime64[s]')
    data_array = np.zeros(len(index_array))
    tmp_df = pd.DataFrame(index=index_array)
    tmp_df.index.rename('Date', inplace=True)
    for c in range(len(countries)):
        file = etm_scenario+'_Dunkelflaute_'+countries[c]+'_'+str(ty)+'_electricity_'+mod+'_'+timeframe+'_demand.xlsx'
        data_tmp = pd.read_excel(path_to_etm_d+file)
        for y in range(eny-sty+1):
            # Generate DateTime index of the year
            idx_tmp = pd.date_range(start='1/1/'+str(sty+y)+' 00:00:00', end='31/12/'+str(sty+y)+' 23:00:00', freq='1h')
            if len(idx_tmp)>8760:
                # Drop the 29th February (if existent)
                idx_tmp = idx_tmp.drop(pd.date_range(start='2/29/'+str(sty+y)+' 00:00:00', end='2/29/'+str(sty+y)+' 23:00:00', freq='1h'))
            index_array[y*(24*365):(y+1)*(24*365)] = idx_tmp
            data_array[y*(24*365):(y+1)*(24*365)]  = data_tmp[sty+y]
        tmp_df[zones_szon[c]] = data_array
    list_sce.append(tmp_df)
data4_demand_h = pd.concat(list_sce,keys=scenarios,names=['Scenario'])
data4_demand_h = -data4_demand_h

data4_demand_y = data4_demand_h.groupby(['Scenario',data4_demand_h.index.get_level_values('Date').year]).sum()
data4_demand_d = get_daily_values_etm(data4_demand_h,'sum')

data4_demand_y.to_pickle(path_to_plot+'Data/ETM_Demand_TY'+str(ty)+'_annual.pkl')
data4_demand_d.to_pickle(path_to_plot+'Data/ETM_Demand_TY'+str(ty)+'_daily.pkl')
data4_demand_h.to_pickle(path_to_plot+'Data/ETM_Demand_TY'+str(ty)+'_hourly.pkl')

print('Preprocessed ETM demand data')
        
#%% Load the PECD 4.1 data 
# PEMMDB installed capacities with the right target year must be already loaded!
# If not, run the PEMMDB capacity loding cell before running this cell.
# Keep track that the same targetyear is chosen!

if np.isin(ty, [2030,2040,2050])==False:
    raise KeyError('Targetyear (ty) must be either 2030, 2040 or 2050 and not '+str(ty))

data_list_tec_cf = [] # cf = capacity factor
data_list_tec_ac = [] # ac = absolute capacity = generation

for t in range(len(tech_agg)):
    data_list_sce_cf = []
    data_list_sce_ac = []
    for s in range(len(scenarios)):
        # ENER or CLIM
        if techs[t]=='TAW' or techs[t]=='TA':
            domain = 'CLIM'
        else:
            domain = 'ENER'
            
        # HIST or PROJ
        if scenarios[s]=='HIST':
            datapath = path_to_pecd4 +'HIST/'+domain+'/'+tech_agg[t]+'/'
        else:
            datapath = path_to_pecd4 +'PROJ/'+domain+'/'+scenarios[s]+'/'+tech_agg[t]+'/'
            
        if os.path.exists(datapath):
            zones = get_zones(countries,aggs[t])  
            
            datafile = glob.glob(datapath+'*'+tech_ens[t]+'*csv')[0]    # find specified file(s)
                
            data_all     = pd.read_csv(datafile, header=tech_headers[t])       # load the data
            data_all['Date'] = pd.to_datetime(data_all['Date'])
            data_all.set_index('Date', inplace=True)
            data_zones = data_all[zones]                                     # only safe zones of interest
            data_zones = data_zones.replace(9.96921e+36,np.nan)        # replace NaN fillvalue with NaN  
             
            #if techs[t]=='WON':
            #    data_caps = cf2cap(data_zones, cap_WON, ref_year, zones)
            #elif techs[t]=='WOF':
            #    data_caps = cf2cap(data_zones, cap_WOF, ref_year, zones)
            #elif techs[t]=='SPV':
            #    data_caps = cf2cap(data_zones, cap_SPV, ref_year, zones)
            
            data_list_sce_cf.append(data_zones)
            #data_list_sce_ac.append(data_caps)
            data_list_sce_ac.append(data_zones*data_cap.loc[(techs[t]),zones])
            print('Loaded data for:  '+datafile)
            
        else:
            raise KeyError('No data in '+datapath)
    data_list_tec_cf.append(pd.concat(data_list_sce_cf,keys=scenarios,names=['Scenario']))
    data_list_tec_ac.append(pd.concat(data_list_sce_ac,keys=scenarios,names=['Scenario']))
    
data_cf = pd.concat(data_list_tec_cf,keys=techs,names=['Technology'])
data_ac = pd.concat(data_list_tec_ac,keys=techs,names=['Technology'])


# Sum up annual and daily values
# CF annual means
data_cf_mean_y = data_cf.groupby(['Technology','Scenario',data_cf.index.get_level_values('Date').year]).mean()
# AC annual sums
data_ac_sum_y = data_ac.groupby(['Technology','Scenario',data_ac.index.get_level_values('Date').year]).sum()
# CF daily means
data_cf_mean_d = get_daily_values_pecd(data_cf,'mean')
# AC daily sums
data_ac_sum_d = get_daily_values_pecd(data_ac,'sum')

# Save it as pickels
data_cf.to_pickle(path_to_plot+'Data/PECD4_CF_zonal_hourly.pkl')
data_ac.to_pickle(path_to_plot+'Data/PECD4_Generation_TY'+str(ty)+'_zonal_hourly.pkl')
data_cf_mean_y.to_pickle(path_to_plot+'Data/PECD4_CF_zonal_annual.pkl')
data_cf_mean_d.to_pickle(path_to_plot+'Data/PECD4_CF_zonal_daily.pkl')
data_ac_sum_y.to_pickle(path_to_plot+'Data/PECD4_Generation_TY'+str(ty)+'_zonal_annual.pkl')
data_ac_sum_d.to_pickle(path_to_plot+'Data/PECD4_Generation_TY'+str(ty)+'_zonal_daily.pkl')

print('Preprocessed PECDv4.1 data')

# Calculate zonal aggregations of Generation (for Otero et al 22 method)

data_ac_national_h = pd.DataFrame()
data_ac_national_d = pd.DataFrame()

for c in range(len(zones_szon)):
    # make a list of all Off- and Onshore zones of a country
    country_zones = get_zones([countries[c]],'PEON') + get_zones([countries[c]],'PEOF')
    # sum up all the zones per country
    data_ac_national_h[zones_szon[c]] = data_ac[country_zones].sum(axis=1)
    data_ac_national_d[zones_szon[c]] = data_ac_sum_d[country_zones].sum(axis=1)

# Sum up WOF, WON and SPV
data_ac_tsum_h = data_ac_national_h.groupby(['Scenario','Date']).sum()
data_ac_tsum_d = data_ac_national_d.groupby(['Scenario','Date']).sum()

data_ac_tsum_h.to_pickle( path_to_plot+'Data/PECD4_Generation_TY'+str(ty)+'_national_hourly.pkl')
data_ac_tsum_d.to_pickle( path_to_plot+'Data/PECD4_Generation_TY'+str(ty)+'_national_daily.pkl')

print('Summed up all the zones of a country for PECDv4.1')

# Calculate national CFs (for Li et al 21 method)
# 1) use CF * installed capacity = generation (for all zones & technologies)
# already pre-made (data_ac)

# 2) aggregate all peon/peof zones to szon (but keep individual technologies)
national_ac = pd.DataFrame()
for c in range(len(zones_szon)):
    country = zones_szon[c][:-2]
    peon_peof_zones = get_zones([country],'PEON') + get_zones([country],'PEOF')
    
    national_ac[zones_szon[c]]  = data_ac[peon_peof_zones].sum(axis=1)

# 2.5) Find all zones that are not completely nan and sum up the installed capacities (per SZON))
zones_not_nan=[]
total_cap = pd.DataFrame(columns=zones_szon, index=techs, data=np.zeros((len(techs),len(zones_szon))))
for z in range(len(data_cf.columns)):
    if np.all(np.isnan(data_cf[data_cf.columns[z]]))==False:
        zones_not_nan.append(data_cf.columns[z])
        szon = data_cf.columns[z][:2]+'00'
        total_cap[szon] = total_cap[szon] + data_cap[data_cf.columns[z]].fillna(0)

# 3) calculate generation / installed capacity = CF to have national CFs
tec_list = []
for t in range(len(techs)):
    tec_list.append( national_ac.loc[techs[t]] /  total_cap[zones_szon].loc[techs[t]])

data_cf_national  = pd.concat(tec_list, keys=techs,names=['Technology'])
data_cf_national.to_pickle( path_to_plot+'Data/PECD4_CF_TY'+str(ty)+'_national_hourly.pkl')

print('Calculated national capacity factors for PECDv4.1')

# Calculate Residual Load and save it 
# Drop all the 29.02. from the original data
data4_cropped = data_ac_tsum_d[~((data_ac_tsum_d.index.get_level_values(1).day == 29) & (data_ac_tsum_d.index.get_level_values(1).month == 2))]
data4_rl_tsum_d =  data4_demand_d - data4_cropped

data4_rl_tsum_d.to_pickle(path_to_plot+'Data/PECD4_ETM_RL_TY'+str(ty)+'_national_daily.pkl')
print('Calculated Residual Load for PECDv4.1')


#%% Load the PECD 3.1 data
# PEMMDB installed capacities with the right target year must be already loaded!
# If not, run the PEMMDB capacity loding cell before running this cell.
# Keep track that the same targetyear is chosen!

if not ty==2033:
    raise KeyError('Targetyear (ty) must be 2033 and not '+str(ty))

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
                data3_list_sce_ac.append(data3_zones*data_cap.loc[(techs[t]),zones])
                print('Loaded data for:  '+datafile)
                
            else:
                raise KeyError('No data in '+datapath)
        data3_list_tec_cf.append(pd.concat(data3_list_sce_cf,keys=scenarios_3,names=['Scenario']))
        data3_list_tec_ac.append(pd.concat(data3_list_sce_ac,keys=scenarios_3,names=['Scenario']))
data3_cf = pd.concat(data3_list_tec_cf,keys=techs,names=['Technology'])
data3_ac = pd.concat(data3_list_tec_ac,keys=techs,names=['Technology'])


# Sum up annual and daily values
# CF annual means
data3_cf_mean_y = data3_cf.groupby(['Technology','Scenario',data3_cf.index.get_level_values('Date').year]).mean()
# AC annual sums
data3_ac_sum_y = data3_ac.groupby(['Technology','Scenario',data3_ac.index.get_level_values('Date').year]).sum()
# CF daily means
data3_cf_mean_d = get_daily_values_pecd(data3_cf,'mean')
# AC daily sums
data3_ac_sum_d = get_daily_values_pecd(data3_ac,'sum')

# Save it as pickels
data3_cf.to_pickle(path_to_plot+'Data/PECD3_CF_zonal_hourly.pkl')
data3_ac.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty)+'_zonal_hourly.pkl')
data3_cf_mean_y.to_pickle(path_to_plot+'Data/PECD3_CF_zonal_annual.pkl')
data3_cf_mean_d.to_pickle(path_to_plot+'Data/PECD3_CF_zonal_daily.pkl')
data3_ac_sum_y.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty)+'_zonal_annual.pkl')
data3_ac_sum_d.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty)+'_zonal_daily.pkl')

print('Preprocessed PECDv3.1 data')

# Calculate zonal aggregations of Generation (for Otero et al 22 method)

data3_ac_national_h = pd.DataFrame()
data3_ac_national_d = pd.DataFrame()

for c in range(len(zones_szon)):
    # make a list of all Off- and Onshore zones of a country
    country_zones = get_zones([countries[c]],'PEON') + get_zones([countries[c]],'PEOF')
    # sum up all the zones per country
    data3_ac_national_h[zones_szon[c]] = data3_ac[country_zones].sum(axis=1)
    data3_ac_national_d[zones_szon[c]] = data3_ac_sum_d[country_zones].sum(axis=1)

# Sum up WOF, WON and SPV
data3_ac_tsum_h = data3_ac_national_h.groupby(['Scenario','Date']).sum()
data3_ac_tsum_d = data3_ac_national_d.groupby(['Scenario','Date']).sum()

data3_ac_tsum_h.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty)+'_national_hourly.pkl')
data3_ac_tsum_d.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty)+'_national_daily.pkl')

print('Summed up all the zones of a country for PECDv3.1')

# Calculate national CFs (for Li et al 21 method)
# 1) use CF * installed capacity = generation (for all zones & technologies)
# already pre-made (data_ac)

# 2) aggregate all peon/peof zones to szon (but keep individual technologies)
national_ac3 = pd.DataFrame()
for c in range(len(zones_szon)):
    country = zones_szon[c][:-2]
    peon_peof_zones = get_zones([country],'PEON') + get_zones([country],'PEOF')
    
    national_ac3[zones_szon[c]] = data3_ac[peon_peof_zones].sum(axis=1)

# 2.5) Find all zones that are not completely nan and sum up the installed capacities (per SZON))
zones_not_nan=[]
total_cap = pd.DataFrame(columns=zones_szon, index=techs, data=np.zeros((len(techs),len(zones_szon))))
for z in range(len(data3_cf.columns)):
    if np.all(np.isnan(data3_cf[data3_cf.columns[z]]))==False:
        zones_not_nan.append(data3_cf.columns[z])
        szon = data3_cf.columns[z][:2]+'00'
        total_cap[szon] = total_cap[szon] + data_cap[data3_cf.columns[z]].fillna(0)

# 3) calculate generation / installed capacity = CF to have national CFs
tec3_list = []
for t in range(len(techs)):
    # The following line uses the total capacity as given in PEMMDB, however this also consists of zones NL012, NL032, NL033 and DE013 that are all nan in PECD
    # This would cause the WOF CF to have a maximum at ~0.3, which is not true
    #tec3_list.append( national_ac3.loc[techs[t]] / data_cap[zones_szon].loc[techs[t]] )
    
    # Hence the total capacities (exlcluding "nan-WOF-zones") was calculated again:
    tec3_list.append( national_ac3.loc[techs[t]] /  total_cap[zones_szon].loc[techs[t]])
    

data3_cf_national = pd.concat(tec3_list,keys=techs,names=['Technology'])
data3_cf_national.to_pickle(path_to_plot+'Data/PECD3_CF_TY'+str(ty)+'_national_hourly.pkl')

print('Calculated national capacity factors for PECDv3.1')

# Calculate Residual Load and save it

start_date = data3_demand_sum_d.index.get_level_values(1)[0]
end_date   = data3_demand_sum_d.index.get_level_values(1)[-1]
data3_cropped1 = data3_ac_tsum_d.query('Date>=@start_date and Date <= @end_date')
data3_cropped2 = data3_cropped1[~((data3_cropped1.index.get_level_values(1).day == 29) & (data3_cropped1.index.get_level_values(1).month == 2))]

data3_rl_tsum_d = data3_demand_sum_d - data3_cropped2 

data3_rl_tsum_d.to_pickle(path_to_plot+'Data/PECD3_PEMMDB_RL_TY'+str(ty)+'_national_daily.pkl')
print('Calculated Residual Load for PECDv3.1')




#%% Load the ERAA23 demand data (based on PECD3.X)
# (This data ended up not being used in the analysis)

time_index = pd.Series(np.zeros(35*365*24)) # 35 years * 365 days * 24 hours
demand     = np.zeros(35*365*24) # 35 years * 365 days * 24 hours
data3_eraa_demand = pd.DataFrame()
for z in range(len(zones_szon)):
    temp = pd.read_excel(path_to_eraa23_d+'Demand_Timeseries_TY2033.xlsx',sheet_name=zones_szon[z], header=1)
    for y in range(2,len(temp.columns)):
        yidx=y-2
        h = temp['Hour']-1
        time_index[yidx*365*24:(yidx+1)*365*24] =  pd.to_datetime(temp['Date'].astype(str)+str(temp.columns[y]) +' '+h.astype(str)+':00:00', format="%d.%m.%Y %H:%M:%S")
        demand[yidx*365*24:(yidx+1)*365*24] = temp[temp.columns[y]]
    data3_eraa_demand[zones_szon[z]] = demand
data3_eraa_demand.index = time_index
data3_eraa_demand.index = pd.to_datetime(data3_eraa_demand.index)
data3_eraa_demand.index.rename('Date', inplace=True)

data3_eraa_demand.to_pickle(path_to_plot+'Data/ERAA23_demand_TY2033_hourly.pkl')

# ERAA (PECD3.1) demand
data3_eraa_demand_sum_y = data3_eraa_demand.groupby([data3_eraa_demand.index.get_level_values('Date').year]).sum()
data3_eraa_demand_sum_d = get_daily_values(data3_eraa_demand,'sum')

data3_eraa_demand_sum_y.to_pickle(path_to_plot+'Data/ERAA23_demand_TY2033_annual.pkl')
data3_eraa_demand_sum_d.to_pickle(path_to_plot+'Data/ERAA23_demand_TY2033_daily.pkl')
print('Preprocessed ERAA23 demand data')

#%% Load the ERAA23 ENS data (based on PECD3.X)

df = pd.DataFrame()
list_dates = np.empty((0),dtype='datetime64[ns]')
list_values = np.empty((0,len(eval_szon))) # dimensions: (date,szon)

years_available = np.arange(1982,2016+1) # TODO: Change, when more is available
for y in range(len(years_available)):
    if years_available[y] == 2014:
        # There is no file for 2014 at the moment. TODO: Update if 2014 is available.
        # Current solution: add synthetic data with no ENS in 2014
        yy = y - 1

        # Load in csv file of a certain year
        tmp_csv = pd.read_csv(path_to_eraa23_ens+'TY2033 Post-EVA FB  S2 CY'+str(years_available[yy])+'_ENS_allzones.csv', header=0)
        # Take only the entries of relevant 'evaluation' zones
        cut_rows = tmp_csv[tmp_csv['Child Name'].isin(eval_szon)][['Child Name','Datetime','Value']]
        cut_rows['Value'] = 0

    else:

        # Load in csv file of a certain year
        tmp_csv = pd.read_csv(path_to_eraa23_ens+'TY2033 Post-EVA FB  S2 CY'+str(years_available[y])+'_ENS_allzones.csv', header=0)
        # Take only the entries of relevant 'evaluation' zones
        cut_rows = tmp_csv[tmp_csv['Child Name'].isin(eval_szon)][['Child Name','Datetime','Value']]
        
    # Correct Datetime (climate year instead of 2033 & datetime format)
    dates = pd.to_datetime(cut_rows[cut_rows['Child Name']==eval_szon[0]]['Datetime']) + pd.DateOffset(years=years_available[y]-2033)
    list_dates = np.concatenate((list_dates,dates.values),axis=0)
    
    # make zones = columns
    tmp = np.zeros((len(dates),len(eval_szon)))
    for c in range(len(eval_szon)):
        tmp[:,c] = cut_rows[cut_rows['Child Name']==eval_szon[c]]['Value'].values
    
    # save this in an array
    list_values = np.concatenate((list_values,tmp),axis=0)
    print(years_available[y])
    
# put list of all years together into one big dataframe
data3_eraa23_ens_h = pd.DataFrame(index=list_dates, columns=eval_szon, data=list_values)
data3_eraa23_ens_h.index.rename('Date', inplace=True)
data3_eraa23_ens_sum_y = data3_eraa23_ens_h.groupby([data3_eraa23_ens_h.index.get_level_values('Date').year]).sum()
data3_eraa23_ens_sum_d = get_daily_values(data3_eraa23_ens_h,'sum')

data3_eraa23_ens_h.to_pickle(path_to_plot+'Data/ERAA23_ENS_TY2033_hourly.pkl')
data3_eraa23_ens_sum_y.to_pickle(path_to_plot+'Data/ERAA23_ENS_TY2033_annual.pkl')
data3_eraa23_ens_sum_d.to_pickle(path_to_plot+'Data/ERAA23_ENS_TY2033_daily.pkl')

print('Preprocessed ERAA23 ENS data')


#%% Load the Adequacy Outlook ENS data (based on PECD3.X)

# Old approach, not good because it considers leap days (demand data does not!)
def datetimes_from_years_and_hours(years, hours_of_year):
    start_of_years = np.array([np.datetime64(datetime(year, 1, 1, 0, 0, 0)) for year in years])
    
    # Calculate the timedelta for each hour of the year in the array
    delta_hours = np.array(hours_of_year, dtype=float)
    delta_timedeltas = np.timedelta64(1, 'h') * delta_hours
    
    # Obtain an array of datetimes by adding the timedeltas to the start of the years
    result_datetimes = start_of_years + delta_timedeltas
    
    return result_datetimes
         
# New, not nice because it is handpicked, but after several frustrating hours of wasting time, this is the best I could come up with
dates = np.arange(np.datetime64("1982-01-01 00:00:00"), np.datetime64("2016-12-31 00:00:00"), np.timedelta64(1, "h"))
df = pd.DataFrame(index=pd.to_datetime(dates))
df.index.rename('Date', inplace=True)
df = df[~((df.index.get_level_values('Date').day == 29) & (df.index.get_level_values('Date').month == 2))]
df = df[~((df.index.get_level_values('Date').day == 31) & (df.index.get_level_values('Date').month == 12))]


temp = pd.read_csv(path_to_ao_ens+'NEL.'+ao_scen+'.tunedAreaData.csv', header=0)       # load the data
i=1 # which iteration to take into account
cy =temp[temp['Iteration']==i]['ClimateYear']
hoy=temp[temp['Iteration']==i]['HourOfYear']

#time_index = pd.to_datetime(datetimes_from_HoY(hoy,cy))
time_index = df.index
ens_eval_szon=[]
for s in eval_szon:
    ens_eval_szon.append('ENS_'+s)
data_tmp = np.asarray(temp[temp['Iteration']==i][ens_eval_szon])


data3_ens = pd.DataFrame(index=time_index, columns=eval_szon, data=np.asarray(temp[temp['Iteration']==i][ens_eval_szon]))
data3_ens.index = pd.to_datetime(data3_ens.index)
data3_ens.index.rename('Date', inplace=True)

data3_ens.to_pickle(path_to_plot+'Data/AO_'+ao_scen+'_ENS_hourly.pkl')

# AO (PECD3.1) ENS
data3_ens_sum_y = data3_ens.groupby([data3_ens.index.get_level_values('Date').year]).sum()
data3_ens_sum_d = get_daily_values(data3_ens,'sum')

data3_ens_sum_y.to_pickle(path_to_plot+'Data/AO_'+ao_scen+'_ENS_annual.pkl')
data3_ens_sum_d.to_pickle(path_to_plot+'Data/AO_'+ao_scen+'_ENS_daily.pkl')
print('Preprocessed Adequacy Outlook ENS data')


