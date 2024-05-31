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
from Dunkelflaute_function_library import get_daily_values_etm, assign_SZON

# Specify parameters

path_to_pecd4     = 'F:/C3S_PECD_v4.1/'                         #'D:/PECD4_1/'
path_to_pecd3     = 'F:/PECD3_1/'            #'D:/PECD3_1/'
path_to_etm_d     = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/PECD4_ETM_demand/'           #'D:/PECD4_1/ETM_Demand/exports/demand/'
path_to_eraa23_d  = '' # This data ended up not being used                  #'D:/ERAA23/Demand Dataset/' 
path_to_eraa23_ens= 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/ERAA23_ENS/'                 #'D:/ERAA23/ENS/'
path_to_ao_ens    = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Adequacy_Outlook_ENS/'      #'D:/AdequacyOutlook/'
path_to_pemmdb_c  = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/PEMMDB_Installed_Capacities/' #'D:/PEMMDB/data_TY2033/01_PEMMDB/'
path_to_pemmdb_d  = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/PEMMDB_TY2033_demand/'    #'D:/PEMMDB/data_TY2033/04_LOAD/TY2033/'
path_to_plot      = 'C:/Users/cozianbas/Documents/Analyses PECD/Scripts/Data_Dunkelflaute_analysis/Dunkelflaute_plots/'      #'D:/Dunkelflaute/'
#path_to_shapefile = 'D:/PECD4_1/ShapeFiles/General/' # Not used

# Countries (NUT0) of interest (must be a list of two letter abbreviations)
#countries = ['DE','NL'] # right now only data for DE and NL is certainly available

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

# Target Years
ty_pecd3 = 2033
ty_pecd4 = 2030

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
        
#zones_peon = get_zones(countries,'PEON')  
#zones_szon = get_zones(countries,'SZON')  
#zones_peof = get_zones(countries,'PEOF')  

scen_timespan = [42,51,51,51] # TODO: Automate the nr of years per scenario
  



















#%% Load the PEMMDB demand data (1982-2016), Target Year: 2033

# Sheets sometimes have different formats
# Two formats:     short = Date, Hour, 1982, 1983, ..., 2016    AND     long = Date, Month, Day, Hour, 1981, 1983, ..., 2019

# SZON regions that are not in Demand dataset:
#'DZ00', Algeria
#'EG00', Egypt
#'FR15', Corsica
#'IL00', Israel
#'IS00', Iceland
#'JO00', Jordania
#'LB00', Lebanon
#'LU00', Luxembourg
#'LY00', Lybia
#'MA00', Marocco
#'MD00', ???
#'PS00', ???
#'SY00', Syria
#'TN00', Tunisia
#'UA01', ???
#'UA02', ???

SZON_available_PEMMDB = ['AL00','AT00','BA00','BE00','BG00','CH00','CY00','CZ00','DE00','DKE1',
                         'DKW1','EE00','ES00','FI00','FR00','GR00','GR03','HR00','HU00','IE00',
                         'ITCA','ITCN','ITCS','ITN1','ITS1','ITSA','ITSI','LT00','LV00','ME00',
                         'MK00','MT00','NL00','NOM1','NON1','NOS0','PL00','PT00','RO00','RS00',
                         'SE01','SE02','SE03','SE04','SI00','SK00','TR00','UK00','UKNI']

zones_szon = SZON_available_PEMMDB


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
    for y, year in enumerate(years):
        if zones_szon[z] in ['CY00','ES00','HR00','MT00','RO00']:
            # Some countries have a different date format
            time_index[y*365*24:(y+1)*365*24] = pd.Series(pd.date_range(start='2018-01-01 00:00:00', end='2018-12-31 23:00:00', freq='h')).apply(lambda x: x.replace(year=year))
        else:
            h = data_raw['Hour']-1
            time_index[y*365*24:(y+1)*365*24] = pd.to_datetime(data_raw['Date'].astype(str)+str(year) +' '+h.astype(str)+':00:00', format="%d.%m.%Y %H:%M:%S")
        demand[y*365*24:(y+1)*365*24] = data_raw[year]
    
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




















#%% Load the ERAA23 ENS data (based on PECD3.X)

# SZON common to Demand dataset for PECD 3.1 and ENS data (first version transmitted by Laurens)
SZON_intersection_Demand_ENS = ['SI00', 'ITSA', 'ITSI', 'GR00', 'ITS1', 'UK00', 'LV00', 'BE00', 'MT00', 'SE03', 'CH00', 'IE00', 'ITN1', 'LT00', 'DKE1', 'BA00', 'RO00', 'AL00', 'PL00', 'FR00', 'DE00', 'SK00', 'EE00', 'ITCS', 'ITCA', 'AT00', 'NL00', 'ME00', 'BG00', 'CY00', 'MK00', 'DKW1', 'HU00', 'PT00', 'ES00', 'SE01', 'SE04', 'ITCN', 'UKNI', 'NON1', 'NOS0', 'CZ00', 'NOM1', 'SE02', 'FI00', 'HR00', 'RS00']

eval_szon = SZON_intersection_Demand_ENS

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




















#%% Load the PEMMDB capacity data of target year 
# Run this before loading any of the PECD data
# We use ty_pecd3 = 2033 for validation

ty_pecd3 = 2033

"""
SZON_available_PEMMDB_IC = ['AL00','AT00','BA00','BE00','BEOF','BG00','CH00','CY00','CZ00','DE00',
                            'DEKF','DKBH','DKE1','DKKF','DKNS','DKW1','EE00','ES00','FI00','FR00',
                            'GR00','GR03','HR00','HU00','IE00','ITA0','ITCA','ITCN','ITCS','ITN1',
                            'ITS1','ITSA','ITSI','LT00','LUB1','LUF1','LUG1','LUV1','LV00','ME00',
                            'MK00','MT00','NL00','NLLL','NOM1','NON1','NOS0','PL00','PT00','RO00',
                            'RS00','SE01','SE02','SE03','SE04','SI00','SK00','UK00','UKNI']
"""
SZON_intersection_Demand_IC = ['NL00', 'ITS1', 'ES00', 'DE00', 'ITCN', 'FI00', 'SI00', 'ME00', 'FR00', 'NOM1', 
                               'UKNI', 'SE01', 'RO00', 'CZ00', 'PL00', 'HU00', 'MK00', 'BA00', 'CY00', 'IE00', 
                               'NOS0', 'GR00', 'LT00', 'ITCA', 'NON1', 'ITCS', 'ITSI', 'RS00', 'UK00', 'SE03', 
                               'EE00', 'HR00', 'SE02', 'AL00', 'BE00', 'MT00', 'ITSA', 'LV00', 'GR03', 'CH00', 
                               'AT00', 'DKW1', 'PT00', 'DKE1', 'BG00', 'SK00', 'ITN1', 'SE04']

#SZON_intersection_Demand_IC = ['NL00', 'DE00']

zones_szon = SZON_intersection_Demand_IC

if np.isin(ty_pecd3, [2022, 2025, 2027, 2030, 2033, 2040, 2050])==False:
    raise KeyError('Targetyear (ty) must be either 2022, 2025, 2027, 2030, 2033, 2040 or 2050 and not '+str(ty_pecd3))

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
    data_won = data_raw.loc[(data_raw['Technology']=='Wind Onshore')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_pecd3]].set_index('PECD Zone')
    data_wof = data_raw.loc[(data_raw['Technology']=='Wind Offshore')&(data_raw['PECD Zone'].isin(peof_s))][['PECD Zone',ty_pecd3]].set_index('PECD Zone')
    data_spv_rt = data_raw.loc[(data_raw['Technology']=='Solar PV Rooftop')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_pecd3]].set_index('PECD Zone')
    data_spv_f  = data_raw.loc[(data_raw['Technology']=='Solar PV Farm')&(data_raw['PECD Zone'].isin(peon))][['PECD Zone',ty_pecd3]].set_index('PECD Zone')
    
    # Replace "slim" PEOF zones with extended names to avoid confusion with PEON zones
    index_mapping = dict(zip(peof_s, peof))
    data_wof.rename(index=index_mapping, inplace=True)

    # Combine SPVs
    data_spv = data_spv_rt + data_spv_f
    
    # Rename the 'Total' columns into the corresponding study zone
    data_spv.rename(index={"Total": zones_szon[c]}, inplace=True)
    data_wof.rename(index={"Total": zones_szon[c]}, inplace=True)
    data_won.rename(index={"Total": zones_szon[c]}, inplace=True)

    # For small territories, only one PEON per SZON.
    # SZON and PZON have the same name => Issue because 2 identical index.
    # Drop one occurrence of the duplicate index
    if len(data_spv.loc[zones_szon[c]]) > 1:
        data_spv = data_spv.loc[[zones_szon[c]]].head(1)
    if len(data_wof.loc[zones_szon[c]]) > 1:
        data_wof = data_wof.loc[[zones_szon[c]]].head(1)
    if len(data_won.loc[zones_szon[c]]) > 1:
        data_won = data_won.loc[[zones_szon[c]]].head(1)
    
    # Reshape, so that zones = columns and technologies = index
    tec_list = pd.concat([data_spv.T, data_wof.T, data_won.T], keys=techs, names=['Technology'])
    tec_list = tec_list.droplevel(level=1)
    
    cap_list.append(tec_list)
   
data_cap = pd.concat(cap_list, axis=1)
data_cap.to_pickle(path_to_plot+'Data/PEMMDB_capacities_TY'+str(ty_pecd3)+'.pkl')

print('Preprocessed PEMMDB capacity data')





















#%% Load the PECD 3.1 data
# PEMMDB installed capacities with the right target year must be already loaded!
# If not, run the PEMMDB capacity loading cell before running this cell.
# Keep track that the same targetyear is chosen!

"""
countries = []
for c in range(len(zones_szon)):
    countries.append(zones_szon[c][:-2])
countries = list(np.unique(countries))
"""
# Countries in `SZON_intersection_Demand_IC`
countries = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK', 
             'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 
             'LV', 'ME', 'MK', 'MT', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 
             'SE', 'SI', 'SK', 'UK']

# Some zone are absent in PECD v3.1 or the PEMMDB Demand 
zone_to_remove = ['ITCA', 'ITCA_OFF', 'FR15', 'DKBI_OFF', 'DKKF_OFF']

# We use TY 2033 for validation
ty_pecd3 = 2033

if not ty_pecd3==2033:
    raise KeyError('Targetyear (ty) must be 2033 and not '+str(ty))

data3_ENS_d = pd.read_pickle(path_to_plot+'Data/PEMMDB_capacities_TY'+str(ty_pecd3)+'.pkl')

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

                for zone_rm in zone_to_remove:
                    if zone_rm in zones:
                        print(f'Remove zone {zone_rm}')
                        zones.remove(zone_rm)
                
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
data3_ac.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_zonal_hourly.pkl')
data3_cf_mean_y.to_pickle(path_to_plot+'Data/PECD3_CF_zonal_annual.pkl')
data3_cf_mean_d.to_pickle(path_to_plot+'Data/PECD3_CF_zonal_daily.pkl')
data3_ac_sum_y.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_zonal_annual.pkl')
data3_ac_sum_d.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_zonal_daily.pkl')

print('Preprocessed PECDv3.1 data')

# Calculate zonal aggregations of Generation (for Otero et al 22 method)

data3_ac_national_h = pd.DataFrame()
data3_ac_national_d = pd.DataFrame()

zones_szon = SZON_intersection_Demand_IC

for zone_szon in zones_szon:
    # make a list of all Off- and Onshore zones of a country
    country = zone_szon[:2]
    peon_country = np.asarray(get_zones([country],'PEON'))
    peof_country = np.asarray(get_zones([country],'PEOF'))
    szon_of_peon = np.asarray(assign_SZON(peon_country, 'PEON'))
    szon_of_peof = np.asarray(assign_SZON(peof_country, 'PEOF'))
    # Get lists of PEON/PEOF for the SZON zone_szon
    peon_of_szon = peon_country[szon_of_peon==zone_szon]
    peof_of_szon = peof_country[szon_of_peof==zone_szon]
    print(f'peon_of_szon = {peon_of_szon}')
    print(f'peof_of_szon = {peof_of_szon}')
    pecd_of_szon = list(peon_of_szon) + list(peof_of_szon) 

    for zone_rm in zone_to_remove:
        if zone_rm in pecd_of_szon:
            print(f'Remove zone {zone_rm}')
            pecd_of_szon.remove(zone_rm)

    # sum up all the zones per country
    data3_ac_national_h[zone_szon] = data3_ac[pecd_of_szon].sum(axis=1)
    data3_ac_national_d[zone_szon] = data3_ac_sum_d[pecd_of_szon].sum(axis=1)

# Sum up WOF, WON and SPV
data3_ac_tsum_h = data3_ac_national_h.groupby(['Scenario','Date']).sum()
data3_ac_tsum_d = data3_ac_national_d.groupby(['Scenario','Date']).sum()

data3_ac_tsum_h.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_national_hourly.pkl')
data3_ac_tsum_d.to_pickle(path_to_plot+'Data/PECD3_Generation_TY'+str(ty_pecd3)+'_national_daily.pkl')

print('Summed up all the zones of a country for PECDv3.1')


"""
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
data3_cf_national.to_pickle(path_to_plot+'Data/PECD3_CF_TY'+str(ty_pecd3)+'_national_hourly.pkl')

print('Calculated national capacity factors for PECDv3.1')
"""













# %%
