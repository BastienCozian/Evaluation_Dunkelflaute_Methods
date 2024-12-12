# -*- coding: utf-8 -*-
"""
This is a collection of all sorts of functions used in the Dunkelflaute analysis.

For questions, refer to benjamin.biewald@tennet.eu
"""

import glob
import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from itertools import combinations


#%% PECD specific functions

def get_zones(countries,agg):
    '''
    Provides PECD zones for a specified country.
    countries = list of two letter abbreviation strings (e.g. ['DE','NL'] for Germany & the Netherlands)
    agg = 'PEON', 'PEOF', 'SZON', 'SZOF' or 'NUT0'
    Returns a list of corresponding zones (e.g. ['DE01' 'DE02', ... , 'NL01','NL02', ...]).
    '''
    
    nut0_zones = ['AL','AT','BA','BE','BG','CH','CY','CZ','DE','DK','DZ','EE','EG',
                  'EH','EL','ES','FI','FR','GR','HR','HU','IE','IL','IS','IT','JO','LB',
                  'LI','LT','LU','LV','LY','MA','MD','ME','MK','MT','NL','NO','PL',
                  'PS','PT','RO','RS','SE','SI','SK','SY','TN','TR','UA','UK','XK']
    
    peon_zones = ['AL00','AT01','AT02','AT03','BA00','BE01','BE02','BE03','BG01',
                  'BG02','CH00','CY00','CZ01','CZ02','DE01','DE02','DE03','DE04',
                  'DE05','DE06','DE07','DKE1','DKW1','DZ01','DZ02','DZ03','EE00',
                  'EG00','ES01','ES02','ES03','ES04','ES05','ES06','ES07','ES08',
                  'ES09','ES10','ES11','ES12','FI01','FI02','FR01','FR02','FR03',
                  'FR04','FR05','FR06','FR07','FR08','FR09','FR10','FR11','FR12',
                  'FR13','FR14','FR15','GR01','GR02','GR03','HR01','HR02','HU01',
                  'HU02','HU03','IE00','IL00','IS00','ITCA','ITCN','ITCS','ITN1',
                  'ITS1','ITSA','ITSI','JO00','LB00','LT00','LU00','LV00','LY01',
                  'LY02','LY03','MA01','MA02','MA03','MA04','MA05','MA06','MD00',
                  'ME00','MK00','MT00','NL01','NL02','NL03','NL04','NOM1','NON1',
                  'NOS1','NOS2','NOS3','PL01','PL02','PL03','PL04','PL05','PS00',
                  'PT01','PT02','RO01','RO02','RO03','RS01','SE01','SE02','SE03',
                  'SE04','SI00','SK00','SY00','TN01','TN02','TN03','TN04','TN05',
                  'TN06','TN07','TN08','TR01','TR02','TR03','TR04','TR05','TR06',
                  'TR07','TR08','TR09','TR10','TR11','TR12','TR13','TR14','TR15',
                  'UA01','UA02','UK01','UK02','UK03','UK04','UK05','UKNI','XK00']
    
    peof_zones = ['AL00_OFF','BA00_OFF','BE01_OFF','BG01_OFF','CY00_OFF','DBOT_OFF',
                  'DBUK_OFF','DE011_OFF','DE012_OFF','DE013_OFF','DE02_OFF','DKBI_OFF',
                  'DKE1_OFF','DKKF_OFF','DKW11_OFF','DKW12_OFF','DZ00_OFF','EE00_OFF',
                  'EG00_OFF','ES01_OFF','ES02_OFF','ES04_OFF','ES06_OFF','ES09_OFF',
                  'ES10_OFF','ES11_OFF','FI01_OFF','FI02_OFF','FR01_OFF','FR02_OFF',
                  'FR03_OFF','FR04_OFF','FR08_OFF','FR09_OFF','FR13_OFF','GR01_OFF',
                  'GR02_OFF','GR03_OFF','HR01_OFF','IE00_OFF','IL00_OFF','IS00_OFF',
                  'ITCA_OFF','ITCN_OFF','ITCS_OFF','ITN1_OFF','ITS1_OFF','ITSA_OFF',
                  'ITSI_OFF','JO00_OFF','LB00_OFF','LT00_OFF','LV00_OFF','LY01_OFF',
                  'LY02_OFF','MA01_OFF','MA02_OFF','MA04_OFF','MA05_OFF','MA06_OFF',
                  'ME00_OFF','MT00_OFF','NL011_OFF','NL012_OFF','NL031_OFF','NL032_OFF',
                  'NL033_OFF','NOM1_OFF','NON1_OFF','NOS21_OFF','NOS22_OFF','NOS3_OFF',
                  'PL04_OFF','PL05_OFF','PS00_OFF','PT01_OFF','PT02_OFF','RO03_OFF',
                  'SE01_OFF','SE02_OFF','SE03_OFF','SE04_OFF','SI00_OFF','SY00_OFF',
                  'TN01_OFF','TN02_OFF','TN04_OFF','TN06_OFF','TN07_OFF','TR01_OFF',
                  'TR02_OFF','TR03_OFF','TR04_OFF','TR06_OFF','TR07_OFF','TR10_OFF',
                  'TR11_OFF','UA02_OFF','UK011_OFF','UK012_OFF','UK02_OFF','UK031_OFF',
                  'UK032_OFF','UK041_OFF','UK042_OFF','UK043_OFF','UK051_OFF',
                  'UK052_OFF','UK053_OFF','UKNI_OFF']
    
    peof_zones_slim = ['AL00','BA00','BE01','BG01','CY00','DBOT',
                  'DBUK','DE011','DE012','DE013','DE02','DKBI',
                  'DKE1','DKKF','DKW11','DKW12','DZ00','EE00',
                  'EG00','ES01','ES02','ES04','ES06','ES09',
                  'ES10','ES11','FI01','FI02','FR01','FR02',
                  'FR03','FR04','FR08','FR09','FR13','GR01',
                  'GR02','GR03','HR01','IE00','IL00','IS00',
                  'ITCA','ITCN','ITCS','ITN1','ITS1','ITSA',
                  'ITSI','JO00','LB00','LT00','LV00','LY01',
                  'LY02','MA01','MA02','MA04','MA05','MA06',
                  'ME00','MT00','NL011','NL012','NL031','NL032',
                  'NL033','NOM1','NON1','NOS21','NOS22','NOS3',
                  'PL04','PL05','PS00','PT01','PT02','RO03',
                  'SE01','SE02','SE03','SE04','SI00','SY00',
                  'TN01','TN02','TN04','TN06','TN07','TR01',
                  'TR02','TR03','TR04','TR06','TR07','TR10',
                  'TR11','UA02','UK011','UK012','UK02','UK031',
                  'UK032','UK041','UK042','UK043','UK051',
                  'UK052','UK053','UKNI']
    
    szon_zones = ['AL00','AT00','BA00','BE00','BG00','CH00','CY00','CZ00','DE00',
                  'DKE1','DKW1','DZ00','EE00','EG00','ES00','FI00','FR00','FR15',
                  'GR00','GR03','HR00','HU00','IE00','IL00','IS00','ITCA','ITCN',
                  'ITCS','ITN1','ITS1','ITSA','ITSI','JO00','LB00','LT00','LU00',
                  'LV00','LY00','MA00','MD00','ME00','MK00','MT00','NL00','NOM1',
                  'NON1','NOS0','PL00','PS00','PT00','RO00','RS00','SE01','SE02',
                  'SE03','SE04','SI00','SK00','SY00','TN00','TR00','UA01','UA02',
                  'UK00','UKNI']
    
    szof_zones = ['AL00','BA00','BE00','BG00','CY00','DBOT','DBUK','DE00','DKBI',
                  'DKE1','DKKF','DKW1','DZ00','EE00','EG00','ES00','FI00','FR00',
                  'GR00','HR00','IE00','IL00','IS00','ITCA','ITCN','ITCS','ITN1',
                  'ITS1','ITSA','ITSI','JO00','LB00','LT00','LV00','LY00','MA00',
                  'ME00','MT00','NL00','NOM1','NON1','NOS0','PL00','PS00','PT00',
                  'RO00','SE01','SE02','SE03','SE04','SI00','SY00','TN00','TR00',
                  'UA02','UK00','UKNI']
    
    peon=[]
    peof=[]
    peof_slim=[]
    szon=[]
    szof=[]
    for c in countries:
        if c in nut0_zones:
            peon.extend([s for s in peon_zones if s.startswith(c)])
            peof.extend([s for s in peof_zones if s.startswith(c)])
            peof_slim.extend([s for s in peof_zones_slim if s.startswith(c)])
            szon.extend([s for s in szon_zones if s.startswith(c)])   
            szof.extend([s for s in szof_zones if s.startswith(c)])  
        else:
            raise KeyError('Country '+c+' not found! Please choose countries from the NUT0 list.')
    
    if agg=='PEON':
        return peon
    elif agg=='PEOF':
        return peof
    elif agg=='PEOF_slim':
        return peof_slim
    elif agg=='SZON':
        return szon
    elif agg=='SZOF':
        return szof
    else:
        return countries
    
#%% Dataframe & Calculating functions

def find_idx_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin(axis=0)
    return idx
    
def get_df_timerange(df, start_date_a, end_date_a):
    '''
    Returns dataframe with TimeIndex between given start and end dates
    '''
    return df.query('Date>=@start_date_a and Date <= @end_date_a')

def get_f_score(ens_mask_d, df_mask_d, beta=1):
    # df_mask_d = DataFrame with masked data of daily Dunkelflauten (0=no DF,  1=DF)
    # ens_mask_d = DataFrame with masked data of ENS (0=no ENS, 2=ENS)
    # beta = weight of recall (how much more important is detecting accuracy in detecting DF (less missing out) over false alarming)
    
    # Procedure:
    # 1.) Define only the common time frame of ENS and DF dataframes
    # 2.) Mask the data into 0 = no DF, 1 = DF and 0= no ENS, 2=ENS (premade by function mask_data)
    # 3.) Add both up and obtain: 0 = True negative, 1 = False positive, 2 = False negative, 3 = True positive
    # 4.) From the counted values, calculate f
    
    
    same_index = df_mask_d.index.intersection(ens_mask_d.index)
    detection_mask = ens_mask_d.loc[same_index] + df_mask_d.loc[same_index]

    true_positive  = (detection_mask==3).sum()
    false_negative = (detection_mask==2).sum()
    false_positive = (detection_mask==1).sum()
    true_negative  = (detection_mask==0).sum()
    
    f = ((1+beta**2)*true_positive) / ((1+beta**2)*true_positive + beta**2 * false_negative + false_positive)
    # f is between 0 and 1, 0= precision or recall are zero, 1=perfect precision and recall
    # precision = ratio of true positives over all positives (how many events are wrongly detected?)
    # recall = ratio of true positives over true positive+false negatives (how many events are missed out?)
    
    data = [f, true_positive, false_negative, false_positive, true_negative]
    result_df = pd.DataFrame(index=['F','TP','FN','FP','TN'], columns=f.index.values, data=data)
    
    return result_df


def mask_data(df, thresh, below, true_value, false_value):
    # Masks the data following when data<thresh (below=True) or data>thresh (below=False)
    if below:
        mask_array = np.where(np.array(df)<thresh, true_value, false_value)
    else:
        mask_array = np.where(np.array(df)>thresh, true_value, false_value)
        
    return_df = pd.DataFrame(index=df.index, columns=df.columns, data=mask_array)
    return return_df

def mask_df_by_entries(df, df_hours, scenarios, true_value, false_value):
    # Masks a dataframe by another given dataframe of entries
    # e.g. every hour that occurs in df_hours, will result in a "true_value", else "false_value"
    # df must be CF dataframe
    # df_days must be cf_hours dataframe 
    
    scen_list = []
    for s in range(len(scenarios)):
        mask_array = np.empty((len(df.loc[('SPV',scenarios[s])].index.get_level_values('Date')),len(df.columns)))
        for c in range(len(df.columns)):
            true_dates = df_hours.loc[(scenarios[s],df.columns[c])]
            mask_array[:,c] = np.where(df.loc[('SPV',scenarios[s])][df.columns[c]].index.get_level_values('Date').isin(true_dates['Date']), true_value, false_value)
            
        scen_list.append(pd.DataFrame(index=df.loc[('SPV',scenarios[s])].index.get_level_values('Date'), columns=df.columns, data=mask_array))
    
    return_df = pd.concat(scen_list,   keys=scenarios, names=['Scenario'])
    return return_df


def cf2cap(cf, cap, refyear, zones):
    # transforms zonal capacity factor data into absolute capacities based on one reference year
    # cf = dataframe with CF data of one scenario and one variable ("data_zones")
    # cap = annual total capacities for all zones and total of zones
    
    ref_cap = cap.loc[(refyear),(zones)]
    
    result = cf * ref_cap
    return result


def datetimes_from_years_and_hours(years, hours_of_year):
    start_of_years = np.array([np.datetime64(datetime(year, 1, 1, 0, 0, 0)) for year in years])
    
    # Calculate the timedelta for each hour of the year in the array
    delta_hours = np.array(hours_of_year, dtype=float)
    delta_timedeltas = np.timedelta64(1, 'h') * delta_hours
    
    # Obtain an array of datetimes by adding the timedeltas to the start of the years
    result_datetimes = start_of_years + delta_timedeltas
    
    return result_datetimes


def get_daily_values_pecd(df, mean_or_sum):
    # Calculated the daily means or sums of a given hourly dataframe (with Multiindex as loaded by the PECD data loading functions)
    # Also reorganizes the Date column (that is split into Year, Month, Day during the process) back into one Date column (Datetype)
    if mean_or_sum=='mean':
        data_cf_mean_d = df.groupby(['Technology','Scenario', df.index.get_level_values('Date').year, df.index.get_level_values('Date').month, df.index.get_level_values('Date').day]).mean()
    else:
        data_cf_mean_d = df.groupby(['Technology','Scenario', df.index.get_level_values('Date').year, df.index.get_level_values('Date').month, df.index.get_level_values('Date').day]).sum()
    data_cf_mean_d.index.rename(['Year','Month','Day'],level=[2,3,4], inplace=True)
    # Reorganizing multiindex, so only one date column is left
    data_cf_mean_d = data_cf_mean_d.reset_index(['Technology', 'Scenario'])
    data_cf_mean_d.index = pd.to_datetime(data_cf_mean_d.index.get_level_values('Year').astype(str) + '-' +
                               data_cf_mean_d.index.get_level_values('Month').astype(str) + '-' +
                               data_cf_mean_d.index.get_level_values('Day').astype(str),
                               format='%Y-%m-%d')
    data_cf_mean_d = data_cf_mean_d.set_index(['Technology', 'Scenario'], append=True)
    data_cf_mean_d = data_cf_mean_d.reorder_levels(['Technology', 'Scenario', None])
    data_cf_mean_d.index.rename(['Date'],level=[2], inplace=True)
    return data_cf_mean_d

def get_daily_values_etm(df, mean_or_sum):
    # Calculated the daily means or sums of a given hourly dataframe (with Multiindex as loaded by the ETM demand data loading function)
    # Also reorganizes the Date column (that is split into Year, Month, Day during the process) back into one Date column (Datetype)
    if mean_or_sum=='mean':
        data_cf_mean_d = df.groupby(['Scenario', df.index.get_level_values('Date').year, df.index.get_level_values('Date').month, df.index.get_level_values('Date').day]).mean()
    else:
        data_cf_mean_d = df.groupby(['Scenario', df.index.get_level_values('Date').year, df.index.get_level_values('Date').month, df.index.get_level_values('Date').day]).sum()
    data_cf_mean_d.index.rename(['Year','Month','Day'],level=[1,2,3], inplace=True)
    # Reorganizing multiindex, so only one date column is left
    data_cf_mean_d = data_cf_mean_d.reset_index(['Scenario'])
    data_cf_mean_d.index = pd.to_datetime(data_cf_mean_d.index.get_level_values('Year').astype(str) + '-' +
                               data_cf_mean_d.index.get_level_values('Month').astype(str) + '-' +
                               data_cf_mean_d.index.get_level_values('Day').astype(str),
                               format='%Y-%m-%d')
    data_cf_mean_d = data_cf_mean_d.set_index(['Scenario'], append=True)
    data_cf_mean_d = data_cf_mean_d.reorder_levels(['Scenario', None])
    data_cf_mean_d.index.rename(['Date'],level=[1], inplace=True)
    return data_cf_mean_d

def get_daily_values(df, mean_or_sum):
    # Calculated the daily means or sums of a given hourly dataframe
    # Also reorganizes the Date column (that is split into Year, Month, Day during the process) back into one Date column (Datetype)
    if mean_or_sum=='mean':
        data_cf_mean_d = df.groupby([df.index.get_level_values('Date').year, df.index.get_level_values('Date').month, df.index.get_level_values('Date').day]).mean()
    else:
        data_cf_mean_d = df.groupby([df.index.get_level_values('Date').year, df.index.get_level_values('Date').month, df.index.get_level_values('Date').day]).sum()
    data_cf_mean_d.index.rename(['Year','Month','Day'],level=[0,1,2], inplace=True)
    # Reorganizing multiindex, so only one date column is left
    data_cf_mean_d.index = pd.to_datetime(data_cf_mean_d.index.get_level_values('Year').astype(str) + '-' +
                               data_cf_mean_d.index.get_level_values('Month').astype(str) + '-' +
                               data_cf_mean_d.index.get_level_values('Day').astype(str),
                               format='%Y-%m-%d')
    data_cf_mean_d.index.rename('Date',inplace=True)
    return data_cf_mean_d


def get_thresholds(data, perc, start_date='1980-01-01', end_date='2016-12-31', empirical=True):
    # data = daily generation / residual load data, e.g. from 1980-2020
    # perc = percentile (e.g. 10th, 90th)
    # below = if below percentile is drought (for generation drought True, for load drought false)
    # empirical = if empirical quantile is used (means cumulative days), else: cumulative generation
    
    #start_date_a = '1980-01-01' # earliest possible (in paper they use 1979)
    #end_date_a = '2019-12-31' # exactly 30 years (in paper they use until 2019)
    # obtain reference distribution
    tmp_data_thresh = get_df_timerange(data, start_date, end_date)
    #tmp_data_thresh = data.query('Date>=@start_date_a and Date <= @end_date_a') 
    
    # Calculate sntandard deviation of reference distribution
    sigma = tmp_data_thresh.std()
    
    # Calculate cumulative distributions (either empirical=occurances, or based on generation)
    if empirical:
        sorted = np.sort(np.array(tmp_data_thresh ,dtype=float), axis=0)
        # Filter out nan values
        # Old version (below) whould remove any time step whoich is nan for at least one country
        #cumu_x[~np.isnan(cumu_x).any(axis=1), :]
        # Attempt new version (below) remove countries with at least one timestep as NaN (not sure this is useful but we keep it for consistency for now)
        #cumu_x = sorted[:,~np.isnan(sorted).any(axis=0)] 
        # New version: do not remove NaN, as it is dealed with most of the time later
        cumu_x = sorted
        mdays = np.arange(0,cumu_x.shape[0])
        cumu_y = np.meshgrid(np.arange(cumu_x.shape[1]),mdays)[1] # creates arrays with columns = zones_szon and rows = cumulative nr of days
        cumu_y = cumu_y/cumu_y[-1,:]
    else:
        cumu_x = np.sort(np.array(tmp_data_thresh ,dtype=float), axis=0)
        # Filter out nan values
        cumu_x = cumu_x[~np.isnan(cumu_x).any(axis=1), :]
        cumu_y = np.cumsum(cumu_x, axis=0)
        cumu_y = cumu_y/cumu_y[-1,:]
        
    # find corresponding threshold value
    thresh = np.zeros(len(tmp_data_thresh.columns))
    thresh_idx = find_idx_nearest(cumu_y, perc)
    for c in range(len(tmp_data_thresh.columns)):
        thresh[c] = cumu_x[thresh_idx[c],c]
    
    return thresh, sigma


# Count the Dunkelflaute events and save them in a dataframe (Otero Method)
def detect_drought_Otero22(scenarios, zones_szon, data_ac_tsum_d, thresh, sigma, below=True):  
    df_days_scen_list=[]
    df_events_scen_list=[]
    for s in range(len(scenarios)):
        df_days_coun_list=[]
        df_events_coun_list=[]
        for c in range(len(zones_szon)):
            
            dates = data_ac_tsum_d.loc[(scenarios[s]),(zones_szon[c])].index
            gen_arr = np.array(data_ac_tsum_d.loc[(scenarios[s]),(zones_szon[c])], dtype=float)
            
            # Initialize the duration counter and Sverity sum
            day_counter=0
            sev_sum = 0
            sev_adapt_sum = 0
            
            # Count DF days and events and do statistics
            
            dunkelflaute_days = []
            startdate_list = []
            duration_list = []
            severity_list = []
            severity_adapt_list = []
            if below:
                for d in range(gen_arr.shape[0]-1):
                    if gen_arr[d]<thresh[c]:                                          # Drought day detected
                        dunkelflaute_days.append(dates[d])
                        if day_counter==0:                                                  # Start of Drought event
                            startdate_list.append(dates[d])
                        day_counter+=1
                        sev_sum+= np.abs(thresh[c]-gen_arr[d])
                        sev_adapt_sum += thresh[c]-gen_arr[d]
                    elif gen_arr[d]>=thresh[c] and gen_arr[d+1]<thresh[c]:          # one day "break", but event continues after that
                        if day_counter>0:                                                   # Continue with drought event
                            dunkelflaute_days.append(dates[d])
                            day_counter+=1
                            sev_sum+= np.abs(thresh[c]-gen_arr[d])
                            sev_adapt_sum += thresh[c]-gen_arr[d]
                    elif gen_arr[d]>=thresh[c] and gen_arr[d+1]>=thresh[c]:         # two days in row no drought day
                        if day_counter>0:                                                   # End of Drought Event
                            # Save results in List
                            duration_list.append(day_counter)
                            severity_list.append(sev_sum/sigma[zones_szon[c]])
                            severity_adapt_list.append(sev_adapt_sum/sigma[zones_szon[c]])
                            day_counter=0
                            sev_sum=0
                            sev_adapt_sum=0
            else:
                for d in range(gen_arr.shape[0]-1):
                    if gen_arr[d]>thresh[c]:                                          # Drought day detected
                        dunkelflaute_days.append(dates[d])
                        if day_counter==0:                                                  # Start of Drought event
                            startdate_list.append(dates[d])
                        day_counter+=1
                        sev_sum+= np.abs(thresh[c]-gen_arr[d])
                        sev_adapt_sum += gen_arr[d]-thresh[c]
                    elif gen_arr[d]<=thresh[c] and gen_arr[d+1]>thresh[c]:          # one day "break", but event continues after that
                        if day_counter>0:                                                   # Continue with drought event
                            dunkelflaute_days.append(dates[d])
                            day_counter+=1
                            sev_sum+= np.abs(thresh[c]-gen_arr[d])
                            sev_adapt_sum += gen_arr[d]-thresh[c]
                    elif gen_arr[d]<=thresh[c] and gen_arr[d+1]<=thresh[c]:         # two days in row no drought day
                        if day_counter>0:                                                   # End of Drought Event
                            # Save results in List
                            duration_list.append(day_counter)
                            severity_list.append(sev_sum/sigma[zones_szon[c]])
                            severity_adapt_list.append(sev_adapt_sum/sigma[zones_szon[c]])
                            day_counter=0
                            sev_sum=0
                            sev_adapt_sum=0
 
            if day_counter>0:
                # Save results in List in case the dataset ended within a Dunkelflaute
                duration_list.append(day_counter)
                severity_list.append(sev_sum/sigma[zones_szon[c]])
                severity_adapt_list.append(sev_adapt_sum/sigma[zones_szon[c]])
                day_counter=0
                sev_sum=0
                sev_adapt_sum=0
            
            # check if list is empty (when no DFs could be detected)
            if not dunkelflaute_days:
                dunkelflaute_days.append(np.nan)
                
            df_days_coun_list.append(pd.DataFrame(data=dunkelflaute_days, index=np.arange(len(dunkelflaute_days)), columns=['Date']))
            df_events_coun_list.append(pd.DataFrame(data=np.array([startdate_list,duration_list,severity_list,severity_adapt_list]).T, index=np.arange(len(startdate_list)), columns=['Startdate','Duration','Severity','Severity (adapted)']))
        df_days_scen_list.append(pd.concat(df_days_coun_list, keys=zones_szon, names=['Countries']))
        df_events_scen_list.append(pd.concat(df_events_coun_list, keys=zones_szon, names=['Countries']))
    df_days   = pd.concat(df_days_scen_list,   keys=scenarios, names=['Scenario'])
    df_events = pd.concat(df_events_scen_list, keys=scenarios, names=['Scenario'])
    
    return df_days, df_events


# Count the Dunkelflaute events and save them in a dataframe (Li Method)
def detect_drought_Li21(scenarios, zones_szon, data_CF_h, thresh):
    df_hours_scen_list=[]
    df_events_scen_list=[]
    for s in range(len(scenarios)):
        df_hours_coun_list=[]
        df_events_coun_list=[]
        for c in range(len(zones_szon)):
            
            dates = data_CF_h.loc[('SPV',scenarios[s]),(zones_szon[c])].index
            gen_arr_spv = np.array(data_CF_h.loc[('SPV',scenarios[s]),(zones_szon[c])], dtype=float)
            gen_arr_wof = np.array(data_CF_h.loc[('WOF',scenarios[s]),(zones_szon[c])], dtype=float)
            gen_arr_won = np.array(data_CF_h.loc[('WON',scenarios[s]),(zones_szon[c])], dtype=float)
            gen_arr_sum = np.array(data_CF_h.loc[(['SPV','WOF','WON'],scenarios[s]),(zones_szon[c])].groupby(['Scenario','Date']).sum(), dtype=float)
            
            # Initialize the duration counter and Sverity sum
            hour_counter=0
            
            # Count DF days and events and do statistics
            
            dunkelflaute_hours = []
            spv_hours = []
            wof_hours = []
            won_hours = []
            startdate_list = []
            duration_list = []
            for d in range(gen_arr_sum.shape[0]-1):
                if gen_arr_spv[d]<thresh and gen_arr_wof[d]<thresh and gen_arr_won[d]<thresh:                                          # Drought hour detected
                    dunkelflaute_hours.append(dates[d])
                    if hour_counter==0:                                                  # Start of Drought event
                        startdate_list.append(dates[d])
                    hour_counter+=1
                else:                                       # No drought hour
                    if hour_counter>0:                                                   # End of Drought Event
                        # Save results in List
                        duration_list.append(hour_counter)
                        hour_counter=0
                  
                # individual "drought" days for each technology
                # This is for now not returned by the function, but might be for future use interesting
                if gen_arr_spv[d]<thresh:           
                    spv_hours.append(dates[d])
                if gen_arr_wof[d]<thresh:
                    wof_hours.append(dates[d])
                if gen_arr_won[d]<thresh:
                    won_hours.append(dates[d])
 
            if hour_counter>0:
                # Save results in List in case the dataset ended within a Dunkelflaute
                duration_list.append(hour_counter)
                hour_counter=0

            
            # check if list is empty (when no DFs could be detected)
            if not dunkelflaute_hours:
                dunkelflaute_hours.append(np.nan)
                
            df_hours_coun_list.append(pd.DataFrame(data=dunkelflaute_hours, index=np.arange(len(dunkelflaute_hours)), columns=['Date']))
            df_events_coun_list.append(pd.DataFrame(data=np.array([startdate_list,duration_list]).T, index=np.arange(len(startdate_list)), columns=['Startdate','Duration']))
        df_hours_scen_list.append(pd.concat(df_hours_coun_list, keys=zones_szon, names=['Countries']))
        df_events_scen_list.append(pd.concat(df_events_coun_list, keys=zones_szon, names=['Countries']))
    df_hours  = pd.concat(df_hours_scen_list,  keys=scenarios, names=['Scenario'])
    df_events = pd.concat(df_events_scen_list, keys=scenarios, names=['Scenario'])
    
    return df_hours, df_events







def find_successive_events_with_leap_years(df_input, time_col='Date'):
    """
    Identify successive hours in a dataframe of timestamps, handling leap year transitions, 
    and compute the duration of each continuous event period.

    Successive hours are defined as timestamps that occur exactly 1 hour apart.
    Leap year transitions (e.g., from 'Feb 28' to 'Mar 1' in a leap year) are
    treated as normal day-to-day transitions.

    Args:
        df (pd.DataFrame): A pandas dataframe with datetime index or a column of datetime values.
        time_col (str): The name of the datetime column (if not the index). Defaults to 'Date'.

    Returns:
        pd.DataFrame: A dataframe with two columns:
            - 'Startdate': The start time of each continuous period of successive hours.
            - 'Duration': The number of successive hours in each period.
    
    Example:
        >>> df = pd.DataFrame({
                'Date': ['2024-02-28 23:00:00', '2024-03-01 00:00:00', '2024-03-01 01:00:00'],
                'DE00': [True, True, True]
            })
        >>> find_successive_events_with_leap_years(df, 'Date')
        Startdate              Duration
        0 2024-02-28 23:00:00  3
    
    Notes:
        - The function checks for transitions between 'Feb 28' and 'Mar 1' in leap years
          and adjusts these differences to be treated as 1-hour transitions.
        - Assumes the input dataframe is sorted by the datetime column.

    """
    df = df_input.copy()

    # Convert the time_col to datetime if it's not the index
    if time_col != df.index.name:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

    # Calculate the time difference between consecutive rows
    df['time_diff'] = df.index.to_series().diff()

    # Handle leap year transitions (from Feb 28 to Mar 1)
    def adjust_leap_year_diff(df):
        for i in range(1, len(df)):
            current_date = df.index[i]
            prev_date = df.index[i - 1]

            # Check if the transition is from Feb 28 to Mar 1 in a leap year
            if (prev_date.month == 2 and prev_date.day == 28 and 
                current_date.month == 3 and current_date.day == 1 and 
                prev_date.is_leap_year):
                # Correct the time difference for leap year transition
                df.loc[current_date, 'time_diff'] = pd.Timedelta(hours=1)

        return df

    # Apply leap year adjustment
    df = adjust_leap_year_diff(df)

    # Define successive periods (1 hour difference)
    df['group'] = (df['time_diff'] != pd.Timedelta(hours=1)).cumsum()

    # Group by the 'group' column
    grouped = df.groupby('group')

    # Create a result dataframe with start date and duration for each group
    result = pd.DataFrame({
        'Startdate': grouped.apply(lambda x: x.index.min(), include_groups=False),  # Start date is the minimum date in each group
        'Duration': grouped.size()  # Duration is the number of rows in each group
    }).reset_index(drop=True)

    return result






def detect_DF_Li21(df_CF_h, zone, thresh, wrt_mean=False):
    """

    Parameters
    ----------

    df_CF_h: DataFrame
        Multi-index DataFrame of hourly capacity factor

        Example:
                                                    DE00
        Technology Scenario Date                         
        SPV        HIST     1982-01-01 00:00:00  0.000000
                            1982-01-01 01:00:00  0.000000
                            1982-01-01 02:00:00  0.000000
                            1982-01-01 03:00:00  0.000000
                            1982-01-01 04:00:00  0.000000
        ...                                           ...
        WON        HIST     2016-12-31 19:00:00  0.297262
                            2016-12-31 20:00:00  0.301915
                            2016-12-31 21:00:00  0.303360
                            2016-12-31 22:00:00  0.290909
                            2016-12-31 23:00:00  0.298991

    zone: str
        Zone name (e.g. 'DE00')

    thresh: str
        Threshold value for the capacity factor of all 3 technologies (SPV, WON, WOF)

    wrt_mean: bool (defalut: False)
        If False, the same threshold is used for all technologies. This method is used in Li et al. (2021).
        If True, the threshold is a share of the mean capacity factor.
        Kittel & Schill (2024) argue that this method gives a better relative weight to each technologies.

    Return
    ------

    df_DF: DataFrame
        Startdate: beginning of CF < threshold. Duration of the event (in hours)

                    Startdate  Duration
        0 1982-02-24 03:00:00         4
        1 1982-02-24 17:00:00         4
        2 1982-03-22 00:00:00         5
        3 1982-05-21 19:00:00         3
        4 1982-10-30 16:00:00         2
        5 2024-02-28 23:00:00         2
    """
    if wrt_mean:
        below_thresh = (df_CF_h.loc[('SPV', 'HIST')] < (df_CF_h.loc[('SPV', 'HIST')].mean() * thresh))  \
                    * (df_CF_h.loc[('WON', 'HIST')] < (df_CF_h.loc[('WON', 'HIST')].mean() * thresh))  \
                    * (df_CF_h.loc[('WOF', 'HIST')] < (df_CF_h.loc[('WOF', 'HIST')].mean() * thresh))
    else:
        below_thresh = (df_CF_h.loc[('SPV', 'HIST')] < thresh) \
                     * (df_CF_h.loc[('WON', 'HIST')] < thresh) \
                     * (df_CF_h.loc[('WOF', 'HIST')] < thresh)
    # only hours < threshold 
    hours_below_thresh = below_thresh[below_thresh[zone]]

    # group these hours into a 
    df_DF = find_successive_events_with_leap_years(hours_below_thresh)

    return df_DF









def mask_Li21(df_24h_events, zone, start_date='1982-01-01', end_date='2016-12-31'):
    """
    Mask of days with a dunkelflaute, based on the definition of Li et al (2021).
    Dunkelflaute: if more than 24 consecutive hours with CF < threhold for the 3 technologies (SPV, WON, WOF).
    We label a day as "dunkelflaute" with value 1 if CF < threshold for at least 12h.

    Parameters
    ----------

    df_DF: DataFrame
        Startdate: beginning of CF < threshold. Duration of the event (in hours)
        
        Example:
                    Startdate  Duration
        0 1982-02-24 03:00:00         4
        1 1982-02-24 17:00:00         4
        2 1982-03-22 00:00:00         5
        3 1982-05-21 19:00:00         3
        4 1982-10-30 16:00:00         2
        5 2024-02-28 23:00:00         2
    
    start_date: date in format 'yyyy-mm-dd'

    end_date: date in format 'yyyy-mm-dd'

    Returns
    -------

    df_mask_DF: DataFrame
        Days in a dunkelflaute event (=1) or not (=0)

        Example:
                    DE00
        1982-01-01   0.0
        1982-01-02   1.0
        ...          ...
        2016-12-31   0.0
    """


    # Build a Dataframe with value 1 if a dunkelflaute (DF) occurs on that date, 0 otherwize
    date_range = pd.date_range(start=start_date, end=end_date)
    # Remove February 29th from the date range
    date_range = date_range[~((date_range.month == 2) & (date_range.day == 29))]
    df_mask_DF = pd.DataFrame(np.zeros(len(date_range)), index=date_range, columns=[zone])

    # loop over all the events lasting more than 24h
    for startdate, total_duration in zip(df_24h_events['Startdate'], df_24h_events['Duration']):
        duration = total_duration
        # Get the index of the starting day
        # Using the 'date_range' index of df_mask_DF -> no need to account for Feb 29th
        DF_idx = df_mask_DF.index.get_loc(startdate.strftime('%Y-%m-%d'))
        
        if startdate.hour <= 12:
            # if during the first day there is more than 12h below the threshold, label this day as dunkelflaute 
            df_mask_DF.iloc[DF_idx] = 1

        duration -= (24 - startdate.hour)

        DF_idx += 1

        while duration >= 12 :
            # if in the following day there is more than 12h below the threshold, label this day as dunkelflaute 
            df_mask_DF.iloc[DF_idx] = 1

            duration -= 24
            DF_idx += 1
        
    return df_mask_DF







def lin_reg(x, y):
    ''' 
    Performs a linear regression (ignoring nans).
    data = one column of a pandas dataframe
    returns a tuple containing:
    dataframe with y=slope*x+intercept, intercept, slope, r-value, 
    p-value of Wald test (t-distrubtion, null hypothesis that slope=0),
    trend-size (last value - first value of y)
    '''
    arrx = np.asarray(x, dtype=float)
    arry = np.asarray(y, dtype=float)
    # Preparing the data (masking all non-nans)
    mask = ~np.isnan(arrx) & ~np.isnan(arry)
    # Linear regression and calculate y=intercept+slope*x
    reg = stats.linregress(arrx[mask], arry[mask])
    y = reg.intercept+reg.slope*arrx[mask]
    regdf = pd.DataFrame(data=y, index=arrx[mask], columns=['y'])
    regtrend = y[-1]-y[0]
    
    return regdf, reg.intercept, reg.slope, reg.rvalue, reg.pvalue, regtrend









def assign_SZON(input_pecd_zones, level):
    '''
    Assign to each PEON/PEOF zone a SZON zone.
    Goal: aggregate production at SZON level from data only defined at PEON level
    Do not mix PEON zones and PEOF zones!

    Parameters
    ----------
    input_pecd_zones : list
        list of PEON or PEOF zones

    level : str
        'PEON' or 'PEOF'
    
    Returns
    -------
    output_szon_zones : list
        list of SZON zones
        '''
        
    peon_zones = ['AL00','AT01','AT02','AT03','BA00','BE01','BE02','BE03','BG01',
                    'BG02','CH00','CY00','CZ01','CZ02','DE01','DE02','DE03','DE04',
                    'DE05','DE06','DE07','DKE1','DKW1','DZ01','DZ02','DZ03','EE00',
                    'EG00','ES01','ES02','ES03','ES04','ES05','ES06','ES07','ES08',
                    'ES09','ES10','ES11','ES12','FI01','FI02','FR01','FR02','FR03',
                    'FR04','FR05','FR06','FR07','FR08','FR09','FR10','FR11','FR12',
                    'FR13','FR14','FR15','GR01','GR02','GR03','HR01','HR02','HU01',
                    'HU02','HU03','IE00','IL00','IS00','ITCA','ITCN','ITCS','ITN1',
                    'ITS1','ITSA','ITSI','JO00','LB00','LT00','LU00','LV00','LY01',
                    'LY02','LY03','MA01','MA02','MA03','MA04','MA05','MA06','MD00',
                    'ME00','MK00','MT00','NL01','NL02','NL03','NL04','NOM1','NON1',
                    'NOS1','NOS2','NOS3','PL01','PL02','PL03','PL04','PL05','PS00',
                    'PT01','PT02','RO01','RO02','RO03','RS01','SE01','SE02','SE03',
                    'SE04','SI00','SK00','SY00','TN01','TN02','TN03','TN04','TN05',
                    'TN06','TN07','TN08','TR01','TR02','TR03','TR04','TR05','TR06',
                    'TR07','TR08','TR09','TR10','TR11','TR12','TR13','TR14','TR15',
                    'UA01','UA02','UK01','UK02','UK03','UK04','UK05','UKNI','XK00']

    peof_zones = ['AL00_OFF','BA00_OFF','BE01_OFF','BG01_OFF','CY00_OFF','DBOT_OFF',
                    'DBUK_OFF','DE011_OFF','DE012_OFF','DE013_OFF','DE02_OFF','DKBI_OFF',
                    'DKE1_OFF','DKKF_OFF','DKW11_OFF','DKW12_OFF','DZ00_OFF','EE00_OFF',
                    'EG00_OFF','ES01_OFF','ES02_OFF','ES04_OFF','ES06_OFF','ES09_OFF',
                    'ES10_OFF','ES11_OFF','FI01_OFF','FI02_OFF','FR01_OFF','FR02_OFF',
                    'FR03_OFF','FR04_OFF','FR08_OFF','FR09_OFF','FR13_OFF','GR01_OFF',
                    'GR02_OFF','GR03_OFF','HR01_OFF','IE00_OFF','IL00_OFF','IS00_OFF',
                    'ITCA_OFF','ITCN_OFF','ITCS_OFF','ITN1_OFF','ITS1_OFF','ITSA_OFF',
                    'ITSI_OFF','JO00_OFF','LB00_OFF','LT00_OFF','LV00_OFF','LY01_OFF',
                    'LY02_OFF','MA01_OFF','MA02_OFF','MA04_OFF','MA05_OFF','MA06_OFF',
                    'ME00_OFF','MT00_OFF','NL011_OFF','NL012_OFF','NL031_OFF','NL032_OFF',
                    'NL033_OFF','NOM1_OFF','NON1_OFF','NOS21_OFF','NOS22_OFF','NOS3_OFF',
                    'PL04_OFF','PL05_OFF','PS00_OFF','PT01_OFF','PT02_OFF','RO03_OFF',
                    'SE01_OFF','SE02_OFF','SE03_OFF','SE04_OFF','SI00_OFF','SY00_OFF',
                    'TN01_OFF','TN02_OFF','TN04_OFF','TN06_OFF','TN07_OFF','TR01_OFF',
                    'TR02_OFF','TR03_OFF','TR04_OFF','TR06_OFF','TR07_OFF','TR10_OFF',
                    'TR11_OFF','UA02_OFF','UK011_OFF','UK012_OFF','UK02_OFF','UK031_OFF',
                    'UK032_OFF','UK041_OFF','UK042_OFF','UK043_OFF','UK051_OFF',
                    'UK052_OFF','UK053_OFF','UKNI_OFF']

    szon_zones = ['AL00','AT00','BA00','BE00','BG00','CH00','CY00','CZ00','DE00',
                    'DKE1','DKW1','DZ00','EE00','EG00','ES00','FI00','FR00','FR15',
                    'GR00','GR03','HR00','HU00','IE00','IL00','IS00','ITCA','ITCN',
                    'ITCS','ITN1','ITS1','ITSA','ITSI','JO00','LB00','LT00','LU00',
                    'LV00','LY00','MA00','MD00','ME00','MK00','MT00','NL00','NOM1',
                    'NON1','NOS0','PL00','PS00','PT00','RO00','RS00','SE01','SE02',
                    'SE03','SE04','SI00','SK00','SY00','TN00','TR00','UA01','UA02',
                    'UK00','UKNI']

    output_szon_zones = []
    peon_is_szon = []

    # List all PEON zones that are also a SZON zone
    for peon_zone in peon_zones:
        if peon_zone in szon_zones:
            peon_is_szon.append(peon_zone)

    if level == 'PEON':

        for peon_zone in input_pecd_zones:
            if peon_zone in szon_zones:
                # If PEON = SZON, then PEON -> SZON
                output_szon_zones.append(peon_zone)
            else:
                # Get the SZON zone of the country, and remove the SZON zones that are also PEON zone.
                # Only one SZON should remain
                single_other_szon = set(get_zones([peon_zone[:2]], 'SZON')) - set(peon_is_szon)
                output_szon_zones.append(single_other_szon.pop())

    elif level == 'PEOF':

        for peof_zone in input_pecd_zones:

            peof_zone_cut = peof_zone[:-4]

            if peof_zone_cut in peon_zones:
                # if PEOF without "_OFF" is a PEON name, then do the same as for PEON

                if peof_zone_cut in szon_zones:
                    # If PEON = SZON, then PEON -> SZON
                    output_szon_zones.append(peof_zone_cut)
                else:
                    # Get the SZON zone of the country, and remove the SZON zones that are also PEON zone.
                    # Only one SZON should remain
                    single_other_szon = set(get_zones([peof_zone_cut[:2]], 'SZON')) - set(peon_is_szon)
                    output_szon_zones.append(single_other_szon.pop())

            else:
                # If PEOF without "_OFF" is is not a PEON name, then it is a special case handled manually
                if peof_zone in ['DE011_OFF','DE012_OFF','DE013_OFF']:
                    output_szon_zones.append('DE00')
                elif peof_zone in ['DKBI_OFF','DKKF_OFF']:
                    # Assume DKBI_OFF and DKKF_OFF -> DKE1, but not sure of that choice. 
                    output_szon_zones.append('DKE1')
                elif peof_zone in ['DKW11_OFF','DKW12_OFF']:
                    output_szon_zones.append('DKW1')
                elif peof_zone in ['NL011_OFF','NL012_OFF','NL031_OFF','NL032_OFF','NL033_OFF']:
                    output_szon_zones.append('NL00')          
                elif peof_zone in ['NOS21_OFF','NOS22_OFF']:
                    output_szon_zones.append('NOS0')
                elif peof_zone in ['UK011_OFF','UK012_OFF','UK031_OFF','UK032_OFF','UK041_OFF','UK042_OFF','UK043_OFF','UK051_OFF','UK052_OFF','UK053_OFF']:
                    output_szon_zones.append('UK00')
                else:
                    print(f'WARNING: Could not assign a SZON to {peof_zone}.')

    else:
        print(f"ERROR: '{level}' should be 'PEON' or 'PEOF'.")
        

    # Check
    #for a, b in zip(input_pecd_zones, output_szon_zones):
    #    print(f'{a} -> {b}')

    return output_szon_zones










def pairwise_comparison(rankings):
    """
    Perform pairwise comparisons to determine how many times each candidate wins.
    """
    # Get a set of all items
    all_items = {item for ranking in rankings for item in ranking}
    
    # Initialize pairwise win counts
    pairwise_wins = {item: {other: 0 for other in all_items if other != item} for item in all_items}
    
    # Perform pairwise comparisons
    for item1, item2 in combinations(all_items, 2):
        # Count preferences for item1 vs. item2
        item1_wins = sum(ranking.index(item1) < ranking.index(item2) for ranking in rankings)
        item2_wins = len(rankings) - item1_wins
        
        # Update win counts
        if item1_wins > item2_wins:
            pairwise_wins[item1][item2] += 1
        elif item2_wins > item1_wins:
            pairwise_wins[item2][item1] += 1

    return pairwise_wins





def condorcet_ranking(pairwise_wins):
    """
    Generate a ranking based on pairwise wins.
    """
    # Calculate total wins for each candidate
    total_wins = {candidate: sum(wins > 0 for wins in opponents.values())
                  for candidate, opponents in pairwise_wins.items()}
    
    # Sort candidates by total wins (descending)
    ranking = sorted(total_wins.keys(), key=lambda x: -total_wins[x])
    return ranking, total_wins
