#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:21:57 2024

@author: Bastien Cozian

Adapted from L.P. Stoop
See: https://github.com/laurensstoop/CREDI
"""


# Load the dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import math
import datetime as dtime #from datetime import timedelta



# =============================================================================
# Definition of climatology based on the Modified Ordinal Day
# =============================================================================


def Modified_Ordinal_Hour(InputDataSet):
    """
    Compute the (modified) ordinal hour.
    
    Adapted from @author Laurens P. Stoop
    https://github.com/laurensstoop/CREDI

    Parameters
    ----------
    InputDataSet : Xarray DataSet
        Feb 29th is (should be?) removed from the timeseries.
        Note: The index is "Date" (not "time").
        Example:

        Dimensions:  (Date: 12775)
        Coordinates:
        * Date     (Date) datetime64[ns] 1982-01-01 1982-01-02 ... 2016-12-31
        Data variables:
            DE00     (Date) float64 1.277e+06 1.368e+06 ... 9e+05 5.759e+05
            NL00     (Date) float64 4.878e+05 4.295e+05 ... 3.333e+05 3.558e+05

    Returns
    -------
    OutputDataSet: Xarray DataArray
        Ordinal hour (from 0 to 24*365)
        Example:

        <xarray.DataArray 'ModifiedOrdinalHour' (Date: 12775)> Size: 102kB
        array([   0,   24,   48, ..., 8712, 8736, 8760], dtype=int64)
        Coordinates:
        * Date     (Date) dateDate64[ns] 102kB 1982-01-01 1982-01-02 ... 2016-12-31
               
    """
    
    """
    ordinal_hour = InputDataSet.Date.dt.dayofyear * 24 + InputDataSet.Date.dt.hour - 24
    ordinal_hour = ordinal_hour.rename('OrdinalHour')
    """
    # Problem: if we define ordinal hour as above, it is based on dayofyear, where dayofyear=31+29 is 
    # 29th of February on leap year 31th March on non-leap year. This implies that 31st December is
    # dayofyear=366 on leap years and 365 on non-leap years, even if 29th was removed from the Dateseries. 
    # Solution: modify dayofyear to correctly average the same calendar date when grouping by ordinal hours. 
    is_leap_year = xr.DataArray(InputDataSet.indexes['Date'].is_leap_year, coords=InputDataSet.coords)
    march_or_later = InputDataSet.Date.dt.month >= 3
    ordinal_hour = InputDataSet.Date.dt.dayofyear * 24 + InputDataSet.Date.dt.hour - 24
    modified_ordinal_hour = ordinal_hour - (is_leap_year & march_or_later) * 24            # Changed wrt L.P. Stoop's code
    modified_ordinal_hour = modified_ordinal_hour.rename('ModifiedOrdinalHour')
    
    # Now we return the output dataset that provides the anomaly 
    return modified_ordinal_hour


# we want a function of climatology in which the 29t of february is counted correctly
def Climatology_Hourly(InputDataSet):

    modified_ordinal_hour = Modified_Ordinal_Hour(InputDataSet)
    
    # we can use this new modified ordinal day definition to get the correct climatology
    OutputDataSet = InputDataSet.groupby(modified_ordinal_hour).mean('Date')
    
    return OutputDataSet


def Climatology_Hourly_Rolling(InputDataSet, RollingWindow=40):
    """
    Compute the climatology with an Hourly Rolling Window (HRW) [1].
    It is a physically-based method to compute the climatology which properly accounts for 
    relevant (hourly to annual) timescales in the electricity sector. 
    
    Adapted from @author Laurens P. Stoop
    https://github.com/laurensstoop/CREDI

    Parameters
    ----------
    InputDataSet : Xarray DataSet
        Hourly timeseries of energy variable (RES, DD, RL, CF, etc.). Contains one or multiple zones (e.g. DE00, NL00).
        Note: The index is "Date" (not "time").
        Example:

        Dimensions:  (Date: 12775)
        Coordinates:
        * Date     (Date) datetime64[ns] 1982-01-01 1982-01-02 ... 2016-12-31
        Data variables:
            DE00     (Date) float64 1.277e+06 1.368e+06 ... 9e+05 5.759e+05
            NL00     (Date) float64 4.878e+05 4.295e+05 ... 3.333e+05 3.558e+05

    RollingWindow : int
        Size of the rolling window (in DAYS)

    Returns
    -------
    OutputDataSet: Xarray DataSet
        Example:

        Dimensions:              (ModifiedOrdinalHour: 365)
        Coordinates:
        * ModifiedOrdinalHour  (ModifiedOrdinalHour) int64 0 24 48 ... 8736 8760
        Data variables:
            DE00                 (ModifiedOrdinalHour) float64 8.265e+05 ... 8.37...
            NL00                 (ModifiedOrdinalHour) float64 3.378e+05 ... 3.39
        
    modified_ordinal_hour: Xarray DataArray
            See function Modified_Ordinal_Hour.
            
    References
    ----------
    .. [1] Laurens P. Stoop et al., The Climatological Renewable Energy Deviation Index (Credi),
       Environ. Res. Lett. 19 (2024).
    """
    modified_ordinal_hour = Modified_Ordinal_Hour(InputDataSet)

    # Average the hours (e.g. 1pm) from ~ day - 20 days to + 20 days
    results = []
    for _, group in InputDataSet.groupby(InputDataSet.Date.dt.hour):
        results.append(group.rolling(Date=RollingWindow, center=True).mean())

    OutputDataSet = xr.merge(results)

    # On this smooth data we determine the climatology
    OutputDataSet = OutputDataSet.groupby(modified_ordinal_hour).mean('Date')

    # Now we return the output dataset that provides the anomaly 
    return OutputDataSet, modified_ordinal_hour


def Climatology_Hourly_Weekly_Rolling(InputDataSet, RollingWindow=9):
    """
    Compute the climatology with an Hourly Weekly Rolling Window (HWRW).
    It is an adaptation of the Hourly Rolling Window (HRW) [1]. 
    This method with HWRW averages the values of the same hour and day of the week
    (e.g. Monday 1st Jan 13:00, Monday 8th Jan 13:00, Monday 15th Jan 13:00 for a RollingWindow=3) 
    while the method with HRW averages the same hour the day with the same hour of the surrounding days 
    (e.g. Monday 1st Jan 13:00, Tuesday 2nd Jan 13:00, Wednesday 3rd Jan 13:00 for a RollingWindow=3).

    The use of HRW is physically-based to compute the climatology of energy variables which properly 
    accounts for relevant (hourly to annual) timescales.
    This HWRW methods additionaly account for the weekly cycle, which is relevant particularly for 
    electricity demand (and other variables based on demand such as residual load) which has a clear weekly cycle.
    
    Adapted by B. Cozian from Laurens P. Stoop
    https://github.com/laurensstoop/CREDI

    Parameters
    ----------
    InputDataSet : Xarray DataSet
        Hourly timeseries of energy variable (RES, DD, RL, CF, etc.). Contains one or multiple zones (e.g. DE00, NL00).
        Note: The index is "Date" (not "time").
        Example:

        Dimensions:  (Date: 12775)
        Coordinates:
        * Date     (Date) datetime64[ns] 1982-01-01 1982-01-02 ... 2016-12-31
        Data variables:
            DE00     (Date) float64 1.277e+06 1.368e+06 ... 9e+05 5.759e+05
            NL00     (Date) float64 4.878e+05 4.295e+05 ... 3.333e+05 3.558e+05

    RollingWindow : int
        Size of the rolling window (in WEEKS)
        If RollingWindow == 3, returns the mean of w-1, w, and w+1.
        If RollingWindow == 4, returns the mean of w-1, w-2, w, w+1.

    Returns
    -------
    OutputDataSet: Xarray DataSet
        Example:

        Dimensions:              (ModifiedOrdinalHour: 365)
        Coordinates:
        * ModifiedOrdinalHour  (ModifiedOrdinalHour) int64 0 24 48 ... 8736 8760
        Data variables:
            DE00                 (ModifiedOrdinalHour) float64 8.265e+05 ... 8.37...
            NL00                 (ModifiedOrdinalHour) float64 3.378e+05 ... 3.39
        
    modified_ordinal_hour: Xarray DataArray
            See function Modified_Ordinal_Hour.
            
    References
    ----------
    .. [1] Laurens P. Stoop et al., The Climatological Renewable Energy Deviation Index (Credi),
       Environ. Res. Lett. 19 (2024).
    """
    modified_ordinal_hour = Modified_Ordinal_Hour(InputDataSet)

    results = []
    df_hours = InputDataSet.Date.dt.hour.to_pandas()
    df_dayofweek = InputDataSet.Date.dt.dayofweek.to_pandas()
    # Average for values with the same hour and dayofweek
    # Need to use Pandas' .groupby() because Xarray's .groupby() does not allow to easily use two groups.
    for _, group in InputDataSet.to_pandas().groupby([df_hours, df_dayofweek]):
        results.append(group.to_xarray().rolling(Date=RollingWindow, center=True).mean())

    OutputDataSet = xr.merge(results)

    # On this smooth data we determine the climatology
    OutputDataSet = OutputDataSet.groupby(modified_ordinal_hour).mean('Date')

    # Now we return the output dataset that provides the anomaly 
    return OutputDataSet, modified_ordinal_hour


# =============================================================================
# CREDI
# =============================================================================



def get_CREDI_events(df_data, zone, extreme_is_high=True, PERIOD_length_days=1, PERIOD_cluster_days=1,
                     start_date='1982-01-01', end_date='2016-12-31', climatology='HRW'):
    """
    Compute the Climatological Renewable Energy Deviation Index (CREDI) [1]. 
    CREDI is by default computed based on Hourly Rolling Window (HRW, see function `Climatology_Hourly_Rolling`)
    but can be computed based on Hourly Weekly Rolling Window (HWRW, not fully implemented yet, 
    see function `Climatology_Hourly_Weekly_Rolling`).

    TODO: add the possibility to use "start_date" and "end_date". Apparently I first need to correct an issue in the code of Ben, 
    but I can't remember what...
    /!\ WARNING /!\ -> Currently, we first compute the climatology based on all values to have a smooth definition 
    (the more data, the smoother) but this need to be accounted for when e.g. computing the percentile valye.
    Moreover, this may not be appropriate when looking at at projection data when we want only e.g. the 2015-2045 period.

    Parameters
    ----------
    df_data : DataFrame
        Input energy variable (Residual load, demand, or wind and solar production).

        Example:
                                    DE00          NL00
        Date                                           
        1982-01-01 00:00:00  20618.535429  18246.572745
        1982-01-01 01:00:00  21718.368473  18091.074620

    zone : str
        Spatial zone, which generally correspond to a SZON (bidding zone) 
        possibly aggregated from PZON zones (see DF_Data_Preprocessing.py).
        Example: 'DE00', 'NL00'

    PERIOD_length_days: int
        Duration of CREDI events.

    PERIOD_cluster_days: int
        Period used to cluster events, i.e. to differentiate/partly exclude overlapping events.
        A 20-25% overlap is accepted (e.g. PERIOD_length_days=4 and PERIOD_cluster_days=3).

    start_date: date in format 'yyyy-mm-dd'

    end_date: date in format 'yyyy-mm-dd'

    climatology: str
        Method to compute the climatology. 
        3 choices: HRW (Hourly Rolling Window), HWRW (Hourly Weekly Rolling Window), and empirical.
        The HWRW is not fully implemented yet.

    Returns
    -------

    ds_CREDI_event : DataFrame
        CREDI value at the end of the event. It is the cummulative sum of anomalies (wrt climatology) 
        over the duration of the event.

        Exemple:
                    ModifiedOrdinalHour          DE00           NL00
        Date                                                        
        1982-01-01                    0 -1.536445e+02    4171.817640
        1982-01-02                   24  4.845000e+05  155744.196369
    
    event_dates : list of Timestamp
        Date of CREDI events ranked in decreasing (resp. increasing) order of the CREDI value 
        for residual load and demand (resp. renewable energy production).
        Note that a 3-day CREDI event at date 1982-01-04 is actually the average 
        from 1982-01-01 T00:00:00 to 1982-01-03 T23:00:00.

        Example: 
        [Timestamp('2015-01-21 00:00:00'),
         Timestamp('1997-12-23 00:00:00'),
         ...
        ]

    event_values : list
        CREDI values  ranked in decreasing (resp. increasing) order for residual load and demand 
        (resp. renewable energy production).

            
    References
    ----------
    .. [1] Laurens P. Stoop et al., The Climatological Renewable Energy Deviation Index (Credi),
       Environ. Res. Lett. 19 (2024).
    """
    
    ## Length of the period to consider for CREDI assessment (in hours)
    # add 1 to get indexes that make sense 
    PERIOD_length = PERIOD_length_days * 24 + 1 # e.g. 193 for 8 days

    # Sampling of the period (in hours)
    PERIOD_stride = 24

    # Set the data + hourly climatology
    ds_data = df_data.to_xarray()

    if climatology == 'HRW':
        ds_clim_HRW, MOH = Climatology_Hourly_Rolling(ds_data, RollingWindow=40)
        ds_anom = ds_data.groupby(MOH) - ds_clim_HRW
    elif climatology == 'HWRW':
        ds_clim_HWRW, MOH = Climatology_Hourly_Weekly_Rolling(ds_data, RollingWindow=9)
        ds_anom = ds_data.groupby(MOH) - ds_clim_HWRW
        print("The HWRW method is not properly implemented yet.")
    elif climatology == 'empirical':
        ds_anom = ds_data - Climatology_Hourly(ds_data)
    else:
        raise KeyError("Choose a method to compute climatology among 'HRW', 'HWRW' and 'empirical.")

    # Compute the cummulative sum over all hours
    ds_CREDI = ds_anom.rolling(Date=PERIOD_length).construct(Date="event_hour", stride=PERIOD_stride).cumsum(dim='event_hour')

    # Generate dataset of last event hours to get one T-day average per day
    ds_CREDI_event = ds_CREDI.sel(event_hour=PERIOD_length-1).to_pandas()

    # Select the smallest 100K events (technically all are listed as we do not drop data)
    if extreme_is_high:
        # if extreme events have high value (e.g. demand, residual load)
        ds_CREDI_sort_event = ds_CREDI_event.nlargest(len(ds_CREDI_event), zone, keep='all')
    else:
        # if extreme events have low value (e.g. renewable production)
        ds_CREDI_sort_event = ds_CREDI_event.nsmallest(len(ds_CREDI_event), zone, keep='all')

    # Filter the full dataset to only get the non-overlapping events
    # Drop all events that are within "PERIOD_cluster_days" 
    event_dates = []
    event_values = []
    while len(ds_CREDI_sort_event) > 0:
        
        # add the events to the list
        event_dates.append(ds_CREDI_sort_event.iloc[0].name)
        event_values.append(ds_CREDI_sort_event[zone].iloc[0])
        
        # now filter this event and the overlapping ones
        ds_CREDI_sort_event = ds_CREDI_sort_event.drop(ds_CREDI_sort_event.loc[(abs(ds_CREDI_sort_event.index - ds_CREDI_sort_event.index[0]) < dtime.timedelta(PERIOD_cluster_days))].index)


    return ds_CREDI_event, event_dates, event_values



def get_f_score_CREDI(ens_mask, df_mask, zone, PERIOD_length_days, beta=1):
    """
    Compute the F-score for CREDI events.
    We consider that a CREDI event is successfully detected an ENS (true positive) if at least one ENS occur during 
    the `PERIOD_length_days`-day-long CREDI event.

    The date of the CREDI events in df_mask are shifted by one day. 
    Indeed, a 3-day CREDI event indexed at day 1982-01-04 is actually the average from 1982-01-01 T00:00:00 to 1982-01-03 T23:00:00.


    TODO: Make sure that 2 successive ENS in the same event are counted as one.

    TODO: make a proper choice to use a single zone or multiple zone. Currently no choice is made.

    TODO: Check that I correctly take only one ENS if two ENS occur within 2 days on the same detected Dunkelflaute event

    TODO: Not sure of the computation of true_negative, need to think more. Do not affect the computation of F-score. 
    Use len(same_index) or number of DF events ?

    TODO: implement the case where PERIOD_length_days != PERIOD_cluster_days

    Parameters
    ----------
    ens_mask : DataFrame
        Value = 2 if ENS, 0 otherwize.

        Example:
                    DE00  NL00
        Date                  
        1982-01-01     0     0
        1982-01-02     2     0
    
    df_mask : DataFrame
        Value = 1 if detected energy drought, 0 otherwize.
    
        Example:
                    DE00
        Date            
        1982-01-01     0
        1982-01-02     1

    zone : str
        e.g. "DE00", "NL00"

    PERIOD_length_days: int
        Duration of CREDI events.

    beta : float
        Relative weight of precision and recall.

    Returns
    -------

    result_df : DataFrame
        DataFrame containing the results (F-score, true positive, false negative, 
        false positive, true negative, precision, recall).
    
        Example:
                DE00
        F      0.104628
        TP    52.000000
        FN     5.000000
        FP   885.000000
        TN  5246.000000
        PR     0.055496
        RE     0.912281

    """
    
    # Take the same calendar days for the two DataFrame
    same_index = df_mask.index.intersection(ens_mask.index)

    ens_mask = ens_mask.loc[same_index]
    df_mask = df_mask.loc[same_index].shift(-1) # shift CREDI events by one day.

    """
    Piece of code that may be useful later.

    # TODO: ensure that it works for Demand (above threshold) and REP (below threshold)
    nb_event = np.sum(np.asarray(event_values) > threshold)

    # Date of the Dunkelflaute event (above/below threshold corresponding to percentile p)
    # Take only the date on the same calendar period
    # Must convert to "list" to get the time fomat "Timestamp" which allows to use "timedelta".
    DF_event_dates = np.intersect1d(list(same_index), event_dates[:nb_event])

    # Dates of ENS
    ENS_event_dates = np.intersect1d(list(same_index), list(ens_mask[zone].loc[ens_mask[zone] > 0].index))

    # Not useful here
    #detection_mask_naive = ens_mask[[zone]].loc[same_index] * 0
    """

    # Naive F-score computation

    true_positive  = 0
    false_negative = 0
    false_positive = 0
    true_negative  = 0

    id = len(same_index) - 1
    while id >= 0:
        #print(df_mask.loc[id])
        #print(ens_mask[[zone]].loc[id])

        if (df_mask.iloc[id].values == 1) and np.any([(ens_mask[[zone]].iloc[id - shift_day].values == 2) for shift_day in range(PERIOD_length_days)]):

            # TRUE POSITIVE
            true_positive += 1
            # Jump over other possible ENS that occur during the detected Dunkelflaute event.
            # TODO: Possibility to sum their values later to compute the severity
            id -= PERIOD_length_days

        elif ens_mask[[zone]].iloc[id].values == 2:
            # FALSE NEGATIVE
            false_negative += 1
            id -= 1

        elif df_mask.iloc[id].values == 1:
            # FALSE POSITIVE
            false_positive += 1
            id -= 1
        
        else:
            id -= 1
 
    true_negative = len(same_index) - true_positive - false_negative - false_positive

    f = ((1+beta**2)*true_positive) / ((1+beta**2)*true_positive + beta**2 * false_negative + false_positive)
    # f is between 0 and 1, 0= precision or recall are zero, 1=perfect precision and recall
    # precision = ratio of true positives over all positives (how many events are wrongly detected?)
    # recall = ratio of true positives over true positive+false negatives (how many events are missed out?)
    # np.divide to properly deal with divide by zero
    precision = np.divide(true_positive, true_positive + false_positive)
    recall = np.divide(true_positive, true_positive + false_negative)
    
    data = [f, true_positive, false_negative, false_positive, true_negative, precision, recall]
    result_df = pd.DataFrame(index=['F','TP','FN','FP','TN','PR','RE'], columns=[zone], data=data)
    
    return result_df