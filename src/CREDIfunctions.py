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


#%%
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
