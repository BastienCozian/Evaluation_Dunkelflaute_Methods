# Evaluation and Comparison of Various Dunkelflaute Methods
Version 1.0

These are the scripts used for evaluating two methods of energy drought detection as part as a TenneT trainee assignment.
For questions, please refer to benjamin.biewald@tennet.eu


# Description

- `Dunkelflaute_function_library.py`

A collection of functions used for several things around Dunkelflaute detection.

- `DF_Data_Preprocessing.py`

Loads all sorts of data and preprocesses it into the unified DataFrame format.
Do not run the whole script in one go! Some parameters (especially target year "ty") need to be individually assigned before running individual cells.
For all PECD 3.1 specific data loading, ty must be 2033.
For all PECD 4.1 specific data loading, ty must be 2030, 2040 or 2050 (2040 is not used in my analysis).

- `DF_Validation_Li.py`
- `DF_Validation_Otero.py`

Both scripts are very similar. They load in the preprocessed data and validate the methods inspired by Li et al. 2021 and Otero et al. 2022 using ENS data.
Using an F-Score it is evaluated how good the detection works for a range of thresholds.
They save the results of the validation as well as validationg plots. These plots can be used to obtain a "best-fitting" threshold for further detection analysis.

- `DF_Detection_Li.py`
- `DF_Detection_Otero.py`

Both scripts are again similar. They load in the preprocessed data, detect energy droughts using the corresponding method and analyse the properties of detected droughts.
The analysis results are saved (namely a dataframe consiting of all individual points in time that are considered a drought, a datamask that masks drought times as 1 and a dataframe listing all drought events with corresponding properties: Startdate, duration, severity (original by Otero et al. 2022) and adapted severity (by me))

## Dependencies

Use ``conda env create -f environment.yml`` to install this project's dependencies and create a ``dunkelflaute_analysis`` conda-environment.