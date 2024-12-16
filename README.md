# Code for 'Evaluation of Dunkelflaute event detection methods considering grid operators’ needs'

Scripts used for evaluating three methods of dunkelflaute detection.
See the paper 'Evaluation of Dunkelflaute event detection methods considering grid operators’ needs' for more details.

For questions, please contact bastien.cozian@rte-france.com or benjamin.biewald@tennet.eu.


## Description

```
.
├── .gitignore
├── README.md
├── environment.yml
├── src                                        <- Scripts and source code for this project 
│   ├── CREDIfunctions.py                      <- Functions developed for the method Stoop'23 (CREDI)
│   ├── Data_Preprocessing.py                  <- Preprocessing of PECDv3.1 data
│   ├── Dunkelflaute_function_library.py       <- Functions used for data analysis
│   ├── Main_Figures.py                        <- Figures of the main paper
│   └── SI_Figures.py                          <- Figures of the supporting information
└── old                                        <- Old scripts 
```

## Dependencies

Use ``conda env create -f environment.yml`` to install this project's dependencies and create the ``dunkelflaute_analysis`` conda-environment.