# OTAPI: Optimizing Treatment Allocation in the Presence of Interference
This repository provides the code for the paper *"OTAPI: Optimizing Treatment Allocation in the Presence of Interference"*.

The structure of the code is as follows:
```
OTAPI/
|_ data/
  |_ semi_synthetic/                   
    |_ BC/
    |_ Flickr/
|_ scripts/
  |_ BC_experiment.py                    
  |_ Flickr_experiment.py
  |_ spillover0_experiment.py        #Experiment with spillover effect == 0    
  |_ spillover0.1_experiment.py      # "" == 0.1
  |_ spillover0.3_experiment.py      # "" == 0.3
  |_ spillover0.5_experiment.py      # "" == 0.5
  |_ spillover0.7_experiment.py      # "" == 0.7
  |_ spillover_figures.py            # Generate figures after all spillover experiments 
  |_ watts_strogatz.py
|_ src/
  |_ data/
    |_ data_generator.py                 # Code to generate synthetic and semi-synthetic data
    |_ datatools.py                 
  |_ methods/
    |_ allocation/         # Allocation optimization
      |_ utils/
        |_ CELF.py                    # Code for CELF algorithm
        |_ Genetic_algorithm.py       # Code for genetic algorithms
        |_ allocation_utils.py        # Utils for the allocations
      |_ extra_allocations.py       # Code to run extra experiments for the appendix
      |_ get_allocations.py         # Code to run and save the allocations
    |_ causal_models/         # Train relational causal estimator
      |_ baselineModels.py          # Baseline models (e.g., TARNet and CFR)
      |_ model_training.py          # Model training and evaluation
      |_ layers.py                  
      |_ model.py                   # NetEst model
      |_ modules.py                 
      |_ model_tuning.py            # Train best model
    |_ utils/
      |_ plotting.py/      # Generate figures
      |_ utils.py/      
```

## Installation.
The ```requirements.txt``` provides the necessary packages.
All code was written for ```python 3.10.13```.

## Usage
Download the data for the BC and Flickr dataset from [Google Drive](https://drive.google.com/drive/folders/16BDvaDuS19Tywji2xddWqV9l1GWJ6Bq1?usp=sharing). The original data comes from [this repo](https://github.com/rguo12/network-deconfounder-wsdm20).
We use the same data as ([Jiang & Sun (2022)](https://github.com/songjiang0909/Causal-Inference-on-Networked-Data). 

Put the data in the ```data/semi_synthetic/``` folder. Now, the results from the paper can be reproduced by setting the ```DIR``` variable to your directory and running the appropriate script. To reproduce Figures 5a and 5b, the results from all the ```spillover*_experiment.py``` are needed
before running the ```spillover_figures.py``` file to generate the figures.

## Acknowledgements
Our code builds upon the code from [Jiang & Sun (2022)](https://github.com/songjiang0909/Causal-Inference-on-Networked-Data). 

Jiang, S. & Sun, Y. (2022). Estimating causal effects on networked observational data via representation learning. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, (pp. 852–861).
