# Inf. Max. with ML methods for Multilayer Networks 

A repository to train and evaluate Influence Maximisation ML models for multilayer networks.

* Authors: Piotr Bródka, Michał Czuba, Adam Piróg, Mateusz Stolarski
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Configuration of the runtime

```bash
conda env create -f env/conda.yaml
conda activate infmax-simulator-icm-mln
```

## Data

Dataset is stored in a separate reository bounded with this project as a git submodule. Thus, to
obtain it, execute: `git submodule init` and `git submodule update`. Then, you have to pull the data
from the DVC remote. In order to access it, please sent a request to get  an access via  e-mail
(michal.czuba@pwr.edu.pl). Then, simply execute in a shell:
* `cd _data_set && dvc pull ns-data-sources/raw/multi_layer_networks/*.dvc && cd ..`
* `cd _data_set && dvc pull ns-data-sources/spreading_potentials/multi_layer_networks/*.dvc && cd ..`

## Structure of the repository
```
.
├── _configs                -> def. of the training configs
├── _data_set               -> evaluated networks
├── env                     -> a definition of the runtime environment
├── misc                    -> miscellaneous scripts helping in trainings
├── ss_models               -> implemented ML models for Influence Maximisation
├── trainers                -> scripts to train models according to provided configs
├── README.md          
└── run_experiments.py      -> main entrypoint to trigger the pipeline
```

## Running the pipeline

To run experiments execute: `run_experiments.py` and provide proper CLI arguments, i.e. a path to 
the configuration file. See examples in `_config/examples` for inspirations.
