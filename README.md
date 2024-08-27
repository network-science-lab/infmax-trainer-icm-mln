# Inf. Max. with ML methods for Multilayer Networks

A repository to train and evaluate Influence Maximisation ML models for multilayer networks.

* Authors: Piotr Bródka, Michał Czuba, Adam Piróg, Mateusz Stolarski
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Configuration of the runtime

```bash
conda env create -f env/conda.yaml
conda activate infmax-trainer-icm-mln
```

`multi_node2vec` is an external codebase that can be executed with a deprecated Python version. Hence,
it has been contenerised. Before using it, please build the docker image:

```bash
cd infmax_models/multi_node2vec
docker build -t multi-node2vec --platform=linux/amd64 .
cd ../..
```

## Data

Dataset is stored in a separate reository bounded with this project as a git submodule. Thus, to
obtain it, execute: `git submodule init` and `git submodule update`. Then, you have to pull the data
from the DVC remote. In order to access it, please sent a request to get  an access via e-mail
(michal.czuba@pwr.edu.pl). Then, simply execute in a shell:
* `cd _data_set && dvc pull ns-data-sources/raw/multi_layer_networks/*.dvc && cd ..`
* `cd _data_set && dvc pull ns-data-sources/spreading_potentials/multi_layer_networks/*.dvc && cd ..`

## Structure of the repository
```
.
├── _configs                -> def. of the training configs
├── _data_set               -> evaluated networks
├── env                     -> a definition of the runtime environment
├── infmax_models           -> implemented ML models for Influence Maximisation
├── misc                    -> miscellaneous scripts helping in trainings
├── trainers                -> scripts to train models according to provided configs
├── README.md          
└── run_experiments.py      -> main entrypoint to trigger the pipeline
```

## Running the pipeline

To run experiments execute: `run_experiments.py` and provide proper CLI arguments, i.e. a path to 
the configuration file. See examples in `_config/examples` for inspirations.
