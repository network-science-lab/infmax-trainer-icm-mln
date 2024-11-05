# Inf. Max. with ML methods for Multilayer Networks

A repository to train and evaluate Influence Maximisation ML models for multilayer networks.

* Authors: Piotr Bródka, Michał Czuba, Adam Piróg, Mateusz Stolarski
* Affiliation: WUST, Wrocław, Lower Silesia, Poland

## Configuration of the runtime

First, initialise the submodule with utilities:

```bash
git submodule init && git submodule update
```

Then, initialise the enviornment:

```bash
conda env create -f env/conda.yaml
conda activate infmax-trainer-icm-mln
```

## Data

Dataset is stored in a separate repository bounded with this project as a git submodule. Thus, to
obtain it you have to pull the data from the DVC remote. In order to access it, please send a
request to get an access via e-mail (michal.czuba@pwr.edu.pl). Then, simply execute in a shell:
* `cd _data_set && dvc pull nsl_data_sources/raw/multi_layer_networks/**.dvc && cd ..`
* `cd _data_set && dvc pull nsl_data_sources/spreading_potentials/multi_layer_networks/**.dvc && cd ..`

## Structure of the repository
```
.
├── _configs                -> def. of the training configs
├── _data_set               -> evaluated networks
├── env                     -> a definition of the runtime environment
├── src                     -> a module with main implementation
│   |── datamodule          -> code for converting datasets into datamodule
│   |── dataset             -> implemented datasets for preparing HeteroData
│   |── hetero_data         -> an extension of HeteroData from torch_geometric
│   |── infmax_models       -> implemented ML models for Influence Maximisation
│   |── training            -> code related to training execution
│   │   ├── trainers        -> scripts to train models according to provided configs
│   │   ├── callbacks.py    -> allowed training callbacks defined in executed config
│   │   └── loggers.py      -> allowed training loggers defined in executed config 
│   |── utils               -> the logic for helpers across whole repository
│   └── wrapper             -> the wrappers for trainable models implemented in torch
├── trainers                -> scripts to train models according to provided configs
├── README.md          
└── run_experiments.py      -> main entrypoint to trigger the pipeline
```

## Running the pipeline

To run experiments execute: `run_experiments.py` and provide proper CLI arguments defined in
`_configs/hydra.yaml`, i.e. a name of the configuration file.

To select device on thorium please set up envitonment variable: `export CUDA_VISIBLE_DEVICES=2` and
then in the config file select list of devices as `[0]`.

### `neptune.ai` dashboard

The dashboard is here: https://app.neptune.ai/o/infmax/org/infmax-gnn/runs/table?viewId=standard-view.
Prior using it please fill in the `AUTH_KEY` in the `.env` file (as shown in `.env-example`).
