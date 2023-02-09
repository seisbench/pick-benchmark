# A benchmark of seismic picking and detection algorithms

*Note: This repository is not actively maintained. Nonetheless, there might be occasional updated. For the reference version used for the benchmark paper listed at the bottom, check this [tagged version](https://github.com/seisbench/pick-benchmark/releases/tag/jgr_revision).*

This repository contains a benchmark of seismic picking and detection algorithms, with a particular focus on deep learning methods.
The benchmark is built on top [SeisBench - A toolbox for machine learning in seismology](https://github.com/seisbench/seisbench) and [pytorch lightning](https://www.pytorchlightning.ai/).
It is intended to be modular and easily extensible.
If you have questions or would like to contribute to the repository, fell free to [raise an issue](https://github.com/seisbench/pick-benchmark/issues). 

# Setup

We recommend to set up the benchmark within a virtual environment, for example using [conda](https://docs.conda.io/en/latest/).
To set up the benchmark code, use the following steps:
1. Clone the repository
1. Switch to the root directory of the repository
1. Run `pip install -r requirements.txt`

You should now be able to run the benchmark code.
Note that this does not download the datasets.
Datasets will be downloaded automatically on their first usage through SeisBench.
Alternatively, you can manually trigger the download through SeisBench by instantiating the dataset classes.
For details on the dataset, see the [SeisBench documentation](https://seisbench.readthedocs.io/en/latest/pages/benchmark_datasets.html).

# Basic structure

The benchmark is split into three parts: training, evaluation and collection and visualization of results.
Each of these is explained below.
All code is located in the `benchmark` folder.
For further details also consult the in code documentation.

## Training

Training is handled by the `train.py` script.
Each training is defined by a config file in json format.
Several examples of config files are contained in the config folder.
Config files should follow the naming convetion `[dataset]_[model].json`. 

The config file specifies the model, the data, the model arguments and the training arguments.
`model` needs to refer to a model defined in `models.py`.
Details can be found below in the section "Adding a model".
`data` can be either the name of a dataset built into SeisBench or the path to a dataset in SeisBench format.
`model_args` are passed to the constructor of the model class.
`trainer_args` are passed to the [pytorch lightning trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html).

Training writes several log files (for a config `[config].json`:
- Checkpoints (incl. model weights) are written to `weights/[config]_[config]`. By default, only the model with the best validation score is kept. This setting can be overwritten using the `trainer_args`.
- Logs in csv format and the config as yaml are written to `weights/[config]`.
- Tensorboard logs are written to `tb_logs/config`.

In addition, `train.py` offers a command line parameter to overwrite the learning rate.
If provided, the learning rate in the config file will be ignored and instead the learning rate from the command line will be used.
The new learning rate will be appended to the paths of the log files.

## Evaluation

To evaluate a trained model, the file `eval.py` is used.
`eval.py` takes as input a folder with weights from a training (the `weights/[config]` folder, not `weights/[config]_[config]`) and a set of targets.
From the targets, it automatically derives whether the evaluation is in-domain or cross-domain.
The targets used in the original benchmark paper can be obtained [here](https://dcache-demo.desy.de:2443/Helmholtz/HelmholtzAI/SeisBench/auxiliary/pick-benchmark/targets/).
For details on the use of targets for evaluation, check the benchmark paper.

The evaluation will write predictions for all targets into the folder `pred/[config]` (for in-domain evaluation) or `pred/[config]_[target]` (for cross-domain evaluation).

## Collecting and visualizing results

The script `collect_results.py` aggregates results from multiple experiments within a folder and calculates summary statistics.
It outputs the performance metrics into once csv file.
The script will automatically select optimal decision thresholds on the development set.
These thresholds are also reported in the results csv.
Further command line parameters can be used to indicate which type of experiments, e.g., in-domain or cross-domain, are collected.
This information is required to correctly parse the config names.

The file `results_summary.py` offers a variety of functions to generate tables and figures representing the results.
However, while the basic functions are implemented in a relatively general fashion, the script has been particularly designed for the benchmark paper.
At the moment, we do not have the capacity to transform it into a generally applicable script.
Therefore, when adding new datasets or models, it will likely require some modifications on this file.
For further questions, please raise an issue.

In a similar spirit, we proved several Jupyter notebooks to visualize data or predictions.
Those are not actively maintained, but we hope they can serve as inspiration.

# Adding a dataset

To add a new dataset, the following steps are required:
- Write a new config. The dataset can be referenced through its path.
- Generate evaluation targets using the script `generate_eval_targets.py`.

Please consider [adding your dataset to SeisBench](https://github.com/seisbench/seisbench/blob/main/CONTRIBUTING.md).

# Adding a model

Benchmark models are [pytorch lightning modules](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).
We created a derived class `SeisBenchModuleLit` that adds a set of abstract methods required for the benchmark.
To add a new model, create a class inheriting from `SeisBenchModuleLit`.
Then you'll need to implement the following:

- The `__init__` method. Make sure to include a call to the super constructor and to `save_hyperparameters`.
- The functionality for pytorch lightning (`forward`, `training_step`, `validation_step`, `configure_optimizer`).
- The augmentations through `get_augmentations`. If required, you can specify separate augmentations for training and evaluation.
- The predict step, translating the model predictions into the format required by the benchmark. The exact specification is given in the docstring.

Now you'll only have to write a config file and can start training and evaluating the model.

# Reference publication

When using code from this repository for your work, please reference the following publication:

[Which picker fits my data? A quantitative evaluation of deep learning based seismic pickers](https://doi.org/10.1029/2021JB023499)
