# Class-wise-and-reduced-calibration-methods

This repository contains the experiment for the "class-wise and reduced calibration methods" paper submitted to ICMLA 2022

# Pre-requisites

This project uses [Poetry](https://python-poetry.org/) for dependency management.

Start by installing that and then proceed to installing the requirements:

```shell
poetry install
```

# Running the Experiments

To run the experiments use:

```shell
poetry run python -m src.experiments.<Experiment Module>
```

Where you would replace <Experiment Module> with the name of one of experiments' module.

For example, to run the Random Forest experiment with Synthetic Data you should use:

```shell
poetry run python -m src.experiments.random_forest_synthetic
```

# Notebooks

The notebooks are generated from the experiment scripts as follows:

```shell
bash scripts/generate_notebooks.sh
```

| Experiment                          | MyBinder                                                                                                                                      |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| [Random Forest on Synthetic Data](notebooks/randomforest_synthetic.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/appliedAI-Initiative/Class-wise-and-reduced-calibration-methods/main?labpath=notebooks%2Frandomforest_synthetic) |
