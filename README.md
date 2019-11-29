[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://gitlab.lis-lab.fr/baptiste.bauvin/multiview-machine-learning-omis/badges/develop/build.svg)](https://gitlab.lis-lab.fr/baptiste.bauvin/multiview-machine-learning-omis/badges/develop/build.svg)
# Mono- and Multi-view classification benchmark

This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.

## Getting Started

### Prerequisites

To be able to use this project, you'll need :

* [Python 3.6](https://docs.python.org/3/) 

And the following python modules :

* [numpy](http://www.numpy.org/), [scipy](https://scipy.org/),
* [matplotlib](http://matplotlib.org/) - Used to plot results,
* [sklearn](http://scikit-learn.org/stable/) - Used for the monoview classifiers,
* [joblib](https://pypi.python.org/pypi/joblib) - Used to compute on multiple threads,
* [h5py](https://www.h5py.org) - Used to generate HDF5 datasets on hard drive and use them to spare RAM,
* [pickle](https://docs.python.org/3/library/pickle.html) - Used to store some results,
* [pandas](https://pandas.pydata.org/) - Used to manipulate data efficiently,
* [six](https://pypi.org/project/six/) - 
* [m2r](https://pypi.org/project/m2r/) - Used to generate documentation from the readme,
* [docutils](https://pypi.org/project/docutils/) - Used to generate documentation,
* [pyyaml](https://pypi.org/project/PyYAML/) - Used to read the config files,
* [plotly](https://plot.ly/) - Used to generate interactive HTML visuals.

They are all tested in  `multiview-machine-mearning-omis/multiview_platform/MonoMutliViewClassifiers/Versions.py` which is automatically checked each time you run the `execute` script

### Installing

Once you cloned the project from this repository, you just have to use :  

```
pip install -e .
```
In the `multiview_machine-learning-omis` directory.

### Running on simulated data

In order to run it you'll need to try on **simulated** data with the command
```python 
from multiview_platform.execute import execute
execute()
```
Results will be stored in the results directory of the installation path : `path/to/install/multiview-machine-learning-omis/results`.

And simulated hdf5 datasets are stored in `path/to/install/multiview-machine-learning-omis/data`


### Discovering the arguments

In order to see all the arguments of this script, their description and default values run :
```
cd multiview-machine-learning-omis/multiview_platform
python execute.py -h
```
The arguments can be passed through a file using `python Exec.py @<path_to_doc>`
The file must be formatted with one newline instead of each space : 
Command line arguments `-debug --CL_type Monoview --CL_algos_monoview Adaboost SVM` will be formatted 
```
-debug
--CL_type
Monoview
--CL_algos_monoview
Adaboost
SVM
```

Moreover, for Monoview algorithms (Multiview is still WIP), it is possible to pass multiple arguments instead of just one. 
Thus, executing `python execute.py --RF_trees 10 100 --RF_max_depth 3 4 --RF_criterion entropy` will result in the generation of several classifiers called
`RandomForest_10_3_entropy`, with 10 trees and a max depth of 3, `RandomForest_10_4_entropy`, with 10 tress and a max depth of 4,  `RandomForest_100_3_entropy`, `RandomForest_100_4_entropy` to test all the passed arguments combinations. 
 


### Understanding `results/` architecture

Results are stored in `multiview-machine-learning-omis/multiview_platform/mono_multi_view_classifiers/results/`
A directory will be created with the name of the database used to run the script.
For each time the script is run, a new directory named after the running date and time will be created.
In that directory:
* If the script is run using more than one statistic iteration (one for each seed), it will create one directory for each iteration and store the statistical analysis in the current directory 
* If it is run with one iteration, the iteration results will be stored in the current directory

The results for each iteration are graphs plotting the classifiers scores and the classifiers config and results are stored in a directory of their own.
To explore the results run the `execute` script and go in `multiview-machine-learning-omis/multiview_platform/mono_multi_view_classifiers/results/plausible/`

### Dataset compatibility

In order to start a benchmark on your dataset, you need to format it so the script can use it. 
You can have either a directory containing `.csv` files or a HDF5 file. 

##### If you have multiple `.csv` files, you must organize them as : 
* `top_directory/database_name-labels.csv`
* `top_directory/database_name-labels-names.csv`
* `top_directory/Views/view_name.csv` or `top_directory/Views/view_name-s.csv` if the view is sparse

With `top_directory` being the last directory in the `pathF` argument
 
##### If you already have an HDF5 dataset file it must be formatted as : 
One dataset for each view called `ViewX` with `X` being the view index with 2 attribures : 
* `attrs["name"]` a string for the name of the view
* `attrs["sparse"]` a boolean specifying whether the view is sparse or not
* `attrs["ranges"]` a `np.array` containing the ranges of each attribute in the view (for ex. : for a pixel the range will be 255, for a real attribute in [-1,1], the range will be 2).
* `attrs["limits"]` a `np.array` containing all the limits of the attributes int he view. (for ex. : for a pixel the limits will be `[0, 255]`, for a real attribute in [-1,1], the limits will be `[-1,1]`).
 

One dataset for the labels called `Labels` with one attribute : 
* `attrs["names"]` a list of strings encoded in utf-8 namig the labels in the right order

One group for the additional data called `Metadata` containing at least 3 attributes : 
* `attrs["nbView"]` an int counting the total number of views in the dataset
* `attrs["nbClass"]` an int counting the total number of different labels in the dataset
* `attrs["datasetLength"]` an int counting the total number of examples in the dataset


### Running on your dataset 

In order to run the script on your dataset you need to use : 
```
cd multiview-machine-learning-omis/multiview_platform
python execute.py -log --name <your_dataset_name> --type <.cvs_or_.hdf5> --pathF <path_to_your_dataset>
```
This will run a full benchmark on your dataset using all available views and labels.
 
You may configure the `--CL_statsiter`, `--CL_split`, `--CL_nbFolds`, `--CL_GS_iter` arguments to start a meaningful benchmark
 

## Running the tests

**/!\ still in development, test sucess is not meaningful ATM /!\\**

In order to run it you'll need to try on simulated data with the command
```
cd multiview-machine-learning-omis/
python -m unittest discover
```

## Author

* **Baptiste BAUVIN**

### Contributors

* **Mazid Osseni**
* **Alexandre Drouin**
* **Nikolas Huelsmann**
