[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.com/babau1/multiview-machine-learning-omis.svg?token=pjoowx3poApRRtwqHTpd&branch=master)](https://travis-ci.com/babau1/multiview-machine-learning-omis)
# Mono- and Multi-view classification benchmark

This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.

## Getting Started

### Prerequisites

To be able to use this project, you'll need :

* [Python 2.7](https://docs.python.org/2/) or [Python 3](https://docs.python.org/3/) 

And the following python modules :
* [pyscm](https://github.com/aldro61/pyscm) - Set Covering Machine, Marchand, M., & Taylor, J. S. (2003) by A.Drouin, F.Brochu, G.Letarte St-Pierre, M.Osseni, P-L.Plante
* [numpy](http://www.numpy.org/), [scipy](https://scipy.org/)
* [matplotlib](http://matplotlib.org/) - Used to plot results
* [sklearn](http://scikit-learn.org/stable/) - Used for the monoview classifiers
* [joblib](https://pypi.python.org/pypi/joblib) - Used to compute on multiple threads
* [h5py](www.h5py.org) - Used to generate HDF5 datasets on hard drive and use them to spare RAM
* [pickle](https://docs.python.org/3/library/pickle.html) - Used to store some results
* ([graphviz](https://pypi.python.org/pypi/graphviz) - Used for decision tree interpretation)


They are all tested in  `multiview-machine-mearning-omis/Code/MonoMutliViewClassifiers/Versions.py` which is automatically checked each time you run the `Exec` script

### Installing

No installation is needed, just the prerequisites.

### Running on simulated data

In order to run it you'll need to try on **simulated** data with the command
```
cd multiview-machine-learning-omis/Code
python Exec.py -log
```
Results will be stored in `multiview-machine-learning-omis/Code/MonoMultiViewClassifiers/Results/`

If you want to run a multiclass (one versus one) benchmark on simulated data, use : 
```
cd multiview-machine-learning-omis/Code
python Exec.py -log --CL_nbClass 3
```

If no path is specified, simulated hdf5 datasets are stored in `multiview-machine-learning-omis/Data`


### Discovering the arguments

In order to see all the arguments of this script, their description and default values run :
```
cd multiview-machine-learning-omis/Code
python Exec.py -h
```


### Understanding `Results/` architecture

Results are stored in `multiview-machine-learning-omis/Code/MonoMultiViewClassifiers/Results/`
A directory will be created with the name of the database used to run the script.
For each time the script is run, a new directory named after the running date and time will be created.
In that directory:
* If the script is run using more than one statistic iteration (one for each seed), it will create one directory for each iteration and store the statistical analysis in the current directory 
* If it is run with one iteration, the iteration results will be stored in the current directory

The results for each iteration are graphs plotting the classifiers scores and the classifiers config and results are stored in a directory of their own.
To explore the results run the `Exec` script and go in `multiview-machine-learning-omis/Code/MonoMultiViewClassifiers/Results/Plausible/`

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
* `attrs["name"]` a boolean specifying whether the view is sparse or not

One dataset for the labels called `Labels` with one attribute : 
* `attrs["names"]` a list of strings encoded in utf-8 namig the labels in the right order

One group for the additional data called `Metadata` containing at least 3 attributes : 
* `attrs["nbView"]` an int counting the total number of views in the dataset
* `attrs["nbClass"]` an int counting the total number of different labels in the dataset
* `attrs["datasetLength"]` an int counting the total number of examples in the dataset


### Running on your dataset 

In order to run the script on your dataset you need to use : 
```
cd multiview-machine-learning-omis/Code
python Exec.py -log --name <your_dataset_name> --type <.cvs_or_.hdf5> --pathF <path_to_your_dataset>
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