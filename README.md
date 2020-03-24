[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://gitlab.lis-lab.fr/baptiste.bauvin/multiview-machine-learning-omis/badges/develop/pipeline.svg)](https://gitlab.lis-lab.fr/baptiste.bauvin/multiview-machine-learning-omis/badges/develop/pipeline.svg)
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
* [plotly](https://plot.ly/) - Used to generate interactive HTML visuals,
* [tabulate](https://pypi.org/project/tabulate/) - Used to generated the confusion matrix.


### Installing

Once you cloned the project from the [gitlab repository](https://gitlab.lis-lab.fr/baptiste.bauvin/multiview-machine-learning-omis/), you just have to use :  

```
pip install -e .
```
In the `multiview_machine-learning-omis` directory to install SuMMIT and its dependencies.

### Running on simulated data

In order to run it you'll need to try on **simulated** data with the command
```python 
from multiview_platform.execute import execute
execute()
```
This will run the first example. For more information about the examples, see the [documentation](http://baptiste.bauvin.pages.lis-lab.fr/multiview-machine-learning-omis/) 
Results will be stored in the results directory of the installation path : 
`path/to/install/multiview-machine-learning-omis/multiview_platform/examples/results`.
The documentation proposes a detailed interpretation of the results. 

### Discovering the arguments

All the arguments of the platform are stored in a YAML config file. Some config files are given as examples. 
The file stored in `multiview-machine-learning-omis/config_files/config.yml` is documented and it is highly recommended
to read it carefully before playing around with the parameters.   

You can create your own configuration file. In order to run the platform with it, run : 
```python
from multiview_platform.execute import execute
execute(config_path="/absolute/path/to/your/config/file")
```

For further information about classifier-specific arguments, see the [documentation](http://baptiste.bauvin.pages.lis-lab.fr/multiview-machine-learning-omis/). 
 

### Dataset compatibility

In order to start a benchmark on your own dataset, you need to format it so SuMMIT can use it. 

[comment]: <> (You can have either a directory containing `.csv` files or a HDF5 file.) 

[comment]: <> (#### If you have multiple `.csv` files, you must organize them as : 
* `top_directory/database_name-labels.csv`
* `top_directory/database_name-labels-names.csv`
* `top_directory/Views/view_name.csv` or `top_directory/Views/view_name-s.csv` if the view is sparse)

[comment]: <> (With `top_directory` being the last directory in the `pathF` argument)
 
##### If you already have an HDF5 dataset file it must be formatted as : 
One dataset for each view called `ViewI` with `I` being the view index with 2 attribures : 
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

The `format_dataset.py` file is documented and can be used to format a multiview dataset in a SuMMIT-compatible HDF5 file.

### Running on your dataset 

Once you have formatted your dataset, to run SuMMIT on it you need to modify the config file as  
```yaml
name: ["your_file_name"]
*
pathf: "path/to/your/dataset"
```
This will run a full benchmark on your dataset using all available views and labels.
 
It is highly recommended to follow the documentation's [tutorials](http://baptiste.bauvin.pages.lis-lab.fr/multiview-machine-learning-omis/tutorials/index.html) to learn the use of each parameter. 
 

## Author

* **Baptiste BAUVIN**

### Contributors

* **Dominique Benielli**
