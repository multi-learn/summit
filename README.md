[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://gitlab.lis-lab.fr/baptiste.bauvin/summit/badges/develop/pipeline.svg)](https://gitlab.lis-lab.fr/baptiste.bauvin/summit/badges/develop/pipeline.svg)
# Supervised MultiModal Integration Tool's Readme

This project aims to be an easy-to-use solution to run a prior benchmark on a dataset and evaluate mono- & multi-view algorithms capacity to classify it correctly.

## Getting Started

### Prerequisites (will be automatically installed)

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

Once you cloned the project from the [gitlab repository](https://gitlab.lis-lab.fr/baptiste.bauvin/summit/), you just have to use :  

```
cd path/to/summit/
pip install -e .
```
In the `summit` directory to install SuMMIT and its dependencies.

### Running on simulated data

In order to run it you'll need to try on **simulated** data with the command
```python 
from multiview_platform.execute import execute
execute("example 1")
```
This will run the first example. 

For more information about the examples, see the [documentation](http://baptiste.bauvin.pages.lis-lab.fr/summit/).
Results will be stored in the results directory of the installation path : 
`path/to/summit/multiview_platform/examples/results`.

The documentation proposes a detailed interpretation of the results through [6 tutorials](http://baptiste.bauvin.pages.lis-lab.fr/summit/). 

### Discovering the arguments

All the arguments of the platform are stored in a YAML config file. Some config files are given as examples. 
The file stored in `summit/config_files/config.yml` is documented and it is highly recommended
to read it carefully before playing around with the parameters.   

You can create your own configuration file. In order to run the platform with it, run : 
```python
from multiview_platform.execute import execute
execute(config_path="/absolute/path/to/your/config/file")
```

For further information about classifier-specific arguments, see the [documentation](http://baptiste.bauvin.pages.lis-lab.fr/summit/). 
 

### Dataset compatibility


In order to start a benchmark on your own dataset, you need to format it so SuMMIT can use it. To do so, a [python script](https://gitlab.lis-lab.fr/baptiste.bauvin/summit/-/blob/master/format_dataset.py) is provided.

For more information, see [Example 6](http://baptiste.bauvin.pages.lis-lab.fr/summit/tutorials/example4.html)

### Running on your dataset 

Once you have formatted your dataset, to run SuMMIT on it you need to modify the config file as  
```yaml
name: ["your_file_name"]
*
pathf: "path/to/your/dataset"
```
This will run a full benchmark on your dataset using all available views and labels.
 
It is highly recommended to follow the documentation's [tutorials](http://baptiste.bauvin.pages.lis-lab.fr/summit/tutorials/index.html) to learn the use of each parameter. 
 

## Author

* **Baptiste BAUVIN**
* **Dominique BENIELLI**
* **Alexis PROD'HOMME**

