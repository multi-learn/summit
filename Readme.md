# Benchmark de classification mono et multi-vue

This project aims to be an easy-to use solution to run a prior benchmark on a dataset abd evaluate mono- and multi-view algorithms capacity to classify it correctly.

## Getting Started

In order to run it you'll need to try on simulated data with the command
```
python multiview-machine-learning-omis/Code/MonoMultiViewClassifiers/ExecClassif.py -log
```
Results will be stored in multiview-machine-learning-omis/Code/MonoMultiViewClassifiers/Results/

### Prerequisites

To be able to use this project, you'll need :

* [Python 2.7](http://www.dropwizard.io/1.0.2/docs/) - The web framework used

And the following python modules :
* [pyscm](https://github.com/aldro61/pyscm) - Set Covering Machine, Marchand, M., & Taylor, J. S. (2003) by A.Drouin, F.Brochu, G.Letarte St-Pierre, M.Osseni, P-L.Plante
* [numpy](http://www.numpy.org/), [scipy](https://scipy.org/)
* [matplotlib](http://matplotlib.org/) - Used to plot results
* [sklearn](http://scikit-learn.org/stable/) - Used for the monoview classifiers
* [joblib](https://pypi.python.org/pypi/joblib) - Used to compute on multiple threads
* [h5py](www.h5py.org) - Used to generate HDF5 datasets on hard drive and use them to sapre RAM
* ([argparse](https://docs.python.org/3/library/argparse.html) - Used to parse the input args)
* ([logging](https://docs.python.org/2/library/logging.html) - Used to generate log)

They are all tested in  `multiview-machine-mearning-omis/Code/MonoMutliViewClassifiers/Versions.py` which is automatically checked each time you run the `ExecClassif` script

### Installing

No installation is needed, just the prerequisites.

## Running the tests

In order to run it you'll need to try on simulated data with the command
```
python multiview-machine-learning-omis/Code/MonoMultiViewClassifiers/ExecClassif.py -log
```
Results will be stored in multiview-machine-learning-omis/Code/MonoMultiViewClassifiers/Results/

## Authors

* **Baptiste BAUVIN** 
