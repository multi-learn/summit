"""In ths file, we explain how to add a metric to the platform.

In order to do that, on needs to add a file with the following functions
which are mandatory for the metric to work with the platform.
"""
import warnings

warnings.warn("the framework module  is deprecated", DeprecationWarning,
              stacklevel=2)
# Author-Info
__author__ = "Baptiste Bauvin"
__status__ = "Prototype"  # Production, Development, Prototype


def score(y_true, y_pred, multiclass=False, **kwargs):
    """Get the metric's score from the ground truth (``y_true``) and predictions (``y_pred``).

   Parameters
   ----------
   y_true : array-like, shape = (n_samples,)
            Target values (class labels).

   y_pred : array-like, shape = (n_samples,)
            Predicted target values (class labels).

   multiclass : boolean (default=False)
                Parameter specifying whether the target values are multiclass or not.

   kwargs : dict
            The arguments stored in this dictionary must be keyed by string of
            integers as "0", .., etc and decrypted in the function

   Returns
   -------
   score : float
           Returns the score of the prediction.
    """
    score = 0.0
    return score


def get_scorer(**kwargs):
    """Get the metric's scorer as in the sklearn.metrics package.

   Parameters
   ----------
   kwargs : dict
           The arguments stored in this dictionary must be keyed by string of
           integers as "0", .., etc and decrypted in the function. These arguments
           are a configuration of the metric.

   Returns
   -------
   scorer : object
           Callable object that returns a scalar score; greater is better. (cf sklearn.metrics.make_scorer)
    """
    scorer = None
    return scorer


def get_config(**kwargs):
    """Get the metric's configuration as a string.

   Parameters
   ----------
   kwargs : dict
           The arguments stored in this dictionary must be keyed by string of
           integers as "0", .., etc and decrypted in the function. These arguments
           are a configuration of the metric.

   Returns
   -------
   configString : string
           The string describing the metric's configuration.
    """

    config_tring = "This is a framework"
    return config_tring
