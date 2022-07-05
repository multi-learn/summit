=====================================
Taking control : Use your own dataset
=====================================

While developing this platform, the goal has been to be able to use it relatively easily on different datasets.
In order to do so, a fixed input format is used, and we chose HDF5 as it allows to store a multiview dataset and its metadata in a single file, while being able to load it partially.

The bare necessities
--------------------

At the moment, in order for the platform to work, the dataset must satisfy the following minimum requirements :

- Each sample must be described in each view, with no missing data (you can use external tools to fill the gaps, or use only the fully-described samples of your dataset)

The dataset structure
---------------------

Let's suppose that one has a multiview dataset consisting of 3 views describing 200 samples:

1. A sound recoding of each sample, described by 100 features,
2. An image of each sample, described by 40 features,
3. A written commentary for each sample, described by 55 features.

So three matrices (200x100 ; 200x40 ; 200x55) make up the dataset. The most usual way to save matrices are `.csv` files. So let us suppose that one has

1. ``sound.csv``,
2. ``image.csv``
3. ``commentary.csv``.

Let us suppose that all this data should be used to classify the examples in three classes : "Human", "Animal" or "Object"  and that on has a ``labels.csv`` file with one value for each sample, 0 if the sample is a human, 1 if it is an animal an 2 if it is an object.

In order to run a benchmark on this dataset, one has to format it using HDF5.

HDF5 conversion
---------------

We will use here a :base_source:`python script <format_dataset.py>`, provided with the platform to convert the dataset in the right format :

.. code-block:: python

    import h5py
    import numpy as np

Let's define the variables that will be used to load the csv matrices :

.. code-block:: python

    # The following variables are defined as an example, you should modify them to fit your dataset files.
    view_names = ["sound", "image", "commentary", ]
    data_file_paths = ["path/to/sound.csv", "path/to/image.csv", "path/to/commentary.csv",]
    labels_file_path = "path/to/labels/file.csv"
    sample_ids_path = "path/to/sample_ids/file.csv"
    labels_names = ["Human", "Animal", "Object"]

Let's create the HDF5 file :

.. code-block:: python

    # HDF5 dataset initialization :
    hdf5_file = h5py.File("path/to/file.hdf5", "w")

Now, for each view, create an HDF5 dataset :

.. code-block:: python

    for view_index, (file_path, view_name) in enumerate(zip(data_file_paths, view_names)):
        # Get the view's data from the csv file
        view_data = np.genfromtxt(file_path, delimiter=",")

        # Store it in a dataset in the hdf5 file,
        # do not modify the name of the dataset
        view_dataset = hdf5_file.create_dataset(name="View{}".format(view_index),
                                                shape=view_data.shape,
                                                data=view_data)
        # Store the name of the view in an attribute,
        # do not modify the attribute's key
        view_dataset.attrs["name"] = view_name

        # This is an artifact of work in progress for sparse support, not available ATM,
        # do not modify the attribute's key
        view_dataset.attrs["sparse"] = False

Let's now create the labels dataset (here also, be sure that the labels are correctly ordered).

.. code-block:: python

    # Get le labels data from a csv file
    labels_data = np.genfromtxt(labels_file_path, delimiter=',')

    # Here, we supposed that the labels file contained numerical labels (0,1,2)
    # that refer to the label names of label_names.
    # The Labels HDF5 dataset must contain only integers that represent the
    # different classes, the names of each class are saved in an attribute

    # Store the integer labels in the HDF5 dataset,
    # do not modify the name of the dataset
    labels_dset = hdf5_file.create_dataset(name="Labels",
                                           shape=labels_data.shape,
                                           data=labels_data)
    # Save the labels names in an attribute as encoded strings,
    # do not modify the attribute's key
    labels_dset.attrs["names"] = [label_name.encode() for label_name in labels_names]

Be sure to sort the label names in the right order (the label must be the same as the list's index, here 0 is "Human", and also :python:`labels_dataset.attrs["name"][0]`)

Let's now store the metadata :

.. code-block:: python

    # Create a Metadata HDF5 group to store the metadata,
    # do not modify the name of the group
    metadata_group = hdf5_file.create_group(name="Metadata")

    # Store the number of views in the dataset,
    # do not modify the attribute's key
    metadata_group.attrs["nbView"] = len(view_names)

    # Store the number of classes in the dataset,
    # do not modify the attribute's key
    metadata_group.attrs["nbClass"] = np.unique(labels_data)

    # Store the number of samples in the dataset,
    # do not modify the attribute's key
    metadata_group.attrs["datasetLength"] = labels_data.shape[0]

Here, we store

- The number of views in the :python:`"nbView"` attribute,
- The number of different labels in the :python:`"nbClass"` attribute,
- The number of samples in the :python:`"datasetLength"` attribute.

Now, the dataset is ready to be used in the platform.
Let's suppose it is stored in ``path/to/file.hdf5``, then by setting the ``pathf:`` line of the config file to
``pathf: path/to/`` and the ``name:`` line to ``name: ["file.hdf5"]``, the benchmark will run on the created dataset.


Adding additional information on the samples
---------------------------------------------

In order to be able to analyze the results with more clarity, one can add the samples IDs to the dataset, by adding a dataset to the metadata group.

Let's suppose that the objects we are trying to classify between "Human", "Animal" and "Object" are all people, bears, cars, planes, and birds. And that one has a ``.csv`` file with an ID for each of them (:python:`"john_115", "doe_562", "bear_112", "plane_452", "bird_785", "car_369", ...` for example)

Then as long as the IDs order corresponds to the sample order in the lines of the previous matrices, to add the IDs in the hdf5 file, just add :

.. code-block:: python

    # Let us suppose that the samples have string ids, available in a csv file,
    # they can be stored in the HDF5 and will be used in the result analysis.
    sample_ids = np.genfromtxt(sample_ids_path, delimiter=',')

    # To sore the strings in an HDF5 dataset, be sure to use the S<max_length> type,
    # do not modify the name of the dataset.
    metadata_group.create_dataset("sample_ids",
                                  data=np.array(sample_ids).astype(np.dtype("S100")),
                                  dtype=np.dtype("S100"))


Be sure to keep the name :python:`"sample_ids"`, as it is mandatory for the platform to find the dataset in the file.


