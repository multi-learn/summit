=====================================
Taking control : Use your own dataset
=====================================

.. role:: python(code)
    :language: python

While developping this platform, the goal has been to bea able to use it relatively easily on different datasets.
In order to do so, a fixed input format is used, and we choosed HDF5 as it allows to store a multiview dataset and its metadata in a single file, while being able to load it partially.

The bare necessities
--------------------

At the moment, in order for the platfro to work, the dataset must satisfy the following minimum requirements :

- Each example must be described in each view, with no missing data (you can use external tools to fill the gaps, or use only the fully-described examples of you dataset)
- ?

The dataset structure
---------------------

Let's suppose that one has a multiview dataset consisting of 3 views describing 200 examples:

1. A sound recoding of each example, described by 100 features,
2. An image of each example, described by 40 features,
3. A written commentary for each example, described by 55 features.

So three matrices (200x100 ; 200x40 ; 200x55) make up the dataset. THe most usual way to save matrices are `.csv` files. So let's suppose that one has

1. ``sound.csv``,
2. ``image.csv``
3. ``commentary.csv``.

LEt's suppose that all this data should be used to classify the examples in two classes : Animal or Object and that on has a ``labels.csv`` file wit one value for each example, a 0 if the example is an Animal and a 1 if it is an Object.

In order to run a benchmark on this dataset, one has to format it using HDF5.

HDF5 conversion
---------------

We will use here a python script to convert the dataset in the right format :

.. code-block:: python

    import h5py
    import numpy as np

Let's load the csv matrices :

.. code-block:: python

    sound_matrix = np.genfromtxt("path/to/sound.csv", delimiter=",")
    image_matrix = np.genfromtxt("path/to/image.csv", delimiter=",")
    commentary_matrix = np.genfromtxt("path/to/commentary.csv", delimiter=",")
    labels = np.genfromtxt("path/to/labels.csv", delimiter=",")

Let's create the HDF5 file :

.. code-block:: python

    hdf5_file = h5py.File("path/to/database_name.hdf5", "w")

Now, let's create the first dataset : the one with the sound view :

.. code-block:: python

    sound_dataset = hdf5_file.create_dataset("View0", data=sound_matrix)
    sound_dataset.attrs["name"] = "Sound"
    sound_dataset.attrs["sparse"] = "False"

**Be sure to use View0 as the name of the dataset**, as it is mandatory for the platform to run (the following datasets will be named :python:`"View1"`, :python:`"View2"`, ...)

For each view available, let's create a dataset similarly (be sure that the examples are described in the same order : line 1 of the sound matrix describes the same example as line 1 of the image one and the commentary one)

.. code-block:: python

    image_dataset = hdf5_file.create_dataset("View1", data=image_matrix)
    image_dataset.attrs["name"] = "Image"
    image_dataset.attrs["sparse"] = "False"

    commentary_dataset = hdf5_file.create_dataset("View2", data=commentary_matrix)
    commentary_dataset.attrs["name"] = "Commentary"
    commentary_dataset.attrs["sparse"] = "False"

Let's now create the labels dataset (here also, be sure that the labels are correctly ordered).

.. code-block:: python

    labels_dataset = hdf5_file.create_dataset("Labels", data=labels)
    labels_dataset.attrs["name"] = ["Animal".encode(), "Object".encode()]

Be sure to sort the label names in the right order (the label must be the same as the list's index, here 0 is Animal, and also :python:`labels_dataset.attrs["name"][0]`)

Let's now store the metadata :

.. code-block:: python

    metadata_group = hdf5_file.create_group("Metadata")
    metadata_group.attrs["nbView"] = 3
    metadata_group.attrs["nbClass"] = 2
    metadata_group.attrs["datasetLength"] = 200

Here, we store

- The number of views in the :python:`"nbView"` attribute,
- The number of different labels in the :python:`"nbClass"` attribute,
- The number of examples in the :python:`"datasetLength"` attribute.

Now, the dataset is ready to be used in the platform.
Let's suppose it is stored in ``path/to/database_name.hdf5``, then by setting the ``pathf:`` line of the config file to
``pathf: path/to/`` and the ``name:`` line to ``name: ["database_name.hdf5"]``, the benchmark will run on the created dataset.


Adding additional information on the examples
---------------------------------------------

In order to be able to analyze the results with more clarity, one can add the examples IDs to the dataset, by adding a dataset to the metadata group.

Let's suppose that the objects we are trying to classify between 'Animal' and 'Object' are all bears, cars, planes, and birds. And that one has a ``.csv`` file with an ID for each of them (:python:`"bear_112", "plane_452", "bird_785", "car_369", ...` for example)

Then as long as the IDs order corresponds to the example order in the lines of the previous matrices, to add the IDs in the hdf5 file, just add :

.. code-block:: python

    id_table = np.genfromtxt("path.to/id.csv", delimiter=",").astype(np.dtype('S10'))
    metadata_group.create_dataset("example_ids", data=id_table, dtype=np.dtype('S10'))

Be sure to keep the name :python:`"example_ids"`, as it is mandatory for the platform to find the IDs dataset in the file.


