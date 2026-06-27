Installation
============

Requirements
------------

- Python 3.11
- pip or conda

Conda environment (recommended)
-------------------------------

.. code-block:: bash

    git clone https://github.com/maiziezhoulab/FEAST
    cd FEAST
    conda env create -f environment.yml
    conda activate feast-py311-conda
    pip install --no-deps -r requirements.txt
    pip install --no-deps -e .

Source checkout
---------------

If you already have the source:

.. code-block:: bash

    cd FEAST
    conda env create -f environment.yml
    conda activate feast-py311-conda
    pip install --no-deps -r requirements.txt
    pip install --no-deps -e .
